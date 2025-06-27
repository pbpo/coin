#include "bert_components_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp" // 실제 커널 런처 함수들을 포함
#include <stdexcept>
#include <vector>
#include <string>

// ============================================================================
// BertEmbeddings 구현부
// ============================================================================

BertEmbeddings::BertEmbeddings(const BertConfig& config, const std::string& name_prefix)
    : config_(config),
      word_embeddings_({config.vocab_size, config.hidden_size}, name_prefix + ".word_embeddings"),
      position_embeddings_({config.max_position_embeddings, config.hidden_size}, name_prefix + ".position_embeddings"),
      token_type_embeddings_({2, config.hidden_size}, name_prefix + ".token_type_embeddings"), // 2개의 토큰 타입(e.g., A, B 문장) 가정
      layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".LayerNorm"),
      dropout_(config.hidden_dropout_prob) {}

std::vector<Parameter*> BertEmbeddings::get_parameters() {
    auto ln_params = layernorm_.get_parameters();
    std::vector<Parameter*> params = {&word_embeddings_, &position_embeddings_, &token_type_embeddings_};
    params.insert(params.end(), ln_params.begin(), ln_params.end());
    return params;
}

Parameter* BertEmbeddings::get_word_embedding_params() {
    return &word_embeddings_;
}

void BertEmbeddings::forward(hipStream_t stream,
                             const GpuTensor& input_ids, const GpuTensor& token_type_ids,
                             GpuTensor& output_embeddings, BertEmbeddingsCache& cache, bool is_training) {
    if (!input_ids.is_allocated() || !word_embeddings_.weights.is_allocated() ||
        !position_embeddings_.weights.is_allocated() || !token_type_embeddings_.weights.is_allocated()) {
        throw std::runtime_error("Required tensors for BertEmbeddings::forward are not allocated.");
    }
    if (token_type_ids.is_allocated() && (token_type_ids.dims_ != input_ids.dims_)) {
        throw std::runtime_error("input_ids and token_type_ids must have the same dimensions.");
    }

    const int batch_size = input_ids.dim_size(0);
    const int seq_len = input_ids.dim_size(1);
    const std::vector<int> expected_output_dims = {batch_size, seq_len, config_.hidden_size};

    if (!output_embeddings.is_allocated() || output_embeddings.dims_ != expected_output_dims) {
        output_embeddings.allocate(expected_output_dims);
    }
    cache.summed_embeddings.allocate(expected_output_dims);
    cache.embeddings_output.allocate(expected_output_dims);

    cache.input_ids_ptr = &input_ids;

    // token_type_ids가 제공되지 않은 경우, 0으로 채워진 더미 텐서를 사용
    GpuTensor dummy_token_type_ids;
    const GpuTensor* token_type_ids_to_use = &token_type_ids;
    if (!token_type_ids.is_allocated()) {
        dummy_token_type_ids.dtype = DataType::INT32;
        dummy_token_type_ids.allocate(input_ids.dims_);
        dummy_token_type_ids.zero_out(stream);
        token_type_ids_to_use = &dummy_token_type_ids;
    }
    cache.token_type_ids_ptr = token_type_ids_to_use;

    // 1. Word + Position + Token Type 임베딩을 합산
    launch_add_embeddings_kernel(stream,
                                 (float*)cache.summed_embeddings.d_ptr_,
                                 (const int*)input_ids.d_ptr_,
                                 (const int*)token_type_ids_to_use->d_ptr_,
                                 (const float*)word_embeddings_.weights.d_ptr_,
                                 (const float*)position_embeddings_.weights.d_ptr_,
                                 (const float*)token_type_embeddings_.weights.d_ptr_,
                                 batch_size, seq_len, config_.hidden_size,
                                 config_.vocab_size, config_.max_position_embeddings);

    // 2. Layer Normalization 적용
    layernorm_.forward(stream, cache.summed_embeddings, cache.embeddings_output, cache.layernorm_cache);

    // 3. Dropout 적용
    dropout_.forward(stream, cache.embeddings_output, cache.dropout_cache, is_training);

    // 최종 결과를 출력 텐서로 복사
    output_embeddings.copy_from_gpu(cache.embeddings_output, stream);
}

void BertEmbeddings::backward(hipStream_t stream,
                              const GpuTensor& grad_output_embeddings,
                              BertEmbeddingsCache& cache) {
    if (!grad_output_embeddings.is_allocated() || !cache.input_ids_ptr || !cache.summed_embeddings.is_allocated()) {
        throw std::runtime_error("Required tensors/cache not ready for BertEmbeddings::backward.");
    }

    // 파라미터들의 그래디언트 텐서 할당
    word_embeddings_.allocate_gradients();
    position_embeddings_.allocate_gradients();
    token_type_embeddings_.allocate_gradients();
    layernorm_.params.allocate_gradients();

    // 역전파 순서: Dropout -> LayerNorm -> 임베딩 합산
    GpuTensor grad_after_dropout, grad_after_layernorm;
    grad_after_dropout.allocate(grad_output_embeddings.dims());
    grad_after_layernorm.allocate(grad_output_embeddings.dims());

    dropout_.backward(stream, grad_after_dropout, grad_output_embeddings, cache.dropout_cache);
    layernorm_.backward(stream, grad_after_dropout, cache.layernorm_cache, grad_after_layernorm);

    // 합산된 임베딩의 그래디언트를 각 임베딩 테이블(Word, Position, Token Type)에 분배 (Scatter-add)
    launch_embedding_backward_kernel(stream,
                                     (float*)word_embeddings_.grad_weights.d_ptr_,
                                     (float*)position_embeddings_.grad_weights.d_ptr_,
                                     (float*)token_type_embeddings_.grad_weights.d_ptr_,
                                     (const float*)grad_after_layernorm.d_ptr_,
                                     (const int*)cache.input_ids_ptr->d_ptr_,
                                     (const int*)cache.token_type_ids_ptr->d_ptr_,
                                     cache.input_ids_ptr->dim_size(0),
                                     cache.input_ids_ptr->dim_size(1),
                                     config_.hidden_size);
}

// ============================================================================
// BertLayer 구현부
// ============================================================================

BertLayer::BertLayer(const BertConfig& config, const std::string& name_prefix)
    : attention_(config, name_prefix),
      ffn_intermediate_dense_(config.hidden_size, config.intermediate_size, config, name_prefix + ".intermediate.dense"),
      ffn_output_dense_(config.intermediate_size, config.hidden_size, config, name_prefix + ".output.dense", true), // BertOutput은 활성화 함수가 없음
      ffn_output_dropout_(config.hidden_dropout_prob),
      ffn_output_layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".output.LayerNorm")
      {}

std::vector<Parameter*> BertLayer::get_parameters() {
    auto attention_params = attention_.get_parameters();
    auto ffn_int_params = ffn_intermediate_dense_.get_parameters();
    auto ffn_out_params = ffn_output_dense_.get_parameters();
    auto ffn_ln_params = ffn_output_layernorm_.get_parameters();

    std::vector<Parameter*> params;
    params.insert(params.end(), attention_params.begin(), attention_params.end());
    params.insert(params.end(), ffn_int_params.begin(), ffn_int_params.end());
    params.insert(params.end(), ffn_out_params.begin(), ffn_out_params.end());
    params.insert(params.end(), ffn_ln_params.begin(), ffn_ln_params.end());
    return params;
}

void BertLayer::forward(rocblas_handle blas_handle, hipStream_t stream,
                        const GpuTensor& input_hidden_states,
                        const GpuTensor& attention_mask,
                        GpuTensor& output_hidden_states,
                        BertLayerCache& cache,
                        bool is_training) {
    if (!input_hidden_states.is_allocated()) {
        throw std::runtime_error("Input hidden_states not allocated for BertLayer forward.");
    }
    if(!output_hidden_states.is_allocated() || output_hidden_states.dims_ != input_hidden_states.dims_){
        output_hidden_states.allocate(input_hidden_states.dims_);
    }
    cache.layer_input_ptr = &input_hidden_states;

    // --- 1. 어텐션 블록 (Self-Attention + Residual + LayerNorm) ---
    // BertAttention의 forward는 내부적으로 self-attention, residual, layernorm을 모두 수행
    // 그 결과는 cache.attention_cache.self_attention_cache.context_layer에 저장됨 (필드 재사용)
    attention_.forward(blas_handle, stream, input_hidden_states, attention_mask, cache.attention_cache, is_training);
    
    // 어텐션 블록의 최종 출력을 FFN의 입력으로 사용하기 위해 복사
    cache.ffn_input_after_attention.allocate(input_hidden_states.dims_);
    cache.ffn_input_after_attention.copy_from_gpu(cache.attention_cache.self_attention_cache.context_layer, stream);

    // --- 2. FFN (Feed-Forward Network) ---
    // FFN Intermediate Layer (Dense + GELU)
    GpuTensor intermediate_output;
    intermediate_output.allocate({input_hidden_states.dim_size(0), input_hidden_states.dim_size(1), config_.intermediate_size});
    ffn_intermediate_dense_.forward(blas_handle, stream, cache.ffn_input_after_attention, intermediate_output, cache.ffn_intermediate_dense_cache);

    // FFN Output Layer (Dense + Dropout)
    GpuTensor ffn_output_dense_result;
    ffn_output_dense_result.allocate(input_hidden_states.dims_);
    ffn_output_dense_.forward(blas_handle, stream, intermediate_output, ffn_output_dense_result, cache.ffn_output_dense_cache);
    ffn_output_dropout_.forward(stream, ffn_output_dense_result, cache.ffn_output_dropout_cache, is_training);

    // FFN 잔차 연결
    GpuTensor ffn_residual_sum_output;
    ffn_residual_sum_output.allocate(ffn_output_dense_result.dims_);
    launch_elementwise_add_kernel(stream, (float*)ffn_residual_sum_output.d_ptr(), (const float*)ffn_output_dense_result.d_ptr(), (const float*)cache.ffn_input_after_attention.d_ptr(), ffn_residual_sum_output.num_elements_);

    // FFN 최종 Layer Normalization
    ffn_output_layernorm_.forward(stream, ffn_residual_sum_output, output_hidden_states, cache.ffn_output_layernorm_cache);
}

void BertLayer::backward(rocblas_handle blas_handle, hipStream_t stream,
                         const GpuTensor& grad_output_hidden_states,
                         BertLayerCache& cache,
                         GpuTensor& grad_input_hidden_states) {
    if (!grad_output_hidden_states.is_allocated() || !cache.layer_input_ptr || !cache.ffn_input_after_attention.is_allocated()) {
        throw std::runtime_error("Required tensors/cache not ready for BertLayer::backward.");
    }
    if(!grad_input_hidden_states.is_allocated() || grad_input_hidden_states.dims_ != cache.layer_input_ptr->dims_){
        grad_input_hidden_states.allocate(cache.layer_input_ptr->dims());
        grad_input_hidden_states.zero_out(stream);
    }

    // --- 역전파: FFN -> 어텐션 블록 순서 ---
    
    // 1. FFN의 최종 LayerNorm 역전파
    GpuTensor grad_ffn_sum_input;
    grad_ffn_sum_input.allocate(grad_output_hidden_states.dims_);
    ffn_output_layernorm_.backward(stream, grad_output_hidden_states, cache.ffn_output_layernorm_cache, grad_ffn_sum_input);

    // grad_ffn_sum_input은 FFN 잔차 연결의 출력에 대한 그래디언트임.
    // 이 그래디언트는 잔차 연결의 두 입력(FFN 출력, 어텐션 출력)으로 나뉘어 전달됨.
    GpuTensor grad_ffn_dropout_output;
    grad_ffn_dropout_output.copy_from_gpu(grad_ffn_sum_input, stream);
    
    GpuTensor grad_attention_block_output;
    grad_attention_block_output.allocate(cache.ffn_input_after_attention.dims_);
    grad_attention_block_output.copy_from_gpu(grad_ffn_sum_input, stream); // 잔차 경로의 그래디언트

    // 2. FFN의 Dropout -> Output Dense -> Intermediate Dense 역전파
    GpuTensor grad_ffn_dense_output_result, grad_intermediate_output, grad_ffn_input;
    grad_ffn_dense_output_result.allocate(grad_ffn_dropout_output.dims_);
    grad_intermediate_output.allocate(cache.ffn_intermediate_dense_cache.input->dims_); // This should be intermediate_output size
    grad_ffn_input.allocate(cache.ffn_input_after_attention.dims_);

    ffn_output_dropout_.backward(stream, grad_ffn_dense_output_result, grad_ffn_dropout_output, cache.ffn_output_dropout_cache);
    ffn_output_dense_.backward(blas_handle, stream, grad_ffn_dense_output_result, cache.ffn_output_dense_cache, grad_intermediate_output);
    ffn_intermediate_dense_.backward(blas_handle, stream, grad_intermediate_output, cache.ffn_intermediate_dense_cache, grad_ffn_input);

    // 3. 어텐션 블록 출력에 대한 그래디언트 합산
    // FFN의 메인 경로를 통과한 그래디언트(grad_ffn_input)와 잔차 경로를 통과한 그래디언트(grad_attention_block_output)를 합산
    launch_accumulate_kernel(stream, (float*)grad_attention_block_output.d_ptr(), (const float*)grad_ffn_input.d_ptr(), grad_attention_block_output.num_elements_);

    // 4. 어텐션 블록 역전파
    // 합산된 그래디언트를 어텐션 블록의 backward 함수에 전달.
    // 이 함수 내부에서 입력 그래디언트(grad_input_hidden_states)가 계산되고 누적됨.
    attention_.backward(blas_handle, stream, grad_attention_block_output, cache.attention_cache, grad_input_hidden_states);
}

// ============================================================================
// BertEncoder 구현부
// ============================================================================

BertEncoder::BertEncoder(const BertConfig& config, const std::string& name_prefix) {
    layers_.reserve(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; ++i) {
        layers_.emplace_back(std::make_unique<BertLayer>(config, name_prefix + ".layer." + std::to_string(i)));
    }
}

std::vector<Parameter*> BertEncoder::get_parameters() {
    std::vector<Parameter*> params;
    for (const auto& layer : layers_) {
        auto layer_params = layer->get_parameters();
        params.insert(params.end(), layer_params.begin(), layer_params.end());
    }
    return params;
}

void BertEncoder::forward(rocblas_handle blas_handle, hipStream_t stream,
                          const GpuTensor& initial_hidden_states,
                          const GpuTensor& attention_mask,
                          GpuTensor& final_hidden_states,
                          BertEncoderCache& cache,
                          bool is_training) {
    if (!initial_hidden_states.is_allocated()) {
        throw std::runtime_error("Initial hidden_states not allocated for BertEncoder forward.");
    }
    if (layers_.empty()) {
        final_hidden_states.copy_from_gpu(initial_hidden_states, stream);
        return;
    }
    if (cache.layer_caches.size() != layers_.size()) {
        throw std::runtime_error("Encoder cache size does not match number of layers.");
    }
    if(!final_hidden_states.is_allocated() || final_hidden_states.dims_ != initial_hidden_states.dims_){
        final_hidden_states.allocate(initial_hidden_states.dims_);
    }

    GpuTensor current_hidden_states, next_hidden_states;
    current_hidden_states.copy_from_gpu(initial_hidden_states, stream);
    next_hidden_states.allocate(initial_hidden_states.dims_);

    for (size_t i = 0; i < layers_.size(); ++i) {
        layers_[i]->forward(blas_handle, stream, current_hidden_states, attention_mask, next_hidden_states, cache.layer_caches[i], is_training);
        current_hidden_states.copy_from_gpu(next_hidden_states, stream);
    }

    final_hidden_states.copy_from_gpu(current_hidden_states, stream);
}

void BertEncoder::backward(rocblas_handle blas_handle, hipStream_t stream,
                           const GpuTensor& grad_final_hidden_states,
                           BertEncoderCache& cache,
                           GpuTensor& grad_initial_hidden_states) {
    if (layers_.empty()) {
        grad_initial_hidden_states.copy_from_gpu(grad_final_hidden_states, stream);
        return;
    }
    if (!grad_final_hidden_states.is_allocated()) {
        throw std::runtime_error("grad_final_hidden_states not allocated for BertEncoder backward.");
    }
    if(!grad_initial_hidden_states.is_allocated() || grad_initial_hidden_states.dims_ != grad_final_hidden_states.dims_){
        grad_initial_hidden_states.allocate(grad_final_hidden_states.dims_);
    }

    GpuTensor current_grad_hidden_states, prev_layer_grad_hidden_states;
    current_grad_hidden_states.copy_from_gpu(grad_final_hidden_states, stream);
    prev_layer_grad_hidden_states.allocate(grad_final_hidden_states.dims_);

    for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
        layers_[i]->backward(blas_handle, stream, current_grad_hidden_states, cache.layer_caches[i], prev_layer_grad_hidden_states);
        current_grad_hidden_states.copy_from_gpu(prev_layer_grad_hidden_states, stream);
    }

    grad_initial_hidden_states.copy_from_gpu(current_grad_hidden_states, stream);
}

// ============================================================================
// BertModel 구현부
// ============================================================================

BertModel::BertModel(const BertConfig& config, const std::string& name_prefix)
    : config_(config),
      embeddings_(config, name_prefix + ".embeddings"),
      encoder_(config, name_prefix + ".encoder")
      {}

std::vector<Parameter*> BertModel::get_parameters() {
    auto embedding_params = embeddings_.get_parameters();
    auto encoder_params = encoder_.get_parameters();
    std::vector<Parameter*> params;
    params.insert(params.end(), embedding_params.begin(), embedding_params.end());
    params.insert(params.end(), encoder_params.begin(), encoder_params.end());
    return params;
}

Parameter* BertModel::get_word_embedding_params() {
    return embeddings_.get_word_embedding_params();
}

void BertModel::forward(rocblas_handle blas_handle, hipStream_t stream,
                        const GpuTensor& input_ids,
                        const GpuTensor& attention_mask,
                        const GpuTensor& token_type_ids,
                        GpuTensor& sequence_output,
                        BertModelCache& cache,
                        bool is_training) {
    // 1. 임베딩 계층
    GpuTensor embedding_output;
    embedding_output.allocate({input_ids.dim_size(0), input_ids.dim_size(1), config_.hidden_size});
    embeddings_.forward(stream, input_ids, token_type_ids, embedding_output, cache.embeddings_cache, is_training);

    // 2. 인코더 계층
    encoder_.forward(blas_handle, stream, embedding_output, attention_mask, sequence_output, cache.encoder_cache, is_training);
}

void BertModel::backward(rocblas_handle blas_handle, hipStream_t stream,
                         const GpuTensor& grad_sequence_output,
                         BertModelCache& cache) {
    GpuTensor grad_embedding_output;
    grad_embedding_output.allocate(grad_sequence_output.dims());

    // 인코더 역전파
    encoder_.backward(blas_handle, stream, grad_sequence_output, cache.encoder_cache, grad_embedding_output);

    // 임베딩 역전파
    embeddings_.backward(stream, grad_embedding_output, cache.embeddings_cache);
}
