#include "attention_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp"
#include "bert_components_hip.hpp" // For BertConfig definition
#include <cmath>                   // For sqrtf
#include <stdexcept>                 // For std::runtime_error

// ============================================================================
// BertSelfAttention 구현부
// ============================================================================

/**
 * @brief BertSelfAttention 생성자.
 * @param config BERT 모델의 설정을 담은 BertConfig 객체.
 * @param name_prefix 파라미터(가중치, 편향)의 이름을 생성하기 위한 접두사.
 */
BertSelfAttention::BertSelfAttention(const BertConfig& config, const std::string& name_prefix)
    : query_(config.hidden_size, config.hidden_size, config, name_prefix + ".query", true),
      key_(config.hidden_size, config.hidden_size, config, name_prefix + ".key", true),
      value_(config.hidden_size, config.hidden_size, config, name_prefix + ".value", true),
      dropout_(config.attention_probs_dropout_prob), // 어텐션 확률에 대한 드롭아웃
      num_attention_heads_(config.num_attention_heads),
      attention_head_size_(config.hidden_size / config.num_attention_heads)
{
    // hidden_size는 헤드 수로 나누어 떨어져야 합니다.
    if (config.hidden_size % config.num_attention_heads != 0) {
        throw std::runtime_error("Hidden size is not divisible by the number of attention heads");
    }
}

/**
 * @brief 이 모듈이 소유한 모든 학습 가능한 파라미터(Q, K, V 가중치 및 편향)의 포인터를 반환합니다.
 */
std::vector<Parameter*> BertSelfAttention::get_parameters() {
    std::vector<Parameter*> params;
    auto q_params = query_.get_parameters();
    auto k_params = key_.get_parameters();
    auto v_params = value_.get_parameters();
    params.insert(params.end(), q_params.begin(), q_params.end());
    params.insert(params.end(), k_params.begin(), k_params.end());
    params.insert(params.end(), v_params.begin(), v_params.end());
    return params;
}

/**
 * @brief Self-Attention의 순전파를 수행합니다.
 * @details 입력 hidden_states를 Q, K, V로 각각 투영하고, multi-head로 reshape한 뒤, 어텐션 스코어를 계산하여 최종 context 벡터를 출력합니다.
 */
void BertSelfAttention::forward(rocblas_handle blas_handle, hipStream_t stream,
                               const GpuTensor& hidden_states,
                               const GpuTensor& attention_mask,
                               SelfAttentionCache& cache,
                               bool is_training) {
    if (!hidden_states.is_allocated()) {
        throw std::runtime_error("Input hidden_states not allocated for BertSelfAttention forward.");
    }

    const int batch_size = hidden_states.dim_size(0);
    const int seq_len = hidden_states.dim_size(1);
    const int hidden_size_config = num_attention_heads_ * attention_head_size_;

    // --- 1. Q, K, V 선형 투영 ---
    cache.q_proj.allocate({batch_size, seq_len, hidden_size_config});
    cache.k_proj.allocate({batch_size, seq_len, hidden_size_config});
    cache.v_proj.allocate({batch_size, seq_len, hidden_size_config});

    query_.forward(blas_handle, stream, hidden_states, cache.q_proj, cache.q_dense_cache);
    key_.forward(blas_handle, stream, hidden_states, cache.k_proj, cache.k_dense_cache);
    value_.forward(blas_handle, stream, hidden_states, cache.v_proj, cache.v_dense_cache);

    // --- 2. Multi-head attention을 위해 텐서 Reshape: (B, S, H) -> (B, N, S, A) ---
    cache.q_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    cache.k_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    cache.v_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});

    launch_transpose_for_scores_kernel(stream, (float*)cache.q_reshaped.d_ptr_, (const float*)cache.q_proj.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_for_scores_kernel(stream, (float*)cache.k_reshaped.d_ptr_, (const float*)cache.k_proj.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_for_scores_kernel(stream, (float*)cache.v_reshaped.d_ptr_, (const float*)cache.v_proj.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);

    // --- 3. Attention Score 계산 (Q * K^T) ---
    cache.attention_scores.allocate({batch_size, num_attention_heads_, seq_len, seq_len});
    const float alpha_gemm = 1.0f, beta_gemm = 0.0f;
    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));

    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_none, rocblas_operation_transpose,
        seq_len, seq_len, attention_head_size_,
        &alpha_gemm,
        (const float*)cache.q_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_,
        (const float*)cache.k_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_,
        &beta_gemm,
        (float*)cache.attention_scores.d_ptr(), seq_len, (long long)seq_len * seq_len,
        batch_size * num_attention_heads_));

    // --- 4. Score 스케일링 및 마스킹 ---
    const float scale_factor = 1.0f / sqrtf(static_cast<float>(attention_head_size_));
    if (attention_mask.is_allocated()) {
        launch_scale_and_mask_kernel(stream, (float*)cache.attention_scores.d_ptr_, (const float*)attention_mask.d_ptr_, batch_size, num_attention_heads_, seq_len, scale_factor);
    } else {
        launch_scale_kernel(stream, (float*)cache.attention_scores.d_ptr_, scale_factor, cache.attention_scores.num_elements_);
    }

    // --- 5. Softmax 적용 ---
    cache.attention_probs.allocate({batch_size, num_attention_heads_, seq_len, seq_len});
    const int M_softmax_fwd = batch_size * num_attention_heads_ * seq_len;
    const int N_softmax_fwd = seq_len;
    launch_softmax_kernel(stream, (float*)cache.attention_probs.d_ptr_, (const float*)cache.attention_scores.d_ptr_, M_softmax_fwd, N_softmax_fwd);

    // --- 6. Dropout 적용 ---
    dropout_.forward(stream, cache.attention_probs, cache.attention_probs_dropout_cache, is_training);

    // --- 7. Context Layer 계산 (Attention Probs * V) ---
    cache.context_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle,
        rocblas_operation_none, rocblas_operation_none,
        seq_len, attention_head_size_, seq_len,
        &alpha_gemm,
        (const float*)cache.attention_probs.d_ptr(), seq_len, (long long)seq_len * seq_len,
        (const float*)cache.v_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_,
        &beta_gemm,
        (float*)cache.context_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_,
        batch_size * num_attention_heads_));

    // --- 8. 최종 Context Layer Reshape: (B, N, S, A) -> (B, S, H) ---
    cache.context_layer.allocate({batch_size, seq_len, hidden_size_config});
    launch_transpose_back_kernel(stream, (float*)cache.context_layer.d_ptr_, (const float*)cache.context_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    
    // 역전파를 위해 입력 포인터 저장
    cache.input_hidden_states = &hidden_states;
}

/**
 * @brief Self-Attention의 역전파를 수행합니다.
 */
void BertSelfAttention::backward(rocblas_handle blas_handle, hipStream_t stream,
                                const GpuTensor& grad_context_layer_output,
                                SelfAttentionCache& cache,
                                GpuTensor& grad_input_hidden_states) {
    if (!grad_context_layer_output.is_allocated() || !cache.input_hidden_states) {
        throw std::runtime_error("Required tensors not allocated for BertSelfAttention backward.");
    }

    const int batch_size = cache.input_hidden_states->dim_size(0);
    const int seq_len = cache.input_hidden_states->dim_size(1);
    const int hidden_size_config = num_attention_heads_ * attention_head_size_;

    if (!grad_input_hidden_states.is_allocated() || grad_input_hidden_states.dims() != cache.input_hidden_states->dims()) {
        grad_input_hidden_states.allocate(cache.input_hidden_states->dims());
        grad_input_hidden_states.zero_out(stream);
    }

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;

    // --- 1. grad_context_layer_output Reshape: (B, S, H) -> (B, N, S, A) ---
    GpuTensor grad_context_reshaped;
    grad_context_reshaped.allocate({batch_size, num_attention_heads_, seq_len, attention_head_size_});
    launch_transpose_for_scores_kernel(stream, (float*)grad_context_reshaped.d_ptr_, (const float*)grad_context_layer_output.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);

    // --- 2. Context Layer (Probs * V) 역전파 ---
    GpuTensor grad_attention_probs, grad_v_reshaped;
    grad_attention_probs.allocate(cache.attention_probs.dims());
    grad_v_reshaped.allocate(cache.v_reshaped.dims());

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_none, rocblas_operation_transpose, seq_len, seq_len, attention_head_size_, &alpha, (const float*)grad_context_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, (const float*)cache.v_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, &beta_zero, (float*)grad_attention_probs.d_ptr(), seq_len, (long long)seq_len * seq_len, batch_size * num_attention_heads_));
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_transpose, rocblas_operation_none, seq_len, attention_head_size_, seq_len, &alpha, (const float*)cache.attention_probs.d_ptr(), seq_len, (long long)seq_len * seq_len, (const float*)grad_context_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, &beta_zero, (float*)grad_v_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, batch_size * num_attention_heads_));

    // --- 3. Dropout 및 Softmax 역전파 ---
    dropout_.backward(stream, grad_attention_probs, grad_attention_probs, cache.attention_probs_dropout_cache);
    GpuTensor grad_scores;
    grad_scores.allocate(cache.attention_scores.dims());
    launch_softmax_backward_kernel(stream, (float*)grad_scores.d_ptr_, (const float*)grad_attention_probs.d_ptr_, (const float*)cache.attention_probs.d_ptr_, batch_size * num_attention_heads_ * seq_len, seq_len);

    // --- 4. 스케일링 역전파 ---
    const float scale_factor = 1.0f / sqrtf(static_cast<float>(attention_head_size_));
    launch_scale_kernel(stream, (float*)grad_scores.d_ptr_, scale_factor, grad_scores.num_elements_);
    
    // --- 5. Attention Score (Q * K^T) 역전파 ---
    GpuTensor grad_q_reshaped, grad_k_reshaped;
    grad_q_reshaped.allocate(cache.q_reshaped.dims());
    grad_k_reshaped.allocate(cache.k_reshaped.dims());
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_none, rocblas_operation_none, seq_len, attention_head_size_, seq_len, &alpha, (const float*)grad_scores.d_ptr(), seq_len, (long long)seq_len * seq_len, (const float*)cache.k_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, &beta_zero, (float*)grad_q_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, batch_size * num_attention_heads_));
    ROCBLAS_CHECK(rocblas_sgemm_strided_batched(blas_handle, rocblas_operation_transpose, rocblas_operation_none, seq_len, attention_head_size_, seq_len, &alpha, (const float*)grad_scores.d_ptr(), seq_len, (long long)seq_len * seq_len, (const float*)cache.q_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, &beta_zero, (float*)grad_k_reshaped.d_ptr(), attention_head_size_, (long long)seq_len * attention_head_size_, batch_size * num_attention_heads_));

    // --- 6. Reshape 역전파 및 Q, K, V의 선형 투영 역전파 ---
    GpuTensor grad_q_proj, grad_k_proj, grad_v_proj;
    grad_q_proj.allocate({batch_size, seq_len, hidden_size_config});
    grad_k_proj.allocate({batch_size, seq_len, hidden_size_config});
    grad_v_proj.allocate({batch_size, seq_len, hidden_size_config});
    launch_transpose_back_kernel(stream, (float*)grad_q_proj.d_ptr_, (const float*)grad_q_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_back_kernel(stream, (float*)grad_k_proj.d_ptr_, (const float*)grad_k_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    launch_transpose_back_kernel(stream, (float*)grad_v_proj.d_ptr_, (const float*)grad_v_reshaped.d_ptr_, batch_size, seq_len, num_attention_heads_, attention_head_size_);
    
    value_.backward(blas_handle, stream, grad_v_proj, cache.v_dense_cache, grad_input_hidden_states);
    key_.backward(blas_handle, stream, grad_k_proj, cache.k_dense_cache, grad_input_hidden_states);
    query_.backward(blas_handle, stream, grad_q_proj, cache.q_dense_cache, grad_input_hidden_states);
}

// ============================================================================
// BertAttention 구현부 (Self-Attention + Output Layer)
// ============================================================================

BertAttention::BertAttention(const BertConfig& config, const std::string& name_prefix)
    : self_attention_(config, name_prefix + ".attention.self"),
      output_dense_(config.hidden_size, config.hidden_size, config, name_prefix + ".attention.output.dense"),
      output_dropout_(config.hidden_dropout_prob),
      output_layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".attention.output.LayerNorm")
      {}

/**
 * @brief 이 모듈 및 하위 모듈의 모든 학습 파라미터를 반환합니다.
 */
std::vector<Parameter*> BertAttention::get_parameters() {
    auto self_params = self_attention_.get_parameters();
    auto dense_params = output_dense_.get_parameters();
    auto ln_params = output_layernorm_.get_parameters();
    self_params.insert(self_params.end(), dense_params.begin(), dense_params.end());
    self_params.insert(self_params.end(), ln_params.begin(), ln_params.end());
    return self_params;
}

/**
 * @brief BertAttention 레이어 전체(Self-Attention + Output)의 순전파를 수행합니다.
 */
void BertAttention::forward(rocblas_handle blas_handle, hipStream_t stream,
                           const GpuTensor& input_tensor,
                           const GpuTensor& attention_mask,
                           BertAttentionCache& cache,
                           bool is_training) {
    if (!input_tensor.is_allocated()) {
        throw std::runtime_error("Input tensor not allocated for BertAttention forward.");
    }
    cache.attention_input = &input_tensor;

    // --- 1. Self-Attention 수행 ---
    self_attention_.forward(blas_handle, stream, input_tensor, attention_mask, cache.self_attention_cache, is_training);

    // --- 2. Self-Attention 출력에 대한 후처리 (Dense + Dropout) ---
    GpuTensor dense_output;
    dense_output.allocate(cache.self_attention_cache.context_layer.dims());
    output_dense_.forward(blas_handle, stream, cache.self_attention_cache.context_layer, dense_output, cache.output_dense_cache);
    output_dropout_.forward(stream, dense_output, cache.output_dropout_cache, is_training);

    // --- 3. 잔차 연결 (Residual Connection) ---
    GpuTensor attention_residual_sum_output;
    attention_residual_sum_output.allocate(dense_output.dims());
    launch_elementwise_add_kernel(stream, (float*)attention_residual_sum_output.d_ptr(), (const float*)dense_output.d_ptr(), (const float*)input_tensor.d_ptr(), attention_residual_sum_output.num_elements_);

    // --- 4. 최종 Layer Normalization ---
    // 최종 출력은 LayerNorm(잔차 연결 결과)입니다.
    // BertLayer에서 사용하기 위해 출력을 cache.self_attention_cache.context_layer에 덮어씁니다. (이는 필드 재사용)
    output_layernorm_.forward(stream, attention_residual_sum_output, cache.self_attention_cache.context_layer, cache.output_layernorm_cache);
}

/**
 * @brief BertAttention 레이어 전체의 역전파를 수행합니다.
 */
void BertAttention::backward(rocblas_handle blas_handle, hipStream_t stream,
                            const GpuTensor& grad_final_output,
                            BertAttentionCache& cache,
                            GpuTensor& grad_input_tensor) {
    if (!grad_final_output.is_allocated() || !cache.attention_input) {
        throw std::runtime_error("Required tensors not allocated for BertAttention backward.");
    }
    if(!grad_input_tensor.is_allocated() || grad_input_tensor.dims() != cache.attention_input->dims()){
        grad_input_tensor.allocate(cache.attention_input->dims());
        grad_input_tensor.zero_out(stream);
    }

    // --- 1. LayerNorm 역전파 ---
    GpuTensor grad_after_layernorm;
    grad_after_layernorm.allocate(grad_final_output.dims());
    output_layernorm_.backward(stream, grad_final_output, cache.output_layernorm_cache, grad_after_layernorm);

    // --- 2. Dropout 역전파 ---
    GpuTensor grad_after_dropout;
    grad_after_dropout.allocate(grad_after_layernorm.dims());
    output_dropout_.backward(stream, grad_after_dropout, grad_after_layernorm, cache.output_dropout_cache);

    // --- 3. Dense (출력) 역전파 ---
    GpuTensor grad_after_dense;
    grad_after_dense.allocate(grad_after_dropout.dims());
    output_dense_.backward(blas_handle, stream, grad_after_dropout, cache.output_dense_cache, grad_after_dense);

    // --- 4. SelfAttention 역전파 ---
    // grad_after_dense는 self_attention 모듈의 출력에 대한 그래디언트입니다.
    // self_attention_.backward는 계산된 그래디언트를 grad_input_tensor에 **누적**합니다.
    self_attention_.backward(blas_handle, stream, grad_after_dense, cache.self_attention_cache, grad_input_tensor);

    // --- 5. 잔차 연결(Residual Connection) 역전파 [핵심] ---
    // LayerNorm 역전파의 결과(grad_after_layernorm)는 합산 연산의 출력에 대한 그래디언트이므로,
    // 합산의 두 입력(attention output, original input) 모두에 전달되어야 합니다.
    // original input 경로의 그래디언트를 최종 입력 그래디언트에 더해줍니다.
    launch_accumulate_kernel(stream,
                             (float*)grad_input_tensor.d_ptr(),
                             (const float*)grad_after_layernorm.d_ptr(),
                             grad_input_tensor.num_elements_);
}
