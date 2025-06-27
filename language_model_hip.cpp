#include "language_model_hip.hpp"
#include "common_hip.hpp"
#include "hip_kernels.hpp" // For loss kernel, reduction kernels
#include "bert_components_hip.hpp" // For BertConfig, BertModel parameter access
#include <stdexcept>
#include <vector>
#include <string>

// ============================================================================
// BertLMPredictionHead Implementation
// ============================================================================
BertLMPredictionHead::BertLMPredictionHead(const BertConfig& config, Parameter& shared_word_embeddings_parameter, const std::string& name_prefix)
    : config_(config),
      transform_dense_(config.hidden_size, config.hidden_size, config, name_prefix + ".transform.dense"), // BertConfig passed here due to original DenseLayer constructor
      // transform_act_fn_(), // GELU is fused into DenseLayer as per nn_layers_hip.cpp
      transform_layernorm_(config.hidden_size, config.layer_norm_eps, name_prefix + ".transform.LayerNorm"),
      decoder_bias_({config.vocab_size}, name_prefix + ".bias"), // Parameter constructor
      shared_word_embeddings_(shared_word_embeddings_parameter) {
        // Ensure shared_word_embeddings's weight tensor is (vocab_size, hidden_size)
        if (shared_word_embeddings_.weights.dims_.size() != 2 ||
            shared_word_embeddings_.weights.dim_size(0) != config.vocab_size ||
            shared_word_embeddings_.weights.dim_size(1) != config.hidden_size) {
            throw std::runtime_error("Shared word embedding dimensions mismatch in BertLMPredictionHead.");
        }
      }

std::vector<Parameter*> BertLMPredictionHead::get_parameters() {
    auto dense_params = transform_dense_.get_parameters();
    auto ln_params = transform_layernorm_.get_parameters();
    // Note: shared_word_embeddings_ is NOT returned here as its ownership is with BertEmbeddings.
    // Only decoder_bias_ is unique to this head, plus transform parameters.
    std::vector<Parameter*> params;
    params.insert(params.end(), dense_params.begin(), dense_params.end());
    params.insert(params.end(), ln_params.begin(), ln_params.end());
    params.push_back(&decoder_bias_);
    return params;
}

void BertLMPredictionHead::forward(rocblas_handle blas_handle, hipStream_t stream,
                                 const GpuTensor& hidden_states, // (B, S, H)
                                 GpuTensor& logits,             // Output: (B, S, V)
                                 BertLMPredictionHeadCache& cache) {
    if (!hidden_states.is_allocated()) {
        throw std::runtime_error("Input hidden_states not allocated for BertLMPredictionHead forward.");
    }
    int batch_size = hidden_states.dim_size(0);
    int seq_len = hidden_states.dim_size(1);
    // int hidden_size = hidden_states.dim_size(2); // Should match config_.hidden_size

    std::vector<int> logits_dims = {batch_size, seq_len, config_.vocab_size};
    if (!logits.is_allocated() || logits.dims_ != logits_dims) {
        logits.allocate(logits_dims);
    }

    cache.hidden_states_input = &hidden_states;

    // 1. Transform part: Dense -> (GELU is fused in DenseLayer) -> LayerNorm
    // Output of DenseLayer (which includes GELU): (B, S, H)
    cache.transform_dense_output.allocate(hidden_states.dims_);
    transform_dense_.forward(blas_handle, stream, hidden_states, cache.transform_dense_output, cache.transform_dense_cache);

    // LayerNorm on the output of dense+GELU
    // Output of LayerNorm: (B, S, H) - this is `transformed_states` in user's original code
    cache.transform_layernorm_output.allocate(hidden_states.dims_);
    transform_layernorm_.forward(stream, cache.transform_dense_output, cache.transform_layernorm_output, cache.transform_layernorm_cache);

    // 2. Decoder part: MatMul with shared word_embeddings + add bias
    // logits = transform_layernorm_output * word_embeddings.weights^T + decoder_bias_
    // transform_layernorm_output: (B*S, H)
    // word_embeddings.weights: (V, H) -> transpose to (H, V)
    // logits: (B*S, V)

    const float alpha = 1.0f, beta = 0.0f;
    int M = batch_size * seq_len;
    int K = config_.hidden_size;
    int N = config_.vocab_size;

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    // C_MN = A_MK * B_KN^T where B is stored as (N,K)
    // A is transform_layernorm_output (M, K)
    // B is shared_word_embeddings_.weights (N=Vocab, K=Hidden)
    // C is logits (M, N=Vocab)
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,       // opA
                                rocblas_operation_transpose,  // opB (weights are V,H, use as H,V)
                                M, N, K,
                                &alpha,
                                (const float*)cache.transform_layernorm_output.d_ptr_, K, // A, lda=K
                                (const float*)shared_word_embeddings_.weights.d_ptr_, K,   // B (stored N,K), ldb=K
                                &beta,
                                (float*)logits.d_ptr(), N));                             // C, ldc=N

    // Add decoder bias. Bias is (V). Needs to be broadcasted/added to each row of (B*S, V)
    // launch_add_bias_kernel(stream, output, input, bias, M_rows, N_cols)
    // Here output=logits, input=logits (in-place), bias=decoder_bias_.weights
    // M_rows = M (batch_size * seq_len), N_cols = N (vocab_size)
    launch_add_bias_kernel(stream,
                           (float*)logits.d_ptr_,
                           (const float*)logits.d_ptr_, // In-place addition
                           (const float*)decoder_bias_.weights.d_ptr_, // Bias tensor
                           M, N);
    // cache.logits_output_ptr = &logits; // Store pointer if needed, but logits is output param
}

void BertLMPredictionHead::backward(rocblas_handle blas_handle, hipStream_t stream,
                                  const GpuTensor& grad_logits, // (B, S, V)
                                  BertLMPredictionHeadCache& cache,
                                  GpuTensor& grad_hidden_states) { // Output: (B, S, H)
    if (!grad_logits.is_allocated() || !cache.hidden_states_input || !cache.hidden_states_input->is_allocated() ||
        !cache.transform_layernorm_output.is_allocated() ) {
        throw std::runtime_error("Required tensors/cache not allocated for BertLMPredictionHead backward.");
    }
    if(!grad_hidden_states.is_allocated() || grad_hidden_states.dims_ != cache.hidden_states_input->dims_){
        grad_hidden_states.allocate(cache.hidden_states_input->dims_);
        grad_hidden_states.zero_out(stream); // Ensure it's zeroed for accumulation by DenseLayer backward
    }

    // Ensure gradient tensors for parameters are allocated
    transform_dense_.params.allocate_gradients();
    transform_layernorm_.params.allocate_gradients();
    decoder_bias_.allocate_gradients(); // For decoder_bias_.grad_weights
    // shared_word_embeddings_.grad_weights is allocated by BertEmbeddings module.

    int M = grad_logits.dim_size(0) * grad_logits.dim_size(1); // batch_size * seq_len
    int N_vocab = grad_logits.dim_size(2); // vocab_size
    int K_hidden = cache.hidden_states_input->dim_size(2); // hidden_size

    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f; // For accumulation

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));

    // 1. Backward for decoder bias: sum grad_logits along batch_seq dimension
    // grad_logits is (M, N_vocab). Sum over M.
    // launch_reduce_sum_axis0_add_kernel(stream, out_grad (N_vocab), in_grad (M, N_vocab), M_reduce, N_keep)
    launch_reduce_sum_axis0_add_kernel(stream,
                                   (float*)decoder_bias_.grad_weights.d_ptr(), // out_grad_bias (V)
                                   (const float*)grad_logits.d_ptr(),        // in_grad_logits (B*S, V)
                                   M, N_vocab);

    // 2. Backward for shared_word_embeddings_.weights (dL/dW_embed)
    // dL/dW_embed = grad_logits^T * transform_layernorm_output
    // grad_logits (M, N_v) -> use as (N_v, M)
    // transform_layernorm_output (M, K_h)
    // dL/dW_embed (N_v, K_h) - This matches dimensions of shared_word_embeddings_.weights
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_transpose, // opA (grad_logits)
                                rocblas_operation_none,    // opB (transform_layernorm_output)
                                N_vocab, K_hidden, M,      // m, n, k
                                &alpha,
                                (const float*)grad_logits.d_ptr(), N_vocab, // A (M,N_v) lda=N_v
                                (const float*)cache.transform_layernorm_output.d_ptr(), K_hidden, // B (M,K_h) ldb=K_h
                                &beta_one, // Accumulate to existing gradients in shared_word_embeddings_
                                (float*)shared_word_embeddings_.grad_weights.d_ptr(), K_hidden)); // C (N_v,K_h) ldc=K_h

    // 3. Backward for transform_layernorm_output (dL/dX_transformed)
    // dL/dX_transformed = grad_logits * shared_word_embeddings_.weights (no transpose on weights)
    // grad_logits (M, N_v)
    // shared_word_embeddings_.weights (N_v, K_h)
    // dL/dX_transformed (M, K_h)
    GpuTensor grad_transform_layernorm_output;
    grad_transform_layernorm_output.allocate(cache.transform_layernorm_output.dims_);
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,    // opA (grad_logits)
                                rocblas_operation_none,    // opB (shared_word_embeddings_.weights)
                                M, K_hidden, N_vocab,      // m, n, k
                                &alpha,
                                (const float*)grad_logits.d_ptr(), N_vocab, // A (M,N_v) lda=N_v
                                (const float*)shared_word_embeddings_.weights.d_ptr(), K_hidden, // B (N_v,K_h) ldb=K_h
                                &beta_zero,
                                (float*)grad_transform_layernorm_output.d_ptr(), K_hidden)); // C (M,K_h) ldc=K_h

    // 4. Backward for transform LayerNorm
    GpuTensor grad_transform_dense_output; // Grad w.r.t. output of transform_dense (input to LayerNorm)
    grad_transform_dense_output.allocate(cache.transform_dense_output.dims_);
    transform_layernorm_.backward(stream, grad_transform_layernorm_output, cache.transform_layernorm_cache, grad_transform_dense_output);

    // 5. Backward for transform Dense (and its fused GELU)
    // grad_hidden_states is the final output gradient w.r.t. input of this head.
    // DenseLayer::backward needs to handle GELU derivative internally. (Currently a GAP in nn_layers_hip.cpp)
    transform_dense_.backward(blas_handle, stream, grad_transform_dense_output, cache.transform_dense_cache, grad_hidden_states);
}


// ============================================================================
// CANBertForMaskedLM Implementation
// ============================================================================
CANBertForMaskedLM::CANBertForMaskedLM(const BertConfig& cfg) : config_(cfg) { // Make a copy of config or ensure lifetime
    bert_model_ = std::make_unique<BertModel>(config_, "bert"); // Pass config by const ref
    // Pass the actual Parameter object for word embeddings from bert_model_ to lm_head_
    lm_head_ = std::make_unique<BertLMPredictionHead>(config_, *bert_model_->get_word_embedding_params(), "cls.predictions");

    ROCBLAS_CHECK(rocblas_create_handle(&blas_handle_));
    HIP_CHECK(hipStreamCreate(&stream_)); // Using default stream 0 can also be an option if no specific stream is needed.

    // Optimizer setup
    auto model_params = get_parameters(); // Collect all parameters
    for(auto* p : model_params) {
        if (p) p->allocate_gradients(); // Ensure gradients are allocated for all params
    }
    // Default AdamW params from original user code. Configurable if needed.
    optimizer_ = std::make_unique<AdamWOptimizer>(model_params, 1e-4f, 0.9f, 0.999f, 1e-6f, 0.01f);
}

CANBertForMaskedLM::~CANBertForMaskedLM() {
    if (blas_handle_) rocblas_destroy_handle(blas_handle_);
    if (stream_) hipStreamDestroy(stream_);
}

void CANBertForMaskedLM::initialize_parameters(float mean, float stddev) {
    for (auto* p : get_parameters()) {
        if (p) p->initialize_random(mean, stddev);
    }
}

std::vector<Parameter*> CANBertForMaskedLM::get_parameters() {
    auto bert_params = bert_model_->get_parameters();
    auto head_params = lm_head_->get_parameters();
    bert_params.insert(bert_params.end(), head_params.begin(), head_params.end());
    return bert_params;
}

void CANBertForMaskedLM::train_step(const GpuTensor& input_ids,
                                  const GpuTensor& attention_mask,
                                  const GpuTensor& token_type_ids,
                                  const GpuTensor& labels, // (B, S)
                                  GpuTensor& loss_output) { // Scalar output
    if (!input_ids.is_allocated() || !attention_mask.is_allocated() || !labels.is_allocated()) {
        throw std::runtime_error("Input tensors (input_ids, attention_mask, labels) must be allocated for train_step.");
    }
    // token_type_ids is optional, BertEmbeddings handles unallocated case by creating a dummy.

    optimizer_->zero_grad(); // Zero out all parameter gradients

    // --- Forward Pass ---
    BertModelCache model_cache(config_.num_hidden_layers); // num_hidden_layers from config
    BertLMPredictionHeadCache head_cache;

    GpuTensor sequence_output; // (B, S, H) - Output from BertModel
    // sequence_output needs to be allocated based on input_ids dims and config.hidden_size
    sequence_output.allocate({input_ids.dim_size(0), input_ids.dim_size(1), config_.hidden_size});

    bert_model_->forward(blas_handle_, stream_, input_ids, attention_mask, token_type_ids,
                         sequence_output, model_cache, true /*is_training*/);

    GpuTensor logits; // (B, S, V) - Output from LM Head
    logits.allocate({input_ids.dim_size(0), input_ids.dim_size(1), config_.vocab_size});
    lm_head_->forward(blas_handle_, stream_, sequence_output, logits, head_cache);

    // --- Loss Calculation & Initial Gradient for Backward Pass ---
    GpuTensor grad_logits; // (B, S, V) - Gradient of loss w.r.t. logits
    grad_logits.allocate(logits.dims_);

    if (!loss_output.is_allocated() || loss_output.num_elements_ != 1) {
        loss_output.allocate({1}); // Scalar loss
    }
    loss_output.zero_out(stream_); // Initialize loss to zero for accumulation

    // This kernel calculates dL/dLogits and sum(-log_probs) for the loss.
    launch_softmax_cross_entropy_loss_backward_optimized(stream_,
                                               (float*)grad_logits.d_ptr_,
                                               (const float*)logits.d_ptr_,
                                               (const int*)labels.d_ptr_,
                                               (float*)loss_output.d_ptr_,
                                               input_ids.dim_size(0), // B
                                               input_ids.dim_size(1), // S
                                               config_.vocab_size);   // V
    HIP_CHECK(hipGetLastError()); // Check after kernel launch

    // --- Backward Pass ---
    GpuTensor grad_sequence_output; // (B, S, H) - Grad w.r.t. output of BertModel
    grad_sequence_output.allocate(sequence_output.dims_);

    lm_head_->backward(blas_handle_, stream_, grad_logits, head_cache, grad_sequence_output);
    bert_model_->backward(blas_handle_, stream_, grad_sequence_output, model_cache);

    // --- Optimizer Step ---
    optimizer_->step();

    // Synchronize stream to ensure all operations are complete, especially if loss is read by CPU next.
    HIP_CHECK(hipStreamSynchronize(stream_));
}
