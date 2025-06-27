#ifndef ATTENTION_HIP_HPP
#define ATTENTION_HIP_HPP

#include "common_hip.hpp"
#include "nn_layers_hip.hpp" // For DenseLayer, Dropout, LayerNorm and their caches
#include "bert_components_hip.hpp" // For full BertConfig definition

// No need to forward declare BertConfig if included above

// --- Cache Structs for Attention Layers ---

struct SelfAttentionCache {
    // Inputs to SelfAttention (stored by pointer for backward pass)
    const GpuTensor* input_hidden_states = nullptr; // Shape: (batch_size, seq_len, hidden_size)
    const GpuTensor* attention_mask = nullptr;    // Shape: (batch_size, 1, 1, seq_len) or (B, S, S) etc.

    // Intermediate tensors computed during forward
    GpuTensor q_proj; // Query projection: (B, S, H)
    GpuTensor k_proj; // Key projection: (B, S, H)
    GpuTensor v_proj; // Value projection: (B, S, H)

    GpuTensor q_reshaped; // Query reshaped for multi-head: (B, N, S, A)
    GpuTensor k_reshaped; // Key reshaped for multi-head: (B, N, S, A)
    GpuTensor v_reshaped; // Value reshaped for multi-head: (B, N, S, A)

    GpuTensor attention_scores; // (B, N, S_q, S_k)
    GpuTensor attention_probs;  // (B, N, S_q, S_k), after softmax and possibly dropout

    // If attention_probs dropout is separate:
    DropoutCache attention_probs_dropout_cache;

    GpuTensor context_reshaped; // Context layer reshaped: (B, N, S_q, A)
    GpuTensor context_layer;    // Final context layer: (B, S_q, H)

    // Caches for the internal DenseLayers (Q, K, V projections)
    // These are needed if DenseLayer::backward requires its cache.
    // The original code's DenseLayerCache was minimal (just const GpuTensor* input).
    // Let's assume DenseLayerCache is sufficient as defined.
    DenseLayerCache q_dense_cache;
    DenseLayerCache k_dense_cache;
    DenseLayerCache v_dense_cache;

    // To store pointers to the input tensors if needed by backward pass logic directly
    // const GpuTensor* input_tensor_ptr; (already have input_hidden_states)
};

struct BertAttentionCache {
    SelfAttentionCache self_attention_cache; // Cache for the BertSelfAttention part
    // BertSelfOutput part cache:
    // The output dense layer is just a DenseLayer, its cache is part of its forward call.
    // However, we need to store the input to the output_dense layer if it's different from self_attention output.
    // Assuming self_attention_cache.context_layer is the input to output_dense.
    DenseLayerCache output_dense_cache;
    DropoutCache output_dropout_cache;
    LayerNormCache output_layernorm_cache;

    const GpuTensor* attention_input = nullptr; // Input to the whole BertAttention block (residual connection)
};


// --- Attention Layer Class Definitions ---

class BertSelfAttention {
private:
    // const BertConfig& config_; // Store if needed for num_heads, head_size etc.
    DenseLayer query_, key_, value_;
    Dropout dropout_; // Dropout for attention probabilities

    int num_attention_heads_;
    int attention_head_size_;
    // float scale_factor_; // 1.0f / sqrtf(attention_head_size_)

public:
    BertSelfAttention(const BertConfig& config, const std::string& name_prefix);

    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& hidden_states, // Input (B, S, H)
                 const GpuTensor& attention_mask, // Mask (e.g., B, 1, 1, S)
                 SelfAttentionCache& cache,
                 bool is_training);

    // grad_input is the gradient w.r.t. hidden_states passed to forward
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_context_layer_output, // Gradient from the layer above
                  SelfAttentionCache& cache,
                  GpuTensor& grad_input_hidden_states);

    std::vector<Parameter*> get_parameters();
};

class BertAttention {
private:
    // const BertConfig& config_;
    BertSelfAttention self_attention_;
    DenseLayer output_dense_; // BertSelfOutput dense layer
    Dropout output_dropout_;
    LayerNorm output_layernorm_; // LayerNorm after residual connection

public:
    BertAttention(const BertConfig& config, const std::string& name_prefix);

    // Output of this layer is (attention_output + residual) -> LayerNorm
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_tensor, // Input for residual connection (B,S,H)
                 const GpuTensor& attention_mask,
                 BertAttentionCache& cache,
                 bool is_training);

    // grad_input is w.r.t. input_tensor
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output_layernorm, // Gradient from the layer above
                  BertAttentionCache& cache,
                  GpuTensor& grad_input_tensor);

    std::vector<Parameter*> get_parameters();
};


#endif // ATTENTION_HIP_HPP
