#ifndef BERT_COMPONENTS_HIP_HPP
#define BERT_COMPONENTS_HIP_HPP

#include "common_hip.hpp"
#include "nn_layers_hip.hpp"
#include "attention_hip.hpp"
#include <vector>
#include <string>
#include <memory> // For std::unique_ptr

// --- BertConfig Struct ---
struct BertConfig {
    int vocab_size = 30522;
    int hidden_size = 768;
    int num_hidden_layers = 12;
    int num_attention_heads = 12;
    int intermediate_size = 3072; // For FFN
    float hidden_dropout_prob = 0.1f;
    float attention_probs_dropout_prob = 0.1f; // Usually same as hidden_dropout_prob
    int max_position_embeddings = 512;
    float layer_norm_eps = 1e-12f; // Common epsilon for LayerNorm
    // Add other config fields as necessary, e.g., initializer_range
};


// --- Cache Structs for BERT Components ---

struct BertEmbeddingsCache {
    // Input tensors (pointers)
    const GpuTensor* input_ids_ptr = nullptr;
    const GpuTensor* token_type_ids_ptr = nullptr;

    // Intermediate tensors
    GpuTensor summed_embeddings; // Before LayerNorm and Dropout: Word + Pos + TokType
    // The final output of embeddings layer is typically passed as GpuTensor& output to forward method
    // Or stored here if it's an intermediate step for a larger model.
    // Let's assume embeddings_output is the final output of this module.
    GpuTensor embeddings_output; // After LayerNorm and Dropout

    // Caches for sub-modules
    LayerNormCache layernorm_cache;
    DropoutCache dropout_cache;
};

struct BertLayerCache {
    BertAttentionCache attention_cache; // Cache for the BertAttention sub-module

    // Feed-Forward Network (FFN) part
    DenseLayerCache ffn_intermediate_dense_cache; // For the first dense layer in FFN (which includes GELU)
    // GeluCache ffn_gelu_cache;                  // No longer needed as DenseLayer incorporates GELU and its cache requirements.
    DenseLayerCache ffn_output_dense_cache;       // For the second dense layer (output) in FFN
    DropoutCache ffn_output_dropout_cache;        // Dropout after FFN dense output
    LayerNormCache ffn_output_layernorm_cache;    // LayerNorm after FFN and residual

    const GpuTensor* layer_input_ptr = nullptr; // Input to this BertLayer (for residual connections)
    GpuTensor ffn_input_after_attention; // Stores output of attention block (input to FFN's residual sum)
                                         // Or more precisely, output of attention's LayerNorm.
};

struct BertEncoderCache {
    std::vector<BertLayerCache> layer_caches;
    BertEncoderCache(int num_layers) : layer_caches(num_layers) {}
    // No specific tensors here, just a collection of layer caches
};

struct BertModelCache {
    BertEmbeddingsCache embeddings_cache;
    BertEncoderCache encoder_cache;
    // BertPoolerCache pooler_cache; // If BertPooler is implemented

    // The final sequence output from the encoder.
    // It's better if BertModel::forward takes GpuTensor& sequence_output as argument.
    // If stored in cache, need to manage its lifecycle. For now, assume it's an output param.
    // GpuTensor final_sequence_output;

    BertModelCache(int num_encoder_layers) : encoder_cache(num_encoder_layers) {}
};


// --- BERT Component Class Definitions ---

class BertEmbeddings {
private:
    const BertConfig& config_; // Store config by reference
    Parameter word_embeddings_;
    Parameter position_embeddings_;
    Parameter token_type_embeddings_;
    LayerNorm layernorm_;
    Dropout dropout_;

public:
    BertEmbeddings(const BertConfig& config, const std::string& name_prefix);

    // output: final embedding tensor (B, S, H)
    void forward(hipStream_t stream,
                 const GpuTensor& input_ids,        // (B, S), int32
                 const GpuTensor& token_type_ids,   // (B, S), int32 (optional, pass empty if not used)
                 GpuTensor& output_embeddings,      // Output tensor
                 BertEmbeddingsCache& cache,
                 bool is_training);

    // grad_output_embeddings: Gradient w.r.t. the output of this module
    // Gradients for input_ids and token_type_ids are typically not computed as they are discrete.
    void backward(hipStream_t stream,
                  const GpuTensor& grad_output_embeddings,
                  BertEmbeddingsCache& cache);

    std::vector<Parameter*> get_parameters();
    Parameter* get_word_embedding_params(); // For sharing with LM head
};


class BertLayer {
private:
    // const BertConfig& config_;
    BertAttention attention_;

    // FFN (BertIntermediate and BertOutput)
    DenseLayer ffn_intermediate_dense_; // BertIntermediate
    // Gelu ffn_activation_;              // GELU is separate in HF, but often fused or follows dense. DenseLayer now incorporates GELU.
    DenseLayer ffn_output_dense_;       // BertOutput dense part
    Dropout ffn_output_dropout_;
    LayerNorm ffn_output_layernorm_;   // LayerNorm for the FFN output + residual

public:
    BertLayer(const BertConfig& config, const std::string& name_prefix);

    // hidden_states_output: final output of this layer (B,S,H)
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_hidden_states, // (B,S,H)
                 const GpuTensor& attention_mask,
                 GpuTensor& output_hidden_states,      // Output tensor
                 BertLayerCache& cache,
                 bool is_training);

    // grad_output_hidden_states: Gradient w.r.t. the output of this layer
    // grad_input_hidden_states: Gradient w.r.t. the input to this layer
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output_hidden_states,
                  BertLayerCache& cache,
                  GpuTensor& grad_input_hidden_states);

    std::vector<Parameter*> get_parameters();
};


class BertEncoder {
private:
    // const BertConfig& config_;
    std::vector<std::unique_ptr<BertLayer>> layers_;

public:
    BertEncoder(const BertConfig& config, const std::string& name_prefix);

    // final_hidden_states: Output of the last BertLayer (B,S,H)
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& initial_hidden_states, // Output from Embeddings (B,S,H)
                 const GpuTensor& attention_mask,
                 GpuTensor& final_hidden_states,         // Output tensor
                 BertEncoderCache& cache,
                 bool is_training);

    // grad_final_hidden_states: Gradient w.r.t. the output of the encoder
    // grad_initial_hidden_states: Gradient w.r.t. the input to the encoder
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_final_hidden_states,
                  BertEncoderCache& cache,
                  GpuTensor& grad_initial_hidden_states);

    std::vector<Parameter*> get_parameters();
};


class BertModel { // This is the main BERT model (Embeddings + Encoder)
private:
    const BertConfig& config_;
    BertEmbeddings embeddings_;
    BertEncoder encoder_;
    // BertPooler pooler_; // Optional: if pooling is needed

public:
    BertModel(const BertConfig& config, const std::string& name_prefix = "bert");

    // sequence_output: (B,S,H) - output of the last encoder layer
    // pooled_output: (B,H) - output of the pooler (optional)
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_ids,
                 const GpuTensor& attention_mask,
                 const GpuTensor& token_type_ids, // Optional
                 GpuTensor& sequence_output,      // Output: hidden states from last layer
                 // GpuTensor& pooled_output,     // Output: pooled output (if pooler exists)
                 BertModelCache& cache,
                 bool is_training);

    // grad_sequence_output: Gradient w.r.t. sequence_output
    // grad_pooled_output: Gradient w.r.t. pooled_output (if pooler exists)
    // Gradients for inputs (input_ids etc.) are not computed here.
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_sequence_output,
                  // const GpuTensor& grad_pooled_output, // If pooler exists
                  BertModelCache& cache);

    std::vector<Parameter*> get_parameters();
    Parameter* get_word_embedding_params(); // For sharing with LM head
};


#endif // BERT_COMPONENTS_HIP_HPP
