#ifndef LANGUAGE_MODEL_HIP_HPP
#define LANGUAGE_MODEL_HIP_HPP

#include "common_hip.hpp"
#include "bert_components_hip.hpp" // For BertModel, BertConfig, Parameter
#include "nn_layers_hip.hpp"       // For DenseLayer, LayerNorm, Gelu and their caches
#include "optimizer_hip.hpp"       // For AdamWOptimizer (forward declared or included if CANBert owns it)
#include <memory>                  // For std::unique_ptr

// Forward declaration for AdamWOptimizer if not fully included
// class AdamWOptimizer;


// --- Cache Struct for BertLMPredictionHead ---
struct BertLMPredictionHeadCache {
    // Input to this head (usually sequence_output from BertModel)
    const GpuTensor* hidden_states_input = nullptr;

    // Intermediate tensors for the transform part
    GpuTensor transform_dense_output;
    GpuTensor transform_gelu_output; // Output of GELU (input to LayerNorm)
    GpuTensor transform_layernorm_output; // Output of LayerNorm (input to decoder matmul)
                                          // This is `cache.transformed_states` in user's original code

    // Caches for sub-modules
    DenseLayerCache transform_dense_cache;
    GeluCache transform_gelu_cache; // If GELU is a separate layer
    LayerNormCache transform_layernorm_cache;

    // The final logits tensor (pointer, as it's usually an output parameter)
    // const GpuTensor* logits_output_ptr = nullptr; // Not strictly needed if logits is an output param of forward
};


// --- Language Model Head Class ---
class BertLMPredictionHead {
private:
    const BertConfig& config_; // Store by reference

    // "transform" part (Dense -> Activation -> LayerNorm)
    DenseLayer transform_dense_;
    // Gelu transform_act_fn_; // Original code had this separate
    LayerNorm transform_layernorm_;

    // "decoder" part (usually a Dense layer without activation, whose weights are tied to word_embeddings)
    // The weights are shared, so this layer itself might not own a Parameter object for weights.
    // It only has its own bias.
    Parameter decoder_bias_; // Name like "cls.predictions.bias" or "predictions.decoder.bias"
    Parameter& shared_word_embeddings_; // Reference to the word embedding table's Parameter object

public:
    BertLMPredictionHead(const BertConfig& config, Parameter& shared_word_embeddings_parameter, const std::string& name_prefix = "cls.predictions");

    // hidden_states: (B, S, H) - input from BertModel's encoder
    // logits: (B, S, V) - output prediction scores
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& hidden_states,
                 GpuTensor& logits, // Output tensor
                 BertLMPredictionHeadCache& cache);

    // grad_logits: (B, S, V) - gradient from the loss function
    // grad_hidden_states: (B, S, H) - gradient to be passed back to BertModel
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_logits,
                  BertLMPredictionHeadCache& cache,
                  GpuTensor& grad_hidden_states); // Output gradient tensor

    std::vector<Parameter*> get_parameters(); // Returns params owned by this head (transform layers, decoder_bias)
};


// --- Main Model Class for Masked Language Modeling ---
class CANBertForMaskedLM {
private:
    BertConfig config_; // Owns the config or takes by const ref
    std::unique_ptr<BertModel> bert_model_;
    std::unique_ptr<BertLMPredictionHead> lm_head_;
    std::unique_ptr<AdamWOptimizer> optimizer_; // Make this part of the class

    rocblas_handle blas_handle_;
    hipStream_t stream_;

public:
    CANBertForMaskedLM(const BertConfig& config);
    ~CANBertForMaskedLM();

    void initialize_parameters(float mean = 0.0f, float stddev = 0.02f);
    std::vector<Parameter*> get_parameters(); // Collects all parameters from bert_model and lm_head

    // Performs one training step: forward, loss, backward, optimizer step
    // loss_output will contain the scalar loss value (e.g., on a 1-element GpuTensor)
    void train_step(const GpuTensor& input_ids,
                    const GpuTensor& attention_mask,
                    const GpuTensor& token_type_ids, // Optional
                    const GpuTensor& labels,         // Target labels for MLM (B, S)
                    GpuTensor& loss_output);         // Output for the loss value

    // predict/inference method (optional, not shown in original user code's train_step)
    // void predict(const GpuTensor& input_ids, const GpuTensor& attention_mask, const GpuTensor& token_type_ids, GpuTensor& logits_output);
};


#endif // LANGUAGE_MODEL_HIP_HPP
