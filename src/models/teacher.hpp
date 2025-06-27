#pragma once

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <memory> // For std::unique_ptr, std::shared_ptr
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error

// HIP and rocBLAS Error Checking Macros
#define HIP_CHECK(error) \
    do { \
        hipError_t err = error; \
        if (err != hipSuccess) { \
            fprintf(stderr, "HIP error in %s at line %d: %s (%d)\n", \
                    __FILE__, __LINE__, hipGetErrorString(err), err); \
            throw std::runtime_error("HIP error: " + std::string(hipGetErrorString(err))); \
        } \
    } while (0)

#define ROCBLAS_CHECK(status) \
    do { \
        rocblas_status stat = status; \
        if (stat != rocblas_status_success) { \
            fprintf(stderr, "rocBLAS error in %s at line %d: %s (%d)\n", \
                    __FILE__, __LINE__, rocblas_status_to_string(stat), stat); \
            throw std::runtime_error("rocBLAS error: " + std::string(rocblas_status_to_string(stat))); \
        } \
    } while (0)


// Configuration for the BERT model
struct BertConfig {
    int vocab_size = 30000; // Default, should be set from tokenizer
    int hidden_size = 256;  // d_model
    int num_hidden_layers = 4;
    int num_attention_heads = 1; // Python code used 1, ensure this is intended. BERT base usually has more.
    int intermediate_size = 512; // Size of the feed-forward layer
    int max_position_embeddings = 126; // Max sequence length
    float layer_norm_eps = 1e-12; // Epsilon for LayerNorm
    // Dropout probabilities are not directly implemented in this conceptual HIP version
    // float hidden_dropout_prob = 0.1;
    // float attention_probs_dropout_prob = 0.1;

    BertConfig() = default;
};

// Forward declarations for model components
class GpuTensor;
class BertEmbeddings;
class BertSelfAttention;
class BertSelfOutput;
class BertAttention;
class BertIntermediate;
class BertOutput;
class BertLayer;
class BertEncoder;
class BertPooler; // Optional, not used in MaskedLM typically
class BertPredictionHeadTransform;
class BertLMPredictionHead;
class BertModel;
class CANBertForMaskedLM;

// Simple GPU Tensor class to manage memory (float type only for now)
// Owns the GPU memory.
class GpuTensor {
public:
    // batch_size, seq_len, hidden_size etc.
    // For a 2D tensor (e.g. batch_size, seq_len), dims would be {batch_size, seq_len}
    // For a 3D tensor (e.g. batch_size, seq_len, hidden_size), dims would be {batch_size, seq_len, hidden_size}
    GpuTensor(const std::vector<int>& dimensions, const std::string& name = "tensor");
    // Create an uninitialized tensor (no memory allocation)
    GpuTensor(const std::string& name = "tensor_uninit");
    // Create a GpuTensor that does NOT own the data (e.g., for slicing or views)
    // This is advanced and needs careful lifetime management. For now, all tensors own data.
    // GpuTensor(float* existing_ptr, const std::vector<int>& dimensions, const std::string& name = "tensor_view");

    ~GpuTensor();

    // Allocate memory based on current dims_
    void allocate();
    void free();

    // Copy data
    void to_gpu(const std::vector<float>& cpu_data);
    std::vector<float> to_cpu() const;
    void copy_from_gpu(const GpuTensor& src); // Device to Device copy

    // Accessors
    float* d_ptr() const { return d_ptr_; }
    const std::vector<int>& dims() const { return dims_; }
    size_t num_elements() const { return num_elements_; }
    const std::string& name() const { return name_; }
    void set_name(const std::string& name) { name_ = name; }
    bool is_allocated() const { return allocated_; }

    // Reshape (careful: only changes metadata, doesn't reallocate or copy unless explicitly done)
    // For simplicity, a true reshape might require creating a new tensor.
    // This version just updates dims and num_elements if they match.
    void reshape(const std::vector<int>& new_dims);

    // Get specific dimension size
    int dim_size(int i) const;
    int batch_size() const { return dims_.empty() ? 0 : dims_[0]; }
    int seq_len() const { return dims_.size() < 2 ? 0 : dims_[1]; }
    int hidden_size() const { return dims_.size() < 3 ? 0 : dims_[2]; } // Common use for 3D tensors

private:
    float* d_ptr_ = nullptr;
    std::vector<int> dims_;
    size_t num_elements_ = 0;
    std::string name_;
    bool allocated_ = false;
    // bool owns_data_ = true; // For views/slices later
};

// Helper for creating parameters (weights/biases)
// Manages its own GpuTensor for weights.
class Parameter {
public:
    GpuTensor weights;
    // Biases are optional for some layers or can be fused.
    // GpuTensor biases;
    // bool has_bias;

    Parameter(const std::vector<int>& dims, const std::string& name)
        : weights(dims, name) /*, has_bias(false) */ {}

    // Parameter(const std::vector<int>& weight_dims, const std::vector<int>& bias_dims, const std::string& name)
    //     : weights(weight_dims, name + "_w"), biases(bias_dims, name + "_b"), has_bias(true) {}

    // TODO: Add initialization methods (e.g., Xavier, Kaiming, from file)
    void initialize_random(float mean = 0.0f, float stddev = 0.02f); // Simple random init
    void initialize_zeros();
    void initialize_ones();

    // Load weights from a file (e.g., a flat binary file of floats)
    // bool load_from_file(const std::string& path);
};


// --- Layer Implementations (Declarations) ---
// Note: For simplicity, these layers will often take rocblas_handle and config by const ref in their methods.
// Alternatively, these could be passed during construction if they don't change.

class DenseLayer {
public:
    DenseLayer(int input_features, int output_features, const BertConfig& config, const std::string& name);
    void forward(rocblas_handle blas_handle, const GpuTensor& input, GpuTensor& output, GpuTensor* bias = nullptr); // input: (batch, features_in), output: (batch, features_out)
    Parameter weights; // (output_features, input_features) for standard GEMM: C = A*B + D, A=input, B=weights^T
                       // If using weights directly (input_features, output_features) need to transpose B in sgemm
    Parameter bias;    // (output_features)
    std::string name;
private:
    const BertConfig& config_;
};

class LayerNorm {
public:
    LayerNorm(int normalized_shape_feat_dim, float eps, const BertConfig& config, const std::string& name); // normalized_shape is usually hidden_size
    void forward(const GpuTensor& input, GpuTensor& output); // input/output: (batch, seq_len, hidden_size)
    Parameter gamma; // weight, (hidden_size)
    Parameter beta;  // bias, (hidden_size)
    float eps_;
    std::string name;
private:
    const BertConfig& config_;
};


class BertEmbeddings {
public:
    BertEmbeddings(const BertConfig& config);
    void forward(const GpuTensor& input_ids, /*const GpuTensor& token_type_ids,*/ GpuTensor& output_embeddings);
    // output_embeddings shape: (batch_size, seq_len, hidden_size)

    Parameter word_embeddings;         // (vocab_size, hidden_size)
    Parameter position_embeddings;     // (max_position_embeddings, hidden_size)
    // Parameter token_type_embeddings; // (type_vocab_size, hidden_size) - Simplified: not using token_type_ids

    LayerNorm layer_norm;
    // Dropout dropout; // Dropout not implemented

private:
    const BertConfig& config_;
    std::unique_ptr<GpuTensor> position_ids_; // To generate on the fly if needed
    void create_position_ids(int batch_size, int seq_len);
};

class BertSelfAttention {
public:
    BertSelfAttention(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, const GpuTensor& attention_mask, GpuTensor& context_layer);
    // hidden_states: (batch_size, seq_len, hidden_size)
    // attention_mask: (batch_size, 1, 1, seq_len) or (batch_size, seq_len) - needs processing
    // context_layer: (batch_size, seq_len, hidden_size)

    DenseLayer query;
    DenseLayer key;
    DenseLayer value;

    // Dropout attention_dropout; // Not implemented

    int num_attention_heads_;
    int attention_head_size_;
private:
    const BertConfig& config_;
    std::string name_prefix_;
    void transpose_for_scores(const GpuTensor& input, GpuTensor& output_transposed, int batch_size, int seq_len);
};

class BertSelfOutput {
public:
    BertSelfOutput(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, const GpuTensor& input_tensor, GpuTensor& output_tensor);
    // hidden_states: (batch_size, seq_len, hidden_size) - from attention
    // input_tensor: (batch_size, seq_len, hidden_size) - residual connection input
    // output_tensor: (batch_size, seq_len, hidden_size)

    DenseLayer dense;
    LayerNorm layer_norm;
    // Dropout dropout; // Not implemented
private:
    const BertConfig& config_;
};

class BertAttention {
public:
    BertAttention(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, const GpuTensor& attention_mask, GpuTensor& output_tensor);

    BertSelfAttention self_attention;
    BertSelfOutput self_output;
private:
    const BertConfig& config_;
};

class BertIntermediate {
public:
    BertIntermediate(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, GpuTensor& output_tensor);
    // hidden_states: (batch, seq_len, hidden_size)
    // output_tensor: (batch, seq_len, intermediate_size)

    DenseLayer dense;
    // Activation function (GeLU) will be applied in a kernel
private:
    const BertConfig& config_;
};

class BertOutput {
public:
    BertOutput(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, const GpuTensor& input_tensor, GpuTensor& output_tensor);
    // hidden_states: (batch, seq_len, intermediate_size) - from intermediate
    // input_tensor: (batch, seq_len, hidden_size) - residual connection input (from BertAttention output)
    // output_tensor: (batch, seq_len, hidden_size)

    DenseLayer dense;
    LayerNorm layer_norm;
    // Dropout dropout; // Not implemented
private:
    const BertConfig& config_;
};

class BertLayer {
public:
    BertLayer(const BertConfig& config, int layer_idx);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, const GpuTensor& attention_mask, GpuTensor& output_tensor);

    BertAttention attention;
    BertIntermediate intermediate;
    BertOutput output;
private:
    const BertConfig& config_;
};

class BertEncoder {
public:
    BertEncoder(const BertConfig& config);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, const GpuTensor& attention_mask, std::vector<GpuTensor>& all_hidden_states, GpuTensor& last_hidden_state);
    // all_hidden_states will store output of each layer if needed, otherwise only last_hidden_state is primary output.

    std::vector<std::shared_ptr<BertLayer>> layers;
private:
    const BertConfig& config_;
};


// For Masked LM, we need the prediction head.
class BertPredictionHeadTransform {
public:
    BertPredictionHeadTransform(const BertConfig& config, const std::string& name_prefix);
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, GpuTensor& output_states);
    // hidden_states: (batch, seq_len, hidden_size)
    // output_states: (batch, seq_len, hidden_size) - after Dense and GeLU

    DenseLayer dense;
    LayerNorm layer_norm;
    // GeLU activation will be a kernel
private:
    const BertConfig& config_;
};

class BertLMPredictionHead {
public:
    BertLMPredictionHead(const BertConfig& config, const GpuTensor& word_embedding_weights); // Needs word embedding weights for tying
    void forward(rocblas_handle blas_handle, const GpuTensor& hidden_states, GpuTensor& output_logits);
    // hidden_states: (batch, seq_len, hidden_size) from BertPredictionHeadTransform
    // output_logits: (batch, seq_len, vocab_size)

    BertPredictionHeadTransform transform;
    // The decoder is a Dense layer whose weights are tied to word_embeddings.
    // It also has a bias.
    // DenseLayer decoder; // This would be: vocab_size (out) x hidden_size (in)
    Parameter bias; // (vocab_size)
    const GpuTensor& word_embedding_weights_tied_; // Reference to the embedding table (vocab_size, hidden_size)
private:
    const BertConfig& config_;
};


// Main BERT Model (as used in HuggingFace `BertModel`)
class BertModel {
public:
    BertModel(const BertConfig& config);
    // Returns the last hidden state. If all_encoder_outputs is true, fills all_hidden_states_
    void forward(rocblas_handle blas_handle, const GpuTensor& input_ids, /*const GpuTensor& token_type_ids,*/ const GpuTensor& attention_mask, GpuTensor& sequence_output);
    // sequence_output: (batch, seq_len, hidden_size) - last hidden state of encoder

    BertEmbeddings embeddings;
    BertEncoder encoder;
    // BertPooler pooler; // Not typically used for MaskedLM straight up

    const GpuTensor& get_word_embedding_weights() const { return embeddings.word_embeddings.weights; }

private:
    const BertConfig& config_;
    // Internal tensors to avoid reallocation if sizes are consistent
    std::unique_ptr<GpuTensor> embedding_output_;
    std::vector<GpuTensor> all_encoder_layer_outputs_; // If needed
};


// The final model for Masked Language Modeling
class CANBertForMaskedLM {
public:
    CANBertForMaskedLM(const BertConfig& config);
    ~CANBertForMaskedLM();

    // Forward pass: takes input_ids and attention_mask, returns logits over vocab
    // input_ids: (batch_size, seq_len) of token IDs
    // attention_mask: (batch_size, seq_len) of 0s and 1s (will be processed for attention mechanism)
    // prediction_scores (logits): (batch_size, seq_len, vocab_size)
    void forward(const GpuTensor& input_ids, const GpuTensor& attention_mask, GpuTensor& prediction_scores);

    // Computes loss (e.g., CrossEntropy) between prediction_scores and labels
    // labels: (batch_size, seq_len) of true token IDs (or -100 for non-masked tokens)
    // loss_output: A GpuTensor holding a single float value for the batch loss (or per-token losses)
    void compute_loss(const GpuTensor& prediction_scores, const GpuTensor& labels, GpuTensor& loss_output);

    // Helper to create an attention mask suitable for BERT from a simple batch x seq_len mask
    void create_extended_attention_mask(const GpuTensor& input_mask, GpuTensor& output_bert_mask);

    // Initialize all parameters (weights and biases) in the model
    void initialize_parameters(float mean = 0.0f, float stddev = 0.02f);


    // For external access if needed (e.g. saving model weights)
    BertModel& get_bert_model() { return *bert_; }
    BertLMPredictionHead& get_lm_head() { return *cls_; }
    rocblas_handle get_blas_handle() { return blas_handle_; }
    const BertConfig& get_config() const { return config_; }

private:
    BertConfig config_; // Store a copy of the config
    std::unique_ptr<BertModel> bert_;
    std::unique_ptr<BertLMPredictionHead> cls_; // "cls" is often used for prediction heads in HF

    rocblas_handle blas_handle_;

    // Intermediate tensors to avoid frequent reallocations if batch/seq_len are stable
    std::unique_ptr<GpuTensor> sequence_output_buffer_; // Output from BertModel
    std::unique_ptr<GpuTensor> extended_attention_mask_buffer_; // For attention mechanism
};

// --- Kernel Launchers (declarations, implementations in .hip.cpp) ---
// These are helper functions to launch HIP kernels.

// Example: LayerNorm kernel
void launch_layer_norm_forward(hipStream_t stream, float* output, float* mean, float* inv_variance,
                               const float* input, const float* gamma, const float* beta,
                               int batch_size, int seq_len, int hidden_size, float epsilon);

// Example: Softmax kernel (for attention)
void launch_softmax_forward(hipStream_t stream, float* output, const float* input,
                            int batch_size, int num_heads, int seq_len, int head_dim_or_key_len); // Adapts to context

// Example: GeLU activation kernel
void launch_gelu_forward(hipStream_t stream, float* output, const float* input, size_t num_elements);

// Example: Embedding lookup and aggregation
void launch_embedding_lookup_sum(hipStream_t stream, float* output_embeddings,
                                 const int* input_ids, const int* position_ids, /*const int* token_type_ids,*/
                                 const float* word_embeddings, const float* position_embeddings, /*const float* token_type_embeddings,*/
                                 int batch_size, int seq_len, int hidden_size, int vocab_size, int max_pos_emb/*, int type_vocab_size*/);

// Example: Bias addition and residual connection kernel
void launch_add_bias_residual(hipStream_t stream, float* output, const float* input,
                              const float* residual, const float* bias,
                              size_t batch_dim, size_t seq_dim, size_t feature_dim);
// Simplified add bias (used after GEMMs often)
void launch_add_bias(hipStream_t stream, float* output_and_input, const float* bias,
                     size_t batch_seq_dim, size_t feature_dim);


// Example: Transpose for attention scores
void launch_transpose_for_scores(hipStream_t stream, const float* input, float* output,
                                 int batch_size, int seq_len, int num_heads, int head_size);
void launch_transpose_bht_to_bth(hipStream_t stream, const float* input, float* output,
                                 int batch_size, int num_heads, int seq_len, int head_size);


// Cross Entropy Loss Kernel
void launch_softmax_cross_entropy_loss_forward(hipStream_t stream,
                                               const float* logits,      // Input: (batch_size, seq_len, vocab_size)
                                               const int* labels,        // Input: (batch_size, seq_len)
                                               float* per_token_loss,    // Output: (batch_size, seq_len)
                                               float* total_loss,        // Output: scalar (on GPU, sum of valid losses)
                                               int batch_size,
                                               int seq_len,
                                               int vocab_size,
                                               int ignore_index = -100);

// Kernel to sum up masked losses
void launch_sum_masked_losses(hipStream_t stream, const float* per_token_loss, const int* labels,
                              float* total_batch_loss, int batch_size, int seq_len, int ignore_index);

// Kernel for creating the extended attention mask
void launch_create_extended_attention_mask(hipStream_t stream, const int* input_mask_2d, float* output_mask_4d,
                                           int batch_size, int seq_len);

#endif // TEACHER_HPP (Guard, though #pragma once is common)
