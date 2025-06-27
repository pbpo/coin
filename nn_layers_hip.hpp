#ifndef NN_LAYERS_HIP_HPP
#define NN_LAYERS_HIP_HPP

#include "common_hip.hpp"
// #include "hip_kernels.hpp" // Not strictly needed here if methods in .cpp include it.
#include "bert_components_hip.hpp" // For full BertConfig definition

// No need to forward declare BertConfig if included above

// --- Cache Structs for NN Layers ---
struct DenseLayerCache {
    const GpuTensor* input_to_gemm = nullptr; // Original input to the GEMM operation
    GpuTensor output_before_gelu; // Output of GEMM + Bias, input to GELU. Owned by cache.
    // Other intermediate tensors if needed
};

struct LayerNormCache {
    GpuTensor mean; // Shape: (B) or (Batch elements corresponding to normalized dim)
    GpuTensor rstd; // Shape: (B)
    const GpuTensor* input = nullptr; // To store a pointer to the input tensor for backward pass
};

struct GeluCache {
    const GpuTensor* input = nullptr; // Input to GELU for backward pass
};

struct DropoutCache {
    GpuTensor mask; // Shape: same as input/output
    // No need to store input typically, as grad_output is the starting point for backward
};

// --- NN Layer Class Definitions ---

class Dropout {
private:
    float dropout_prob_;
    float scale_; // 1.0 / (1.0 - dropout_prob_) for inverted dropout

public:
    Dropout(float prob);
    // Takes input, modifies it in-place if output is same as input, or writes to output.
    // Cache stores the mask.
    void forward(hipStream_t stream, GpuTensor& input_output, DropoutCache& cache, bool is_training);
    // Takes grad_output, computes grad_input.
    void backward(hipStream_t stream, GpuTensor& grad_input, const GpuTensor& grad_output, const DropoutCache& cache);
};

class DenseLayer {
private:
    std::string name_;
    // BertConfig might not be directly needed here if all params are passed,
    // but can be useful for consistency or if some config values affect layer behavior.
    // const BertConfig& config_; // Optional, depending on design

public:
    Parameter params; // Contains weights and optionally bias
    const BertConfig& config_; // Store BertConfig if methods need it beyond constructor
    DenseLayer(int in_features, int out_features, const BertConfig& config, const std::string& name, bool has_bias = true);

    // output = activation(input * W^T + bias)
    // The activation (e.g. GELU) is handled outside or by a specialized DenseLayer if fused.
    // This basic DenseLayer just performs: input * W^T + bias
    // The original code's DenseLayer::forward includes GELU. This will be part of its .cpp
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input, GpuTensor& output,
                 DenseLayerCache& cache);

    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output, const DenseLayerCache& cache,
                  GpuTensor& grad_input);

    std::vector<Parameter*> get_parameters() { return {&params}; }

};

class LayerNorm {
private:
    // const BertConfig& config_; // Store if needed for eps or other params
    float epsilon_;
    int normalized_shape_size_; // The size of the dimension being normalized (e.g., hidden_size)
public:
    Parameter params; // gamma (weights) and beta (bias)

    LayerNorm(int normalized_dim_size, float eps, const std::string& name); // eps from BertConfig usually

    void forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, LayerNormCache& cache);
    void backward(hipStream_t stream, const GpuTensor& grad_output, const LayerNormCache& cache, GpuTensor& grad_input);
    std::vector<Parameter*> get_parameters() { return {&params}; }
};

class Gelu {
public:
    Gelu() = default; // No parameters

    // Output = GELU(input)
    // The original DenseLayer fused GELU. If this Gelu class is used standalone:
    void forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, GeluCache& cache);

    // Computes grad_input = grad_output * dGELU/dInput
    void backward(hipStream_t stream, GpuTensor& grad_input, const GpuTensor& grad_output, const GeluCache& cache);
    // No parameters to return for Gelu itself
};


#endif // NN_LAYERS_HIP_HPP
