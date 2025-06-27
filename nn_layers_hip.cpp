#include "nn_layers_hip.hpp"
#include "hip_kernels.hpp" // For kernel launchers
#include "common_hip.hpp"  // For GpuTensor, Parameter, ROCBLAS_CHECK
#include <stdexcept>       // For runtime_error
#include <vector>
#include <string>
#include <cstdlib> // For std::rand for dropout seed

// --- Dropout Implementation ---
Dropout::Dropout(float prob) : dropout_prob_(prob) {
    if (prob < 0.0f || prob > 1.0f) {
        throw std::runtime_error("Dropout probability must be between 0 and 1.");
    }
    scale_ = (prob > 0.0f && prob < 1.0f) ? 1.0f / (1.0f - prob) : 1.0f;
}

void Dropout::forward(hipStream_t stream, GpuTensor& input_output, DropoutCache& cache, bool is_training) {
    if (is_training && dropout_prob_ > 0.0f) {
        if (!input_output.is_allocated()) {
            throw std::runtime_error("Input tensor for Dropout::forward is not allocated.");
        }
        cache.mask.allocate(input_output.dims_);
        // Using a simple seed generation. For more robust results, a better RNG seeding strategy might be needed.
        unsigned long long seed = static_cast<unsigned long long>(std::rand());

        launch_dropout_forward(stream,
                               (float*)input_output.d_ptr_, // Output is in-place modification of input
                               (float*)cache.mask.d_ptr_,
                               (const float*)input_output.d_ptr_, // Input
                               input_output.num_elements_,
                               dropout_prob_,
                               scale_,
                               seed);
    }
    // If not is_training or dropout_prob_ is 0, input_output remains unchanged (identity operation).
}

void Dropout::backward(hipStream_t stream, GpuTensor& grad_input, const GpuTensor& grad_output, const DropoutCache& cache) {
    if (!grad_output.is_allocated()) {
        throw std::runtime_error("grad_output tensor for Dropout::backward is not allocated.");
    }
    if (!grad_input.is_allocated() || grad_input.num_elements_ != grad_output.num_elements_) {
        grad_input.allocate(grad_output.dims_);
    }

    if (dropout_prob_ > 0.0f) {
        if (!cache.mask.is_allocated()) {
            throw std::runtime_error("Dropout mask in cache is not allocated for backward pass.");
        }
        launch_dropout_backward(stream,
                                (float*)grad_input.d_ptr_,
                                (const float*)grad_output.d_ptr_,
                                (const float*)cache.mask.d_ptr_,
                                grad_input.num_elements_,
                                scale_);
    } else {
        // If dropout was not applied (prob=0), then grad_input is just grad_output.
        grad_input.copy_from_gpu(grad_output, stream);
    }
}

// --- DenseLayer Implementation ---
DenseLayer::DenseLayer(int in_features, int out_features, const BertConfig& config, const std::string& name, bool has_bias)
    : params(has_bias ? Parameter({out_features, in_features}, {out_features}, name) : Parameter({out_features, in_features}, name)),
      name_(name), config_(config) {}


void DenseLayer::forward(rocblas_handle blas_handle, hipStream_t stream,
                         const GpuTensor& input, GpuTensor& output,
                         DenseLayerCache& cache) {
    if (!input.is_allocated() || !params.weights.is_allocated()) {
        throw std::runtime_error("Input or weights tensor not allocated for DenseLayer::forward for layer " + name_);
    }
    if (params.has_bias_ && !params.bias.is_allocated()) {
        throw std::runtime_error("Bias tensor not allocated for DenseLayer::forward for layer " + name_ + " but has_bias is true.");
    }

    // Input shape: (batch_dims..., in_features)
    // Weight shape: (out_features, in_features)
    // Output shape: (batch_dims..., out_features)

    // Determine M, K, N for GEMM: C_MxN = A_MxK * B_KxN
    // A = input (reshaped to 2D: [batch_size_combined, in_features])
    // B = weights (transposed: [in_features, out_features])
    // C = output (reshaped to 2D: [batch_size_combined, out_features])

    int in_features = params.weights.dim_size(1); // K
    int out_features = params.weights.dim_size(0); // N (for C) or M (for B if B is W)

    if (input.dims_.back() != in_features) {
        throw std::runtime_error("Input feature size mismatch for DenseLayer " + name_ +
                                 ". Expected " + std::to_string(in_features) +
                                 ", got " + std::to_string(input.dims_.back()));
    }

    size_t batch_size_combined = 1;
    std::vector<int> output_dims;
    for (size_t i = 0; i < input.dims_.size() - 1; ++i) {
        batch_size_combined *= input.dim_size(i);
        output_dims.push_back(input.dim_size(i));
    }
    output_dims.push_back(out_features);

    if (!output.is_allocated() || output.dims_ != output_dims) {
        output.allocate(output_dims);
    }

    int M = static_cast<int>(batch_size_combined);
    int K = in_features;
    int N = out_features;

    // rocBLAS expects column-major. Our weights are (out_features, in_features)
    // If we want C_MN = A_MK * B_KN, then B is W^T.
    // If W is (out, in), W^T is (in, out).
    // So, C_MxN_out = Input_MxK_in * (Weights_N_out_x_K_in)^T
    // Here, lda=K, ldb=K, ldc=N
    // A: input (M, K), lda = K
    // B: weights (N, K), ldb = K (rocblas_operation_transpose means B is used as KxN)
    // C: output (M, N), ldc = N

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta = 0.0f;

    // Perform: output = input * weights^T
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,         // opA
                                rocblas_operation_transpose,    // opB (use weights as KxN)
                                M, N, K,
                                &alpha,
                                (const float*)input.d_ptr_, K, // A, lda
                                (const float*)params.weights.d_ptr_, K, // B (stored as N,K), ldb
                                &beta,
                                (float*)output.d_ptr_, N));   // C, ldc

    // The original DenseLayer::forward in user code also calls launch_add_bias_gelu.
    // This means this DenseLayer is actually Dense+Bias+GELU.
    // If only bias, a separate kernel. If bias+gelu, a fused kernel.
    if (params.has_bias_) {
        // Assuming output from GEMM is the input to add_bias_gelu
        // M for add_bias_gelu is batch_size_combined, N is out_features
        // The bias vector is of size out_features.
        // The launch_add_bias_gelu_kernel expects bias[col]
        launch_add_bias_gelu_kernel(stream,
                                   (float*)output.d_ptr_,          // Output (in-place)
                                   (const float*)output.d_ptr_,    // Input (result from GEMM)
                                   (const float*)params.bias.d_ptr_,
                                   M, N);                          // M=batch_combined, N=out_features
    }

    cache.input = &input; // Store for backward pass
}

void DenseLayer::backward(rocblas_handle blas_handle, hipStream_t stream,
                          const GpuTensor& grad_output, const DenseLayerCache& cache,
                          GpuTensor& grad_input) {
    // grad_output shape: (batch_dims..., out_features)
    // input shape (from cache): (batch_dims..., in_features)
    // weights shape: (out_features, in_features)

    if (!grad_output.is_allocated() || !cache.input || !cache.input->is_allocated() || !params.weights.is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for DenseLayer::backward for layer " + name_);
    }

    int out_features = params.weights.dim_size(0);
    int in_features = params.weights.dim_size(1);

    if (grad_output.dims_.back() != out_features) {
        throw std::runtime_error("grad_output feature size mismatch for DenseLayer " + name_);
    }
    if (!grad_input.is_allocated() || grad_input.dims_ != cache.input->dims_) {
        grad_input.allocate(cache.input->dims_);
    }
    if (!params.grad_weights.is_allocated() || params.grad_weights.dims_ != params.weights.dims_) {
        params.grad_weights.allocate(params.weights.dims_);
    }
    if (params.has_bias_ && (!params.grad_bias.is_allocated() || params.grad_bias.dims_ != params.bias.dims_)) {
        params.grad_bias.allocate(params.bias.dims_);
    }


    size_t batch_size_combined = 1;
    for (size_t i = 0; i < grad_output.dims_.size() - 1; ++i) {
        batch_size_combined *= grad_output.dim_size(i);
    }

    int M_grad_out = static_cast<int>(batch_size_combined); // M for grad_output (batch_combined)
    int N_grad_out = out_features;                         // N for grad_output (out_features)
    int K_input_feat = in_features;                        // K for input (in_features)


    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta_one = 1.0f; // Use beta_one for accumulating gradients

    // 1. Calculate grad_input = grad_output * weights
    //    grad_input (M_grad_out, K_input_feat) = grad_output (M_grad_out, N_grad_out) * weights (N_grad_out, K_input_feat)
    //    A = grad_output (M, N_out), B = weights (N_out, K_in) -> C = grad_input (M, K_in)
    //    lda = N_out, ldb = K_in, ldc = K_in
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,       // opA
                                rocblas_operation_none,       // opB
                                M_grad_out, K_input_feat, N_grad_out,
                                &alpha,
                                (const float*)grad_output.d_ptr_, N_grad_out,      // A, lda
                                (const float*)params.weights.d_ptr_, K_input_feat, // B, ldb
                                &beta_one, // Accumulate if grad_input is not zeroed, otherwise beta_zero
                                (float*)grad_input.d_ptr_, K_input_feat));     // C, ldc


    // 2. Calculate grad_weights = input^T * grad_output (누적)
    //    grad_weights (K_input_feat, N_grad_out) = input^T (K_input_feat, M_grad_out) * grad_output (M_grad_out, N_grad_out)
    //    A = input (M, K_in), B = grad_output (M, N_out) -> C = grad_weights (K_in, N_out)
    //    opA = rocblas_operation_transpose
    //    lda_A_orig = K_in, ldb = N_out, ldc = N_out
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_transpose,  // opA (input is M,K -> use as K,M)
                                rocblas_operation_none,       // opB
                                K_input_feat, N_grad_out, M_grad_out,
                                &alpha,
                                (const float*)cache.input->d_ptr_, K_input_feat, // A (original M,K), lda
                                (const float*)grad_output.d_ptr_, N_grad_out,   // B (M,N), ldb
                                &beta_one, // Accumulate gradients
                                (float*)params.grad_weights.d_ptr_, N_grad_out)); // C (K,N), ldc

    // 3. Calculate grad_bias = sum(grad_output, axis=0) (누적)
    if (params.has_bias_) {
        // grad_output is (M_grad_out, N_grad_out) = (batch_combined, out_features)
        // grad_bias is (out_features)
        // We need to sum grad_output along the batch_combined dimension (axis 0).
        launch_reduce_sum_axis0_add_kernel(stream,
                                   (float*)params.grad_bias.d_ptr(),
                                   (const float*)grad_output.d_ptr(),
                                   M_grad_out, N_grad_out); // M=rows_to_reduce (batch_combined), N=cols_to_keep (out_features)
    }
}


// --- LayerNorm Implementation ---
LayerNorm::LayerNorm(int normalized_dim_size, float eps, const std::string& name)
    : params({normalized_dim_size}, {normalized_dim_size}, name), // gamma and beta
      epsilon_(eps), normalized_shape_size_(normalized_dim_size) {}

void LayerNorm::forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, LayerNormCache& cache) {
    if (!input.is_allocated() || !params.weights.is_allocated() || !params.bias.is_allocated()) {
        throw std::runtime_error("Input or parameters not allocated for LayerNorm::forward for layer " + params.name);
    }

    // Input shape: (..., normalized_shape_size_)
    // Output shape: same as input
    // Mean/Rstd shape: (..., 1) effectively, but stored as (num_rows_for_norm)

    if (input.dims_.back() != normalized_shape_size_) {
        throw std::runtime_error("LayerNorm input's last dimension mismatch. Expected " +
                                 std::to_string(normalized_shape_size_) + ", got " +
                                 std::to_string(input.dims_.back()));
    }
    if (!output.is_allocated() || output.dims_ != input.dims_) {
        output.allocate(input.dims_);
    }

    size_t B_rows = 1; // Number of rows to normalize independently
    for (size_t i = 0; i < input.dims_.size() - 1; ++i) {
        B_rows *= input.dim_size(i);
    }
    int C_cols = normalized_shape_size_;

    cache.mean.allocate({(int)B_rows});
    cache.rstd.allocate({(int)B_rows});
    cache.input = &input;

    launch_layer_norm_forward_optimized(stream,
                              (float*)output.d_ptr_,
                              (float*)cache.mean.d_ptr_,
                              (float*)cache.rstd.d_ptr_,
                              (const float*)input.d_ptr_,
                              (const float*)params.weights.d_ptr_, // gamma
                              (const float*)params.bias.d_ptr_,   // beta
                              static_cast<int>(B_rows), C_cols, epsilon_);
}

void LayerNorm::backward(hipStream_t stream, const GpuTensor& grad_output, const LayerNormCache& cache, GpuTensor& grad_input) {
    if (!grad_output.is_allocated() || !cache.input || !cache.input->is_allocated() ||
        !cache.mean.is_allocated() || !cache.rstd.is_allocated() ||
        !params.weights.is_allocated() || !params.bias.is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for LayerNorm::backward for layer " + params.name);
    }

    if (!grad_input.is_allocated() || grad_input.dims_ != grad_output.dims_) {
        grad_input.allocate(grad_output.dims_);
    }
    if (!params.grad_weights.is_allocated() || params.grad_weights.dims_ != params.weights.dims_) {
        params.grad_weights.allocate(params.weights.dims_);
    }
     if (!params.grad_bias.is_allocated() || params.grad_bias.dims_ != params.bias.dims_) {
        params.grad_bias.allocate(params.bias.dims_);
    }


    size_t B_rows = 1;
    for (size_t i = 0; i < cache.input->dims_.size() - 1; ++i) {
        B_rows *= cache.input->dim_size(i);
    }
    int C_cols = cache.input->dims_.back();

    launch_layer_norm_backward_optimized(stream,
                               (float*)grad_input.d_ptr_,
                               (float*)params.grad_weights.d_ptr_, // grad_gamma
                               (float*)params.grad_bias.d_ptr_,   // grad_beta
                               (const float*)grad_output.d_ptr_,
                               (const float*)cache.input->d_ptr_,
                               (const float*)params.weights.d_ptr_, // gamma
                               (const float*)cache.mean.d_ptr_,
                               (const float*)cache.rstd.d_ptr_,
                               static_cast<int>(B_rows), C_cols);
}

// --- Gelu Implementation ---
void Gelu::forward(hipStream_t stream, const GpuTensor& input, GpuTensor& output, GeluCache& cache) {
    if (!input.is_allocated()) {
        throw std::runtime_error("Input tensor not allocated for Gelu::forward.");
    }
    if (!output.is_allocated() || output.dims_ != input.dims_) {
        output.allocate(input.dims_);
    }
    cache.input = &input;

    // Note: The original code's DenseLayer fused GELU.
    // If GELU is used standalone, it needs its own kernel call.
    // The add_bias_gelu_kernel also performs bias. If input is already biased, just need GELU part.
    // For a pure GELU forward, we'd need a kernel like `gelu_forward_kernel(output, input, num_elements)`
    // The provided `add_bias_gelu_kernel` can be used if we pass a zero bias tensor.
    // Or, a more direct approach: if this Gelu class is used, it should call a dedicated GELU kernel.
    // For now, let's assume the call to `launch_add_bias_gelu_kernel` with a zero bias if no bias is intended.
    // However, the `launch_add_bias_gelu_kernel` expects M, N.
    // A simpler `gelu_kernel(output, input, num_elements)` would be better.
    // The file has `gelu_backward_kernel_impl` but no standalone `gelu_forward_kernel_impl`.
    // It has `add_bias_gelu_kernel_impl`.
    // This Gelu class's forward is tricky without a dedicated kernel.
    // Let's assume the user's intent was that GELU is usually fused,
    // and if standalone, it might be part of a custom sequence.
    // For this implementation, copying input to output if no dedicated kernel.
    // This is NOT a GELU computation. A proper GELU kernel is needed.
    // *** This is a placeholder - a real GELU kernel call is needed here ***
    // For example, if we had `launch_gelu_forward_kernel(stream, (float*)output.d_ptr_, (const float*)input.d_ptr_, input.num_elements_);`
    // Since it's missing, and DenseLayer handles its own GELU, this standalone GELU's forward is problematic.
    // For now, let's make it an error or a pass-through, as no pure GELU forward kernel was in the original code dump.
    // throw std::runtime_error("Standalone Gelu::forward requires a dedicated gelu_forward_kernel, which is not available in the provided kernels. It's usually fused.");
    // Update: The `add_bias_gelu_kernel_impl` can do this if bias is nullptr or zero.
    // But it requires M, N. If input is 1D, M=1, N=num_elements.
    // If input is e.g. (B,S,H), then M=B*S, N=H.
    if (input.dims_.empty()) throw std::runtime_error("Gelu input has no dimensions.");
    if (input.num_elements_ == 0) return;

    int M = 1;
    for(size_t i=0; i < input.dims_.size() -1; ++i) M*= input.dim_size(i);
    int N = input.dims_.back();

    GpuTensor zero_bias_dummy; // Create a dummy zero bias if needed
    // This is a workaround. A direct GELU kernel is better.
    // We are calling add_bias_gelu with a conceptual zero bias.
    // The kernel `add_bias_gelu_kernel_impl` takes `const float* bias`.
    // If we don't have a bias, we can't just pass nullptr if the kernel dereferences it.
    // The most robust way is to have a dedicated `gelu_kernel`.
    // Given the constraints, let's assume the input to this Gelu is the raw values *before* any activation.
    // And this function applies GELU.
    // The kernel `add_bias_gelu_kernel_impl` applies `input[idx] + bias[col]` then GELU.
    // If we want just GELU(input), we need a kernel that does `gelu(input[idx])`.
    // The available `gelu_fn_device` can be wrapped in a simple kernel.
    // For now, the `launch_add_bias_gelu_kernel` is the closest.
    // We will *not* add a bias here. This implies the `input` is what GELU should be applied to.
    // This means the `add_bias_gelu_kernel` is being misused or needs a mode.
    // A simple solution: use `gelu_backward_kernel`'s structure for a forward pass.
    // Create a temporary simple gelu_forward_kernel for this.
    // For the scope of this file, I'll assume the `DenseLayer` handles its GELU,
    // and this standalone Gelu is for other uses, requiring its own kernel not fully provided.
    // Let's use the structure of gelu_backward_kernel to make a simple forward one for now.
    // This is an ad-hoc solution based on available device functions.
    // It would be better to define this kernel in hip_kernels.cpp.
    // For now, let's assume `output.copy_from_gpu(input, stream);` and then a separate GELU call.
    // The current `DenseLayer::forward` *includes* GELU. So this standalone `Gelu::forward`
    // might be for a different pattern.
    // Let's assume it's an error for now, as it's not clearly defined how it should operate with available kernels.
     throw std::runtime_error("Standalone Gelu::forward is not fully implemented with a dedicated kernel in this context. It's typically fused in DenseLayer.");
}

void Gelu::backward(hipStream_t stream, GpuTensor& grad_input, const GpuTensor& grad_output, const GeluCache& cache) {
    if (!grad_output.is_allocated() || !cache.input || !cache.input->is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for Gelu::backward.");
    }
    if (!grad_input.is_allocated() || grad_input.dims_ != grad_output.dims_) {
        grad_input.allocate(grad_output.dims_);
    }

    launch_gelu_backward_kernel(stream,
                               (float*)grad_input.d_ptr_,
                               (const float*)grad_output.d_ptr_,
                               (const float*)cache.input->d_ptr_,
                               grad_input.num_elements_);
}
