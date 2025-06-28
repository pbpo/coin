#include "nn_layers_hip.hpp"
#include "hip_kernels.hpp" // For kernel launchers
#include "common_hip.hpp"  // For GpuTensor, Parameter, ROCBLAS_CHECK
#include <stdexcept>       // For runtime_error
#include <vector>
#include <string>
#include <cstdlib> // For std::rand for dropout seed (now replaced)
#include <random>  // For std::random_device, std::mt19937_64

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

        // Use std::random_device and std::mt19937 for better seed generation
        static std::random_device rd; // Static to be initialized once
        static std::mt19937_64 gen(rd()); // Static to be seeded once
        unsigned long long seed = gen();

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


// *** 수정됨: 비효율적인 forward 로직 개선 ***
void DenseLayer::forward(rocblas_handle blas_handle, hipStream_t stream,
                         const GpuTensor& input, GpuTensor& output,
                         DenseLayerCache& cache) {
    if (!input.is_allocated() || !params.weights.is_allocated()) {
        throw std::runtime_error("Input or weights tensor not allocated for DenseLayer::forward for layer " + name_);
    }

    int in_features = params.weights.dim_size(1);
    int out_features = params.weights.dim_size(0);

    if (input.dims_.back() != in_features) {
        throw std::runtime_error("Input feature size mismatch for DenseLayer " + name_);
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

    // 역전파를 위해 입력 저장
    cache.input_to_gemm = &input;
    // 역전파를 위해 GELU의 입력(GEMM 결과 + Bias) 저장
    cache.output_before_gelu.allocate(output.dims_);

    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta = 0.0f;

    // 연산: output = input * weights^T
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,
                                rocblas_operation_transpose,
                                M, N, K,
                                &alpha,
                                (const float*)input.d_ptr_, K,
                                (const float*)params.weights.d_ptr_, K,
                                &beta,
                                (float*)cache.output_before_gelu.d_ptr_, N)); // 결과를 바로 cache.output_before_gelu에 저장

    if (params.has_bias_) {
        // 융합 커널을 사용하여 Bias 덧셈과 GELU를 한 번에 처리
        launch_add_bias_gelu_kernel(stream,
                                   (float*)output.d_ptr_, // 최종 출력
                                   (const float*)cache.output_before_gelu.d_ptr_, // GEMM 결과
                                   (const float*)params.bias.d_ptr_,
                                   M, N);
        // Bias가 더해진 결과를 다시 cache에 저장 (GELU 역전파에 필요)
        launch_add_bias_kernel(stream,
                               (float*)cache.output_before_gelu.d_ptr(),
                               (const float*)cache.output_before_gelu.d_ptr(),
                               (const float*)params.bias.d_ptr_,
                               M, N);

    } else {
        // Bias가 없으면 GELU만 적용
        launch_gelu_forward_kernel(stream,
                                  (float*)output.d_ptr(),
                                  (const float*)cache.output_before_gelu.d_ptr(), // GEMM 결과가 GELU 입력
                                  output.num_elements_);
    }
}

void DenseLayer::backward(rocblas_handle blas_handle, hipStream_t stream,
                          const GpuTensor& grad_output_after_gelu, const DenseLayerCache& cache,
                          GpuTensor& grad_input_gemm) { // Renamed grad_input to grad_input_gemm for clarity
    // grad_output_after_gelu shape: (batch_dims..., out_features)
    // input_to_gemm shape (from cache): (batch_dims..., in_features)
    // weights shape: (out_features, in_features)
    // output_before_gelu shape (from cache): (batch_dims..., out_features)

    if (!grad_output_after_gelu.is_allocated() ||
        !cache.input_to_gemm || !cache.input_to_gemm->is_allocated() ||
        !cache.output_before_gelu.is_allocated() ||
        !params.weights.is_allocated()) {
        throw std::runtime_error("Required tensors not allocated for DenseLayer::backward for layer " + name_);
    }

    int out_features = params.weights.dim_size(0);
    int in_features = params.weights.dim_size(1);

    if (grad_output_after_gelu.dims_.back() != out_features) {
        throw std::runtime_error("grad_output feature size mismatch for DenseLayer " + name_);
    }
    if (!grad_input_gemm.is_allocated() || !cache.input_to_gemm || grad_input_gemm.dims_ != cache.input_to_gemm->dims_) {
        grad_input_gemm.allocate(cache.input_to_gemm->dims_);
    }
    if (!params.grad_weights.is_allocated() || params.grad_weights.dims_ != params.weights.dims_) {
        params.grad_weights.allocate(params.weights.dims_);
    }
    if (params.has_bias_ && (!params.grad_bias.is_allocated() || params.grad_bias.dims_ != params.bias.dims_)) {
        params.grad_bias.allocate(params.bias.dims_);
    }


    size_t batch_size_combined = 1;
    for (size_t i = 0; i < grad_output_after_gelu.dims_.size() - 1; ++i) {
        batch_size_combined *= grad_output_after_gelu.dim_size(i);
    }

    int M_grad_out = static_cast<int>(batch_size_combined); // M for grad_output (batch_combined)
    int N_grad_out = out_features;                         // N for grad_output (out_features)
    int K_input_feat = in_features;                        // K for input (in_features)

    // The input grad_output_after_gelu is dL/d(GELU_Output).
    // We need dL/d(LinearOutput) to proceed with GEMM backward.
    // LinearOutput = GEMM_Result + Bias (this is stored in cache.output_before_gelu)

    GpuTensor grad_linear_output; // Gradient w.r.t. (GEMM_Result + Bias)
    grad_linear_output.allocate(grad_output_after_gelu.dims_);

    // Step 1: Backward through GELU and Bias addition.
    // This kernel calculates grad_linear_output (dL/d(InputToGelu)) and params.grad_bias.
    // Note: params.grad_bias must be zeroed out by optimizer before each accumulation step if accumulating over mini-batches.
    // The kernel launch_gelu_add_bias_backward_kernel uses atomicAdd for grad_bias, so it accumulates within a call.
    if (params.has_bias_) {
        launch_gelu_add_bias_backward_kernel(stream,
                                       (float*)grad_linear_output.d_ptr_,
                                       (float*)params.grad_bias.d_ptr_,
                                       (const float*)grad_output_after_gelu.d_ptr(),
                                       (const float*)cache.output_before_gelu.d_ptr(), // This was (GEMM_out + Bias)
                                       M_grad_out, N_grad_out);
    } else {
        // If no bias, GELU was applied directly to GEMM output.
        // We need a GELU backward kernel here. `launch_gelu_backward_kernel` is available.
        // grad_bias calculation is skipped.
        launch_gelu_backward_kernel(stream,
                                    (float*)grad_linear_output.d_ptr(),
                                    (const float*)grad_output_after_gelu.d_ptr(),
                                    (const float*)cache.output_before_gelu.d_ptr(), // This was GEMM_out
                                    grad_linear_output.num_elements_);
        // params.grad_bias remains zero or untouched if not allocated.
    }


    ROCBLAS_CHECK(rocblas_set_stream(blas_handle, stream));
    const float alpha = 1.0f, beta_one = 1.0f; // Use beta_one for accumulating gradients

    // Step 2: Calculate grad_input_gemm = grad_linear_output * weights
    // grad_input_gemm (M_grad_out, K_input_feat) = grad_linear_output (M_grad_out, N_grad_out) * weights (N_grad_out, K_input_feat)
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_none,       // opA
                                rocblas_operation_none,       // opB (weights are N_out x K_in)
                                M_grad_out, K_input_feat, N_grad_out,
                                &alpha,
                                (const float*)grad_linear_output.d_ptr_, N_grad_out,      // A, lda
                                (const float*)params.weights.d_ptr_, K_input_feat, // B, ldb
                                &beta_one, // Accumulate to existing grad_input_gemm (e.g. from other paths in a complex model)
                                           // For a simple Dense layer, if grad_input_gemm is not pre-zeroed, this should be beta_zero.
                                           // Assuming grad_input_gemm is zeroed by caller or this is the only contribution.
                                           // Let's assume beta_one for accumulation, implying grad_input_gemm might be used by multiple paths.
                                (float*)grad_input_gemm.d_ptr_, K_input_feat));     // C, ldc


    // Step 3: Calculate grad_weights = input_to_gemm^T * grad_linear_output (누적)
    // grad_weights (K_input_feat, N_grad_out) = input_to_gemm^T (K_input_feat, M_grad_out) * grad_linear_output (M_grad_out, N_grad_out)
    ROCBLAS_CHECK(rocblas_sgemm(blas_handle,
                                rocblas_operation_transpose,  // opA (input_to_gemm is M,K -> use as K,M)
                                rocblas_operation_none,       // opB
                                K_input_feat, N_grad_out, M_grad_out,
                                &alpha,
                                (const float*)cache.input_to_gemm->d_ptr_, K_input_feat, // A (original M,K), lda
                                (const float*)grad_linear_output.d_ptr_, N_grad_out,   // B (M,N), ldb
                                &beta_one, // Accumulate gradients
                                (float*)params.grad_weights.d_ptr_, N_grad_out)); // C (K,N), ldc

    // grad_bias is already calculated by launch_gelu_add_bias_backward_kernel if bias was present.
    // If there was no bias, params.grad_bias should not be touched or should be zero.
    // The original code's grad_bias calculation was:
    // launch_reduce_sum_axis0_add_kernel(stream, (float*)params.grad_bias.d_ptr(), (const float*)grad_output.d_ptr(), M_grad_out, N_grad_out);
    // This is now handled by the fused backward kernel for add_bias_gelu.
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
    launch_gelu_forward_kernel(
        stream,
        (float*)output.d_ptr_,
        (const float*)input.d_ptr_,
        input.num_elements_);
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
