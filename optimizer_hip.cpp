#include "optimizer_hip.hpp" // 헤더 파일 이름 수정
#include "common_hip.hpp"     // 공통 헤더 포함
#include <rocsolver/rocsolver.h>
#include <numeric>
#include <cmath>

// rocSOLVER 오류 체크 매크로
#define ROCSOLVER_CHECK(cmd) do { \
    rocblas_status err = cmd; \
    if (err != rocblas_status_success) { \
        fprintf(stderr, "rocSOLVER Error: status %d in %s at line %d\n", err, __FILE__, __LINE__); \
        throw std::runtime_error("rocSOLVER Error"); \
    } \
} while(0)

// --- 커스텀 커널 ---
__global__ void matrix_set_kernel(float* C, const float* A, const float* B, float alpha, float beta, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] = alpha * A[idx] + beta * B[idx];
}
__global__ void matrix_add_inplace_kernel(float* C, const float* A, float alpha, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) C[idx] += alpha * A[idx];
}
__global__ void set_identity_kernel(float* matrix, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) matrix[row * n + col] = (row == col) ? 1.0f : 0.0f;
}
__global__ void add_diagonal_kernel(float* matrix, int n, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) matrix[i * n + i] += value;
}
__global__ void elementwise_power_kernel(float* d, float exponent, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) d[idx] = powf(d[idx], exponent);
}
__global__ void scale_columns_kernel(float* out, const float* V, const float* d, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) out[row * n + col] = V[row * n + col] * d[col];
}
__global__ void adamw_kernel(float* weights, const float* grad, float* m, float* v,
                             float lr, float beta1, float beta2, float epsilon, float weight_decay,
                             float beta1_t, float beta2_t, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (weight_decay > 0.0f) {
            weights[idx] -= lr * weight_decay * weights[idx];
        }
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad[idx];
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad[idx] * grad[idx];
        float m_hat = m[idx] / (1.0f - beta1_t);
        float v_hat = v[idx] / (1.0f - beta2_t);
        weights[idx] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// --- 커널 래퍼 함수 ---
void launch_matrix_set(hipStream_t stream, GpuTensor& C, float alpha, const GpuTensor& A, float beta, const GpuTensor& B) {
    size_t n = C.num_elements_;
    hipLaunchKernelGGL(matrix_set_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, (float*)C.d_ptr_, (const float*)A.d_ptr_, (const float*)B.d_ptr_, alpha, beta, n);
}
void launch_matrix_add_inplace(hipStream_t stream, GpuTensor& C, float alpha, const GpuTensor& A) {
    size_t n = C.num_elements_;
    hipLaunchKernelGGL(matrix_add_inplace_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, (float*)C.d_ptr_, (const float*)A.d_ptr_, alpha, n);
}
void launch_set_identity(hipStream_t stream, GpuTensor& t) {
    int n = t.dim_size(0);
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    hipLaunchKernelGGL(set_identity_kernel, blocks, threads, 0, stream, (float*)t.d_ptr_, n);
}

// --- ShampooOptimizer 구현 ---
ShampooOptimizer::ShampooOptimizer(std::vector<Parameter*>& params, float lr, int update_freq, float beta2, float epsilon)
    : params_(params), lr_(lr), update_freq_(update_freq), beta2_(beta2), epsilon_(epsilon) {
    
    hipStream_t stream = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params_[i];
        if (!p || !p->weights.is_allocated()) continue;
        
        const GpuTensor* w_ptr = &p->weights;
        int n = w_ptr->dim_size(0);
        int m = w_ptr->dim_size(1);

        preconditioners_L_[w_ptr].allocate({n, n});
        preconditioners_R_[w_ptr].allocate({m, m});
        stats_GGT_[w_ptr].allocate({n, n});
        stats_GTG_[w_ptr].allocate({m, m});

        launch_set_identity(stream, preconditioners_L_.at(w_ptr));
        launch_set_identity(stream, preconditioners_R_.at(w_ptr));
        stats_GGT_.at(w_ptr).zero_out(stream);
        stats_GTG_.at(w_ptr).zero_out(stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

void ShampooOptimizer::compute_matrix_inverse_root(hipStream_t stream, rocblas_handle handle, GpuTensor& out, const GpuTensor& in) {
    int n = in.dim_size(0);
    if (n == 0) return;
    
    rocblas_set_stream(handle, stream);

    GpuTensor A_copy, d_eigenvalues, e_superdiag;
    A_copy.allocate({n, n});
    d_eigenvalues.allocate({n});
    e_superdiag.allocate({n > 1 ? n - 1 : 1});

    HIP_CHECK(hipMemcpyAsync(A_copy.d_ptr_, in.d_ptr_, in.size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    
    dim3 threads_diag(256);
    dim3 blocks_diag((n + 255) / 256);
    hipLaunchKernelGGL(add_diagonal_kernel, blocks_diag, threads_diag, 0, stream, (float*)A_copy.d_ptr_, n, epsilon_);

    rocblas_int devInfo;
    ROCSOLVER_CHECK(rocsolver_ssyevd(handle, rocsolver_evect_original, rocsolver_fill_upper, n, (float*)A_copy.d_ptr_, n, (float*)d_eigenvalues.d_ptr_, (float*)e_superdiag.d_ptr_, &devInfo));

    if (devInfo != 0) {
        launch_set_identity(stream, out);
        return;
    }

    hipLaunchKernelGGL(elementwise_power_kernel, blocks_diag, threads_diag, 0, stream, (float*)d_eigenvalues.d_ptr_, -0.25f, n);

    GpuTensor temp; temp.allocate({n, n});
    dim3 threads_2d(16, 16);
    dim3 blocks_2d((n + 15) / 16, (n + 15) / 16);
    hipLaunchKernelGGL(scale_columns_kernel, blocks_2d, threads_2d, 0, stream, (float*)temp.d_ptr_, (const float*)A_copy.d_ptr_, (const float*)d_eigenvalues.d_ptr_, n);

    const float alpha = 1.0f, beta = 0.0f;
    ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose, n, n, n, &alpha, (const float*)temp.d_ptr_, n, (const float*)A_copy.d_ptr_, n, &beta, (float*)out.d_ptr_, n));
}

void ShampooOptimizer::step(hipStream_t stream, rocblas_handle handle) {
    t_++;
    rocblas_set_stream(handle, stream);

    for (size_t i = 0; i < params_.size(); ++i) {
        Parameter* p = params_[i];
        if (!p || !p->weights.is_allocated() || !p->grad_weights.is_allocated()) continue;

        GpuTensor& grad = p->grad_weights;
        const GpuTensor* w_ptr = &p->weights;
        int n = grad.dim_size(0);
        int m = grad.dim_size(1);
        const float alpha = 1.0f, beta_one_minus_beta2 = 1.0f - beta2_;

        GpuTensor ggt; ggt.allocate({n, n});
        if (n > 0 && m > 0) ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_none, n, m, &alpha, (const float*)grad.d_ptr_, m, &beta2_, (float*)ggt.d_ptr_, n));

        GpuTensor gtg; gtg.allocate({m, m});
        if (n > 0 && m > 0) ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_transpose, m, n, &alpha, (const float*)grad.d_ptr_, m, &beta2_, (float*)gtg.d_ptr_, m));
        
        launch_matrix_set(stream, stats_GGT_.at(w_ptr), 1.0f, ggt, beta2_, stats_GGT_.at(w_ptr));
        launch_matrix_set(stream, stats_GTG_.at(w_ptr), 1.0f, gtg, beta2_, stats_GTG_.at(w_ptr));
        
        if (t_ % update_freq_ == 0) {
            compute_matrix_inverse_root(stream, handle, preconditioners_L_.at(w_ptr), stats_GGT_.at(w_ptr));
            compute_matrix_inverse_root(stream, handle, preconditioners_R_.at(w_ptr), stats_GTG_.at(w_ptr));
        }

        GpuTensor preconditioned_grad; preconditioned_grad.allocate(grad.dims_);
        GpuTensor temp_grad; temp_grad.allocate({n, m});
        
        const float gemm_alpha = 1.0f, gemm_beta = 0.0f;
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, n, m, n, &gemm_alpha, (const float*)preconditioners_L_.at(w_ptr).d_ptr_, n, (const float*)grad.d_ptr_, m, &gemm_beta, (float*)temp_grad.d_ptr_, n));
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none, n, m, m, &gemm_alpha, (const float*)temp_grad.d_ptr_, n, (const float*)preconditioners_R_.at(w_ptr).d_ptr_, m, &gemm_beta, (float*)preconditioned_grad.d_ptr_, n));

        launch_matrix_add_inplace(stream, p->weights, -lr_, preconditioned_grad);
    }
}


// --- AdamWOptimizer 구현 ---
AdamWOptimizer::AdamWOptimizer(std::vector<Parameter*>& params, float lr, float beta1, float beta2, float epsilon, float weight_decay)
    : params_(params), lr_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), weight_decay_(weight_decay) {

    hipStream_t stream = 0;
    for (size_t i = 0; i < params.size(); ++i) {
        Parameter* p = params_[i];
        if (!p) continue;
        if (p->weights.is_allocated()) {
            const GpuTensor* w_ptr = &p->weights;
            m_states_[w_ptr].allocate(w_ptr->dims_);
            v_states_[w_ptr].allocate(w_ptr->dims_);
            m_states_.at(w_ptr).zero_out(stream);
            v_states_.at(w_ptr).zero_out(stream);
        }
        if (p->has_bias_ && p->bias.is_allocated()) {
            const GpuTensor* b_ptr = &p->bias;
            m_states_[b_ptr].allocate(b_ptr->dims_);
            v_states_[b_ptr].allocate(b_ptr->dims_);
            m_states_.at(b_ptr).zero_out(stream);
            v_states_.at(b_ptr).zero_out(stream);
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));
}

void AdamWOptimizer::step(hipStream_t stream) {
    t_++;
    float beta1_t = powf(beta1_, t_);
    float beta2_t = powf(beta2_, t_);

    for (size_t i = 0; i < params_.size(); ++i) {
        Parameter* p = params_[i];
        if (!p) continue;
        
        if (p->weights.is_allocated() && p->grad_weights.is_allocated()) {
            const GpuTensor* w_ptr = &p->weights;
            size_t n = w_ptr->num_elements_;
            hipLaunchKernelGGL(adamw_kernel, dim3((n + 255) / 256), dim3(256), 0, stream,
                (float*)w_ptr->d_ptr_, (const float*)p->grad_weights.d_ptr_, (float*)m_states_.at(w_ptr).d_ptr_, (float*)v_states_.at(w_ptr).d_ptr_,
                lr_, beta1_, beta2_, epsilon_, weight_decay_, beta1_t, beta2_t, n);
        }
        
        if (p->has_bias_ && p->bias.is_allocated() && p->grad_bias.is_allocated()) {
             const GpuTensor* b_ptr = &p->bias;
             size_t n = b_ptr->num_elements_;
             hipLaunchKernelGGL(adamw_kernel, dim3((n + 255) / 256), dim3(256), 0, stream,
                (float*)b_ptr->d_ptr_, (const float*)p->grad_bias.d_ptr_, (float*)m_states_.at(b_ptr).d_ptr_, (float*)v_states_.at(b_ptr).d_ptr_,
                lr_, beta1_, beta2_, epsilon_, 0.0f, beta1_t, beta2_t, n);
        }
    }
}

void AdamWOptimizer::zero_grad(hipStream_t stream) {
    for (auto* p_param : params_) {
        if (p_param) {
            if (p_param->grad_weights.is_allocated()) {
                p_param->grad_weights.zero_out(stream);
            }
            if (p_param->has_bias_ && p_param->grad_bias.is_allocated()) {
                p_param->grad_bias.zero_out(stream);
            }
        }
    }
}
