#include "optimizer_hip.hpp"
#include "common_hip.hpp"
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
// *** 추가됨: RXTX 알고리즘에 필요한 커널 ***
__global__ void transpose_copy_block_kernel(float* Dst, const float* Src, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) Dst[col * n + row] = Src[row * n + col];
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
    HIP_CHECK(hipGetLastError());
}
void launch_matrix_add_inplace(hipStream_t stream, GpuTensor& C, float alpha, const GpuTensor& A) {
    size_t n = C.num_elements_;
    hipLaunchKernelGGL(matrix_add_inplace_kernel, dim3((n + 255) / 256), dim3(256), 0, stream, (float*)C.d_ptr_, (const float*)A.d_ptr_, alpha, n);
    HIP_CHECK(hipGetLastError());
}
void launch_set_identity(hipStream_t stream, GpuTensor& t) {
    int n = t.dim_size(0);
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    hipLaunchKernelGGL(set_identity_kernel, blocks, threads, 0, stream, (float*)t.d_ptr_, n);
    HIP_CHECK(hipGetLastError());
}
// *** 추가됨: RXTX 알고리즘에 필요한 래퍼 함수 ***
void launch_transpose_copy_block(hipStream_t stream, GpuTensor& Dst, const GpuTensor& Src) {
    int n = Src.dim_size(0);
    dim3 threads(16, 16);
    dim3 blocks((n + 15) / 16, (n + 15) / 16);
    hipLaunchKernelGGL(transpose_copy_block_kernel, blocks, threads, 0, stream, (float*)Dst.d_ptr_, (const float*)Src.d_ptr_, n);
    HIP_CHECK(hipGetLastError());
}

// *** 추가됨: RXTX 알고리즘에 필요한 헬퍼 함수 ***
static GpuTensor make_block_view(const GpuTensor& X, int r_idx, int c_idx, int r_bl, int c_bl, int ld_full) {
    GpuTensor view;
    // GpuTensor의 d_ptr_은 void* 이므로 float*로 캐스팅하여 주소 계산
    float* ptr = (float*)X.d_ptr_ + (size_t(r_idx) * r_bl * ld_full) + (size_t(c_idx) * c_bl);
    view.set_view(ptr, {r_bl, c_bl});
    return view;
}

// *** 추가됨: RXTX 알고리즘 구현 ***
void launch_rxtx_multiplication(hipStream_t stream, rocblas_handle handle, GpuTensor& C, const GpuTensor& X);

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
        
        const float alpha = 1.0f;
        const float beta = 0.0f; // For ssyrk when calculating fresh ggt/gtg

        // 임시 GpuTensor에 G*G^T와 G^T*G 계산
        GpuTensor ggt; ggt.allocate({n, n});
        if (n > 0 && m > 0) {
            // *** RXTX 알고리즘 호출 복원 ***
            if (n % 4 == 0 && m % 4 == 0 && n >= 4 && m >= 4) {
                launch_rxtx_multiplication(stream, handle, ggt, grad);
            } else {
                ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_none, 
                                            n, m, &alpha, (const float*)grad.d_ptr_, m, 
                                            &beta, (float*)ggt.d_ptr_, n));
            }
        }

        GpuTensor gtg; gtg.allocate({m, m});
        if (n > 0 && m > 0) {
            ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_transpose, 
                                        m, n, &alpha, (const float*)grad.d_ptr_, m, 
                                        &beta, (float*)gtg.d_ptr_, m));
        }
        
        // 이동 평균 통계량 업데이트
        launch_matrix_set(stream, stats_GGT_.at(w_ptr), 1.0f - beta2_, ggt, beta2_, stats_GGT_.at(w_ptr));
        launch_matrix_set(stream, stats_GTG_.at(w_ptr), 1.0f - beta2_, gtg, beta2_, stats_GTG_.at(w_ptr));
        
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

// *** 추가됨: RXTX 알고리즘의 전체 구현 ***
void launch_rxtx_multiplication(hipStream_t stream, rocblas_handle handle, GpuTensor& C, const GpuTensor& X) {
    const int n_full = X.dim_size(0);
    const int m_full = X.dim_size(1);
    if (n_full % 4 != 0 || m_full % 4 != 0) {
        throw std::runtime_error("Matrix dimensions must be divisible by 4 for this RXTX implementation.");
    }
    const int n_sub  = n_full / 4;
    const int m_sub  = m_full / 4;

    std::vector<GpuTensor> Xb;
    Xb.reserve(16);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Xb.push_back(make_block_view(X, i, j, n_sub, m_sub, m_full));
        }
    }
    
    std::vector<GpuTensor> m(26), s(8), y(2), w(11), z(8), L(26), R(26);
    for (auto &t : m) t.allocate({ n_sub, n_sub });
    for (auto &t : s) t.allocate({ n_sub, n_sub });
    for (auto &t : y) t.allocate({ n_sub, m_sub });
    for (auto &t : w) t.allocate({ n_sub, m_sub });
    for (auto &t : z) t.allocate({ n_sub, n_sub });
    for (auto &t : L) t.allocate({ n_sub, m_sub });
    for (auto &t : R) t.allocate({ n_sub, m_sub });

    launch_matrix_set(stream, y[0],  1.0f, Xb[12], -1.0f, Xb[13]);
    launch_matrix_set(stream, y[1],  1.0f, Xb[11], -1.0f, Xb[9]);
    launch_matrix_set(stream, w[0],  1.0f, Xb[1], 1.0f, Xb[3]); launch_matrix_add_inplace(stream, w[0], -1.0f, Xb[7]);
    launch_matrix_set(stream, w[1],  1.0f, Xb[0], -1.0f, Xb[4]); launch_matrix_add_inplace(stream, w[1], -1.0f, Xb[5]);
    launch_matrix_set(stream, w[2],  1.0f, Xb[5], 1.0f, Xb[6]);
    launch_matrix_set(stream, w[3],  1.0f, Xb[13], 1.0f, Xb[14]);
    launch_matrix_set(stream, w[4],  1.0f, y[1], 1.0f, Xb[15]);
    launch_matrix_set(stream, w[5],  1.0f, Xb[9], 1.0f, Xb[10]);
    launch_matrix_set(stream, w[6],  1.0f, Xb[8], 1.0f, y[0]);
    launch_matrix_set(stream, w[7],  1.0f, Xb[8], -1.0f, Xb[7]);
    launch_matrix_set(stream, w[8],  1.0f, Xb[6], -1.0f, Xb[10]);
    launch_matrix_set(stream, w[9],  1.0f, Xb[5], -1.0f, Xb[6]);
    launch_matrix_set(stream, w[10], 1.0f, Xb[1], -1.0f, Xb[2]);

    launch_matrix_set(stream, L[0], -1.0f, w[0],  1.0f, Xb[2]);   launch_matrix_set(stream, R[0],  1.0f, Xb[7],   1.0f, Xb[10]);
    launch_matrix_set(stream, L[1],  1.0f, w[1],  1.0f, Xb[6]);   launch_matrix_set(stream, R[1],  1.0f, Xb[14],  1.0f, Xb[4]);
    launch_matrix_set(stream, L[2], -1.0f, Xb[1], 1.0f, Xb[11]);  launch_matrix_set(stream, R[2],  1.0f, w[4],    0.0f, R[2]);
    launch_matrix_set(stream, L[3],  1.0f, Xb[8],-1.0f, Xb[5]);   launch_matrix_set(stream, R[3],  1.0f, w[6],    0.0f, R[3]);
    launch_matrix_set(stream, L[4],  1.0f, Xb[1], 1.0f, Xb[10]);  launch_matrix_set(stream, R[4],  1.0f, Xb[14], -1.0f, w[2]);
    launch_matrix_set(stream, L[5],  1.0f, Xb[5], 1.0f, Xb[10]);  launch_matrix_set(stream, R[5],  1.0f, w[2],   -1.0f, Xb[10]);
    HIP_CHECK(hipMemcpyAsync(L[6].d_ptr_, Xb[10].d_ptr_, L[6].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[6],  1.0f, w[2],    0.0f, R[6]);
    HIP_CHECK(hipMemcpyAsync(L[7].d_ptr_, Xb[1].d_ptr(),  L[7].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[7],  1.0f, w[2],   -1.0f, w[3]); launch_matrix_add_inplace(stream, R[7], 1.0f, w[4]);
    HIP_CHECK(hipMemcpyAsync(L[8].d_ptr_, Xb[5].d_ptr(),  L[8].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[8],  1.0f, w[6],   -1.0f, w[5]); launch_matrix_add_inplace(stream, R[8], 1.0f, w[2]);
    launch_matrix_set(stream, L[9],  1.0f, w[0], -1.0f, Xb[2]); launch_matrix_add_inplace(stream, L[9], 1.0f, Xb[6]); launch_matrix_add_inplace(stream, L[9],1.0f, Xb[10]);
    HIP_CHECK(hipMemcpyAsync(R[9].d_ptr_, Xb[10].d_ptr_, R[9].size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[10], 1.0f, Xb[4],-1.0f, w[9]);   HIP_CHECK(hipMemcpyAsync(R[10].d_ptr(), Xb[4].d_ptr(), R[10].size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[11], 1.0f, w[10],1.0f, Xb[3]);   HIP_CHECK(hipMemcpyAsync(R[11].d_ptr(), Xb[7].d_ptr(), R[11].size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[12],-1.0f, w[1], 1.0f, Xb[2]);   HIP_CHECK(hipMemcpyAsync(R[12].d_ptr(), Xb[14].d_ptr(),R[12].size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[13],-1.0f, w[1], 0.0f, L[13]);  launch_matrix_set(stream, R[13], 1.0f, w[6],    1.0f, w[3]);
    launch_matrix_set(stream, L[14], 1.0f, w[0], 0.0f, L[14]);   launch_matrix_set(stream, R[14], 1.0f, w[5],    1.0f, w[4]);
    launch_matrix_set(stream, L[15], 1.0f, Xb[0],-1.0f, Xb[7]);   launch_matrix_set(stream, R[15], 1.0f, Xb[8],  -1.0f, Xb[15]);
    HIP_CHECK(hipMemcpyAsync(L[16].d_ptr_, Xb[11].d_ptr(),L[16].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[16],-1.0f, y[1],    0.0f, R[16]);
    HIP_CHECK(hipMemcpyAsync(L[17].d_ptr_, Xb[8].d_ptr(), L[17].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[17], 1.0f, y[0],    0.0f, R[17]);
    launch_matrix_set(stream, L[18],-1.0f, w[10],0.0f, L[18]);  launch_matrix_set(stream, R[18],-1.0f, Xb[14],  1.0f, Xb[6]);
    launch_matrix_set(stream, L[19], 1.0f, Xb[4], 1.0f, w[7]);   HIP_CHECK(hipMemcpyAsync(R[19].d_ptr(), Xb[8].d_ptr(), R[19].size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipMemcpyAsync(L[20].d_ptr(), Xb[7].d_ptr(), L[20].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[20], 1.0f, Xb[11],  1.0f, w[7]);
    launch_matrix_set(stream, L[21],-1.0f, w[9], 0.0f, L[21]);  launch_matrix_set(stream, R[21], 1.0f, Xb[4],   1.0f, w[8]);
    HIP_CHECK(hipMemcpyAsync(L[22].d_ptr_, Xb[0].d_ptr(), L[22].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[22], 1.0f, Xb[12], -1.0f, Xb[4]);
    HIP_CHECK(hipMemcpyAsync(L[23].d_ptr_, Xb[0].d_ptr(), L[23].size_in_bytes(), hipMemcpyDeviceToDevice, stream)); launch_matrix_set(stream, R[23], 1.0f, Xb[0],   1.0f, Xb[12]);
    launch_matrix_set(stream, L[24], 1.0f, Xb[8], 1.0f, Xb[1]);   HIP_CHECK(hipMemcpyAsync(R[24].d_ptr_, Xb[13].d_ptr(),R[24].size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    launch_matrix_set(stream, L[25], 1.0f, Xb[5], 1.0f, Xb[9]);   HIP_CHECK(hipMemcpyAsync(R[25].d_ptr(), Xb[9].d_ptr(), R[25].size_in_bytes(), hipMemcpyDeviceToDevice, stream));
    
    const float alpha = 1.0f, beta = 0.0f;
    rocblas_set_stream(handle, stream);

    for (int i = 0; i < 26; ++i) {
        ROCBLAS_CHECK(rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_transpose,
                      n_sub, n_sub, m_sub, &alpha, (float*)L[i].d_ptr_, m_sub, (const float*)R[i].d_ptr_, m_sub, &beta, (float*)m[i].d_ptr_, n_sub));
    }
    
    const int s_indices[] = {0, 1, 2, 3, 12, 13, 14, 15};
    for (int i = 0; i < 8; ++i) {
        ROCBLAS_CHECK(rocblas_ssyrk(handle, rocblas_fill_upper, rocblas_operation_none,
                      n_sub, m_sub, &alpha, (const float*)Xb[s_indices[i]].d_ptr_, m_full, &beta, (float*)s[i].d_ptr_, n_sub));
    }

    auto Cblock = [&](int r, int c) { return make_block_view(C, r, c, n_sub, n_sub, n_full); };
    launch_matrix_set(stream, z[0],  1.0f, m[6],   -1.0f, m[10]); launch_matrix_add_inplace(stream, z[0], -1.0f, m[11]);
    launch_matrix_set(stream, z[1],  1.0f, m[0],    1.0f, m[11]); launch_matrix_add_inplace(stream, z[1],  1.0f, m[20]);
    launch_matrix_set(stream, z[2],  1.0f, m[2],    1.0f, m[16]); launch_matrix_add_inplace(stream, z[2], -1.0f, m[23]);
    launch_matrix_set(stream, z[3],  1.0f, m[1],    1.0f, m[10]); launch_matrix_add_inplace(stream, z[3],  1.0f, m[22]);
    launch_matrix_set(stream, z[4],  1.0f, m[4],    1.0f, m[6]);  launch_matrix_add_inplace(stream, z[4],  1.0f, m[7]);
    launch_matrix_set(stream, z[5],  1.0f, m[3],   -1.0f, m[17]); launch_matrix_add_inplace(stream, z[5], -1.0f, m[19]);
    launch_matrix_set(stream, z[6],  1.0f, m[5],   -1.0f, m[6]);  launch_matrix_add_inplace(stream, z[6], -1.0f, m[8]);
    launch_matrix_set(stream, z[7],  1.0f, m[16],   1.0f, m[17]);
    
    { auto t = Cblock(0,0); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, s[0], 1.0f, s[1]); launch_matrix_add_inplace(stream, t, 1.0f, s[2]); launch_matrix_add_inplace(stream, t, 1.0f, s[3]); }
    { auto t = Cblock(0,1); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[1], -1.0f, m[4]); launch_matrix_add_inplace(stream, t, -1.0f, z[0]); launch_matrix_add_inplace(stream, t, 1.0f, m[12]); launch_matrix_add_inplace(stream, t, 1.0f, m[18]); }
    { auto t = Cblock(0,2); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[1], 1.0f, z[2]); launch_matrix_add_inplace(stream, t, 1.0f, m[14]); launch_matrix_add_inplace(stream, t, 1.0f, m[15]); }
    { auto t = Cblock(0,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[3], -1.0f, z[2]); launch_matrix_add_inplace(stream, t, -1.0f, z[4]); launch_matrix_add_inplace(stream, t, 1.0f, m[12]); }
    { auto t = Cblock(1,1); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[0], 1.0f, m[5]); launch_matrix_add_inplace(stream, t, -1.0f, z[0]); launch_matrix_add_inplace(stream, t, 1.0f, m[9]); launch_matrix_add_inplace(stream, t, 1.0f, m[21]); }
    { auto t = Cblock(1,2); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[1], -1.0f, z[5]); launch_matrix_add_inplace(stream, t, 1.0f, z[6]); launch_matrix_add_inplace(stream, t, 1.0f, m[9]); }
    { auto t = Cblock(1,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, z[3], 1.0f, z[5]); launch_matrix_add_inplace(stream, t, 1.0f, m[13]); launch_matrix_add_inplace(stream, t, 1.0f, m[15]); }
    { auto t = Cblock(2,2); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[3], -1.0f, z[6]); launch_matrix_add_inplace(stream, t, -1.0f, z[7]); launch_matrix_add_inplace(stream, t, 1.0f, m[25]); }
    { auto t = Cblock(2,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, m[2], 1.0f, z[4]); launch_matrix_add_inplace(stream, t, 1.0f, z[7]); launch_matrix_add_inplace(stream, t, 1.0f, m[24]); }
    { auto t = Cblock(3,3); t.zero_out(stream); launch_matrix_set(stream, t, 1.0f, s[4], 1.0f, s[5]); launch_matrix_add_inplace(stream, t, 1.0f, s[6]); launch_matrix_add_inplace(stream, t, 1.0f, s[7]); }
    
    for (int i = 1; i < 4; ++i) {
        for (int j = 0; j < i; ++j) {
            auto C_ij = Cblock(i, j);
            auto C_ji = Cblock(j, i);
            launch_transpose_copy_block(stream, C_ij, C_ji);
        }
    }
}

