#ifndef OPTIMIZER_HIP_HPP
#define OPTIMIZER_HIP_HPP

#include "common_hip.hpp"
#include <vector>
#include <map>

// --- Shampoo Optimizer 클래스 ---
class ShampooOptimizer {
private:
    std::vector<Parameter*>& params_;
    float lr_, beta2_, epsilon_;
    int update_freq_, t_ = 0;
    
    // 상태 저장을 위해 파라미터 텐서 포인터를 키로 사용하는 map
    std::map<const GpuTensor*, GpuTensor> preconditioners_L_;
    std::map<const GpuTensor*, GpuTensor> preconditioners_R_;
    std::map<const GpuTensor*, GpuTensor> stats_GGT_;
    std::map<const GpuTensor*, GpuTensor> stats_GTG_;

    void compute_matrix_inverse_root(hipStream_t stream, rocblas_handle handle, GpuTensor& out, const GpuTensor& in);

public:
    ShampooOptimizer(std::vector<Parameter*>& params, float lr = 1e-3, int update_freq = 20, float beta2=0.999f, float epsilon=1e-8f);
    void step(hipStream_t stream, rocblas_handle handle);
};

// --- AdamW Optimizer 클래스 ---
class AdamWOptimizer {
private:
    std::vector<Parameter*>& params_;
    float lr_, beta1_, beta2_, epsilon_, weight_decay_;
    int t_ = 0;
    
    // 상태 저장을 위해 파라미터 텐서 포인터를 키로 사용하는 map
    std::map<const GpuTensor*, GpuTensor> m_states_;
    std::map<const GpuTensor*, GpuTensor> v_states_;

public:
    AdamWOptimizer(std::vector<Parameter*>& params, float lr = 1e-3, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f, float weight_decay = 1e-2f);
    void step(hipStream_t stream);
    void zero_grad(hipStream_t stream = 0); // zero_grad 추가
};

#endif // OPTIMIZER_HIP_HPP
