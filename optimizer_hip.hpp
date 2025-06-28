#ifndef OPTIMIZER_HIP_HPP
#define OPTIMIZER_HIP_HPP

#include "common_hip.hpp" // For Parameter, GpuTensor
#include <vector>
#include <string> // Not strictly needed here but often useful

// --- AdamWOptimizer Class Definition ---
class AdamWOptimizer {
private:
    std::vector<Parameter*>& params_ref_; // Store references to parameters it will update
    float lr_;
    float beta1_;
    float beta2_;
    float epsilon_;
    float weight_decay_;
    int t_; // Timestep counter for bias correction

    // Optimizer states (momentum and variance estimates)
    // These will be vectors of GpuTensors, corresponding to each learnable parameter tensor (weights and biases)
    std::vector<GpuTensor> m_states_; // First moment vectors (for weights and biases)
    std::vector<GpuTensor> v_states_; // Second moment vectors (for weights and biases)

    // Helper to initialize m_states and v_states
    void initialize_optimizer_states();

public:
    AdamWOptimizer(std::vector<Parameter*>& params, // Pass by reference
                   float learning_rate = 1e-3f,    // Common default
                   float beta1 = 0.9f,
                   float beta2 = 0.999f,
                   float epsilon = 1e-8f,          // Adjusted from 1e-6 to common Adam default
                   float weight_decay = 0.01f);

    void zero_grad(hipStream_t stream = 0); // Zero out gradients of all managed parameters
    void step(hipStream_t stream = 0);      // Perform one optimization step
};

#endif // OPTIMIZER_HIP_HPP
