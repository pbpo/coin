#include "optimizer_hip.hpp"
#include "hip_kernels.hpp" // For launch_adamw_update_kernel
#include "common_hip.hpp"  // For Parameter, GpuTensor, HIP_CHECK
#include <stdexcept>

// --- AdamWOptimizer Implementation ---

AdamWOptimizer::AdamWOptimizer(std::vector<Parameter*>& params,
                               float learning_rate, float beta1, float beta2,
                               float epsilon, float weight_decay)
    : params_ref_(params), lr_(learning_rate), beta1_(beta1), beta2_(beta2),
      epsilon_(epsilon), weight_decay_(weight_decay), t_(0) {
    if (params_ref_.empty()) {
        // It's valid to have an optimizer with no parameters, though it won't do anything.
        // Consider if this should be a warning or an error based on expected usage.
    }
    initialize_optimizer_states();
}

void AdamWOptimizer::initialize_optimizer_states() {
    m_states_.clear();
    v_states_.clear();
    // Each parameter object can have weights and (optionally) bias.
    // So, for each Parameter*, we might have up to two GpuTensors for m_states and v_states.
    // Let's size them to params_ref_.size() * 2 for worst case, then populate.
    // A more precise way would be to count actual tensors.

    for (const auto* p_param : params_ref_) {
        if (p_param) {
            if (p_param->weights.is_allocated()) {
                m_states_.emplace_back(GpuTensor(p_param->weights.dims_, p_param->name + "_w_m"));
                v_states_.emplace_back(GpuTensor(p_param->weights.dims_, p_param->name + "_w_v"));
            } else { // Push empty tensor to keep indexing consistent if a param sub-tensor isn't there
                m_states_.emplace_back(GpuTensor(p_param->name + "_w_m_unalloc"));
                v_states_.emplace_back(GpuTensor(p_param->name + "_w_v_unalloc"));
            }

            if (p_param->has_bias_ && p_param->bias.is_allocated()) {
                m_states_.emplace_back(GpuTensor(p_param->bias.dims_, p_param->name + "_b_m"));
                v_states_.emplace_back(GpuTensor(p_param->bias.dims_, p_param->name + "_b_v"));
            } else {
                m_states_.emplace_back(GpuTensor(p_param->name + "_b_m_unalloc"));
                v_states_.emplace_back(GpuTensor(p_param->name + "_b_v_unalloc"));
            }
        }
    }
    // Zero out the newly created optimizer state tensors
    for (auto& m_state : m_states_) {
        if (m_state.is_allocated()) m_state.zero_out(0); // Use default stream for init
    }
    for (auto& v_state : v_states_) {
        if (v_state.is_allocated()) v_state.zero_out(0); // Use default stream for init
    }
     HIP_CHECK(hipStreamSynchronize(0)); // Ensure zeroing is complete
}


void AdamWOptimizer::zero_grad(hipStream_t stream) {
    for (auto* p_param : params_ref_) {
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

void AdamWOptimizer::step(hipStream_t stream) {
    t_++; // Increment timestep

    int state_idx = 0; // Manual index for m_states_ and v_states_
    for (auto* p_param : params_ref_) {
        if (p_param) {
            // Update weights
            if (p_param->weights.is_allocated() && p_param->grad_weights.is_allocated()) {
                if (state_idx < m_states_.size() && m_states_[state_idx].is_allocated() &&
                    state_idx < v_states_.size() && v_states_[state_idx].is_allocated()) {
                    launch_adamw_update_kernel(stream,
                                           (float*)p_param->weights.d_ptr_,
                                           (const float*)p_param->grad_weights.d_ptr_,
                                           (float*)m_states_[state_idx].d_ptr_,
                                           (float*)v_states_[state_idx].d_ptr_,
                                           lr_, beta1_, beta2_, epsilon_,
                                           weight_decay_, // Apply weight decay for weights
                                           t_, p_param->weights.num_elements_);
                } else {
                     throw std::runtime_error("Optimizer M/V state not allocated for weights of: " + p_param->name);
                }
            }
            state_idx++;


            // Update bias (if it exists and has gradients)
            if (p_param->has_bias_ && p_param->bias.is_allocated() && p_param->grad_bias.is_allocated()) {
                 if (state_idx < m_states_.size() && m_states_[state_idx].is_allocated() &&
                    state_idx < v_states_.size() && v_states_[state_idx].is_allocated()) {
                    launch_adamw_update_kernel(stream,
                                           (float*)p_param->bias.d_ptr_,
                                           (const float*)p_param->grad_bias.d_ptr_,
                                           (float*)m_states_[state_idx].d_ptr_,
                                           (float*)v_states_[state_idx].d_ptr_,
                                           lr_, beta1_, beta2_, epsilon_,
                                           0.0f, // No weight decay for biases typically
                                           t_, p_param->bias.num_elements_);
                 } else {
                     throw std::runtime_error("Optimizer M/V state not allocated for bias of: " + p_param->name);
                 }
            }
            state_idx++;
        }
    }
    // HIP_CHECK(hipStreamSynchronize(stream)); // Synchronize if subsequent operations depend on this step immediately
}
