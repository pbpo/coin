#ifndef HIP_KERNELS_HPP
#define HIP_KERNELS_HPP

#include "common_hip.hpp" // For GpuTensor, BertConfig
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h> // For hiprandState_t

// --- Constants ---
constexpr int THREADS_PER_BLOCK_DEFAULT = 256; // Default, can be overridden
constexpr float M_PI_F = 3.14159265358979323846f;


// RAII(Scope-Bound Resource Management)를 위한 간단한 GPU 포인터 배열 래퍼
struct GpuPtrArray {
    float** d_ptr_ = nullptr;
    size_t count_ = 0;
    hipStream_t stream_ = 0; // Store stream for async deallocation if desired

    GpuPtrArray(size_t count, hipStream_t stream = 0);
    ~GpuPtrArray();

    // 복사 및 이동 생성/대입 연산자 삭제
    GpuPtrArray(const GpuPtrArray&) = delete;
    GpuPtrArray& operator=(const GpuPtrArray&) = delete;
    GpuPtrArray(GpuPtrArray&&) = delete;
    GpuPtrArray& operator=(GpuPtrArray&&) = delete;
};


// --- Kernel Launchers Declarations ---

// Batched GEMM Pointer Setup
void launch_setup_batched_gemm_pointers(
    hipStream_t stream,
    GpuPtrArray& A_ptrs, GpuPtrArray& B_ptrs, GpuPtrArray& C_ptrs,
    const GpuTensor& A_tensor, const GpuTensor& B_tensor, GpuTensor& C_tensor,
    size_t batch_count);

// Dropout
void launch_dropout_forward(
    hipStream_t stream, float* output, float* mask, const float* input,
    size_t num_elements, float prob, float scale, unsigned long long seed);

void launch_dropout_backward(
    hipStream_t stream, float* grad_input, const float* grad_output, const float* mask,
    size_t num_elements, float scale);

// Layer Normalization (Optimized and basic versions)
void launch_layer_norm_forward_optimized(
    hipStream_t stream, float* out, float* mean, float* rstd,
    const float* inp, const float* gamma, const float* beta,
    int B, int C, float epsilon);

void launch_layer_norm_backward_optimized( // Renamed from launch_layer_norm_backward to avoid conflict if basic one is also declared
    hipStream_t stream, float* grad_input, float* grad_gamma, float* grad_beta,
    const float* grad_output, const float* input,
    const float* gamma, const float* mean, const float* rstd,
    int B, int C);

// GELU and Add_Bias_GELU
void launch_add_bias_gelu_kernel( // Keep distinct name from the other add_bias_gelu
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N);

void launch_gelu_backward_kernel( // Keep distinct name
    hipStream_t stream, float* grad_input, const float* grad_output, const float* input, size_t num_elements);

// Reduction Kernels
void launch_reduce_sum_axis0_add_kernel( // Keep distinct name
    hipStream_t stream, float* out_grad, const float* in_grad, int M, int N);

void launch_reduce_sum_axis1_add_kernel( // New kernel launcher from user code
    hipStream_t stream, float* out_grad, const float* in_grad, int rows, int cols);


// Softmax Cross Entropy Loss
void launch_softmax_cross_entropy_loss_backward_optimized( // Renamed from launch_softmax_cross_entropy_loss_backward
    hipStream_t stream, float* grad_logits, const float* logits,
    const int* labels, float* total_loss,
    int B, int S, int V);

// AdamW Optimizer
void launch_adamw_update_kernel( // Keep distinct name
    hipStream_t stream, float* params, const float* grads, float* m, float* v,
    float lr, float beta1, float beta2, float eps, float weight_decay, int t, size_t num_elements);

// Additional Utility Kernels from user code (second part)
void launch_add_bias_kernel( // Keep distinct name
    hipStream_t stream, float* output, const float* input, const float* bias, int M, int N);

void launch_add_row_bias_gelu_kernel( // New kernel launcher from user code
    hipStream_t stream, float* output, const float* input, const float* bias, int rows, int cols);

void launch_scale_kernel( // Keep distinct name
    hipStream_t stream, float* data, float scale, size_t num_elements);

// Embedding Kernels
void launch_add_embeddings_kernel( // Keep distinct name
    hipStream_t stream, float* output, const int* input_ids, const int* token_type_ids,
    const float* word_embeddings, const float* position_embeddings,
    const float* token_type_embeddings, int batch_size, int seq_len,
    int hidden_size, int vocab_size, int max_position_embeddings);

void launch_embedding_backward_kernel( // Keep distinct name
    hipStream_t stream, float* grad_word_embeddings, float* grad_position_embeddings,
    float* grad_token_type_embeddings, const float* grad_output,
    const int* input_ids, const int* token_type_ids,
    int batch_size, int seq_len, int hidden_size);

// Transpose Kernels for Attention
void launch_transpose_for_scores_kernel( // Keep distinct name
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size);

void launch_transpose_back_kernel( // Keep distinct name
    hipStream_t stream, float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size);

void launch_untranspose_kernel( // Added based on usage in BertSelfAttention::backward
    float* output, const float* input,
    int batch_size, int seq_len, int num_heads, int head_size, hipStream_t stream);


// Attention Kernels
void launch_scale_and_mask_kernel( // Keep distinct name
    hipStream_t stream, float* attention_scores, const float* attention_mask,
    int batch_size, int num_heads, int seq_len, float scale);

void launch_softmax_kernel( // Keep distinct name
    hipStream_t stream, float* output, const float* input,
    int M_rows, int N_softmax_dim); // Changed signature for clarity

void launch_softmax_backward_kernel( // Added based on usage in BertSelfAttention::backward
    hipStream_t stream, float* grad_input, const float* grad_output, const float* output,
    int M_rows, int N_softmax_dim); // Changed signature for clarity

// Placeholder for missing element-wise add/accumulate kernels
void launch_elementwise_add_kernel(hipStream_t stream, float* out, const float* in1, const float* in2, size_t num_elements);
void launch_accumulate_kernel(hipStream_t stream, float* target_and_out, const float* to_add, size_t num_elements);

// Backward kernel for add_bias_gelu operation
void launch_gelu_add_bias_backward_kernel(
    hipStream_t stream,
    float* grad_input_before_bias, // Output: Gradient w.r.t. input that bias was added to
    float* grad_bias,              // Output: Gradient w.r.t. bias (accumulated)
    const float* grad_output_after_gelu, // Input: Gradient from the layer above (after GELU)
    const float* input_before_gelu,      // Input: The tensor that was fed into GELU (input + bias)
    int M, // Number of rows (e.g., batch_size * seq_len)
    int N  // Number of columns (e.g., hidden_size or intermediate_size)
);


#endif // HIP_KERNELS_HPP
