#ifndef ATTENTION_HIP_HPP
#define ATTENTION_HIP_HPP

#include "common_hip.hpp"
#include "nn_layers_hip.hpp"
#include "bert_components_hip.hpp"
#include <vector>
#include <string>

// ... (SelfAttentionCache 구조체는 변경 없음)

struct BertAttentionCache {
    SelfAttentionCache self_attention_cache;
    DenseLayerCache output_dense_cache;
    DropoutCache output_dropout_cache;
    LayerNormCache output_layernorm_cache;
    const GpuTensor* attention_input = nullptr;
};

// ... (BertSelfAttention 클래스는 변경 없음)

class BertAttention {
private:
    BertSelfAttention self_attention_;
    DenseLayer output_dense_;
    Dropout output_dropout_;
    LayerNorm output_layernorm_;

public:
    BertAttention(const BertConfig& config, const std::string& name_prefix);

    // *** 수정됨: 모호한 출력을 명시적 출력 텐서로 변경 ***
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_tensor,
                 const GpuTensor& attention_mask,
                 GpuTensor& output_tensor, // 명시적 출력 텐서 추가
                 BertAttentionCache& cache,
                 bool is_training);

    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output_tensor,
                  BertAttentionCache& cache,
                  GpuTensor& grad_input_tensor);

    std::vector<Parameter*> get_parameters();
};

#endif // ATTENTION_HIP_HPP

