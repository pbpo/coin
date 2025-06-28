#ifndef ATTENTION_HIP_HPP
#define ATTENTION_HIP_HPP

#include "common_hip.hpp"          // GpuTensor, Parameter 등 공통 데이터 구조 포함
#include "nn_layers_hip.hpp"       // DenseLayer, Dropout, LayerNorm 등 신경망 계층 포함
#include "bert_components_hip.hpp" // BertConfig 등 BERT 구성 요소 포함
#include <vector>
#include <string>

// --- 어텐션 계층을 위한 캐시(Cache) 구조체 정의 ---
// 역전파 계산 시 필요한 중간 값들을 저장하기 위한 구조체입니다.

/**
 * @struct SelfAttentionCache
 * @brief BertSelfAttention 계층의 순전파 과정에서 생성되는 중간 텐서들을 저장합니다.
 * 이 값들은 역전파 시 그래디언트 계산에 재사용됩니다.
 */
struct SelfAttentionCache {
    // 순전파 입력 텐서들의 포인터
    const GpuTensor* input_hidden_states = nullptr; // 입력 텐서 (B, S, H)
    const GpuTensor* attention_mask = nullptr;    // 어텐션 마스크 텐서

    // 순전파 과정에서 계산되는 중간 텐서들
    GpuTensor q_proj; // Query 선형 투영 결과 (B, S, H)
    GpuTensor k_proj; // Key 선형 투영 결과 (B, S, H)
    GpuTensor v_proj; // Value 선형 투영 결과 (B, S, H)

    GpuTensor q_reshaped; // Multi-head 처리를 위해 Reshape된 Query (B, N, S, A)
    GpuTensor k_reshaped; // Multi-head 처리를 위해 Reshape된 Key (B, N, S, A)
    GpuTensor v_reshaped; // Multi-head 처리를 위해 Reshape된 Value (B, N, S, A)

    GpuTensor attention_scores; // 어텐션 스코어 (Q * K^T) 결과 (B, N, S, S)
    GpuTensor attention_probs;  // Softmax와 Dropout이 적용된 어텐션 확률 (B, N, S, S)

    GpuTensor context_reshaped; // 어텐션 확률과 Value를 곱한 후의 Context (B, N, S, A)
    GpuTensor context_layer;    // 최종적으로 Reshape된 Context (B, S, H)

    // 내부 신경망 계층들의 캐시
    DenseLayerCache q_dense_cache;
    DenseLayerCache k_dense_cache;
    DenseLayerCache v_dense_cache;
    DropoutCache attention_probs_dropout_cache;
};

/**
 * @struct BertAttentionCache
 * @brief BertAttention 전체 모듈(Self-Attention + Self-Output)의 캐시 구조체.
 */
struct BertAttentionCache {
    SelfAttentionCache self_attention_cache; // BertSelfAttention 계층의 캐시

    // BertSelfOutput 부분의 캐시
    DenseLayerCache output_dense_cache;     // 출력 Dense 레이어의 캐시
    DropoutCache output_dropout_cache;        // 출력 Dropout의 캐시
    LayerNormCache output_layernorm_cache;    // 최종 LayerNorm의 캐시

    // 잔차 연결(Residual Connection)을 위해 BertAttention 블록의 원본 입력 텐서를 저장
    const GpuTensor* attention_input = nullptr;
};


// --- 어텐션 계층 클래스 정의 ---

/**
 * @class BertSelfAttention
 * @brief BERT의 핵심인 Multi-Head Self-Attention을 계산하는 클래스.
 * 입력으로 들어온 hidden_states를 Q, K, V로 투영하고, 어텐션 스코어를 계산하여 Context 벡터를 생성합니다.
 */
class BertSelfAttention {
private:
    DenseLayer query_, key_, value_; // Q, K, V를 생성하기 위한 Dense 레이어
    Dropout dropout_;                // 어텐션 확률에 적용될 드롭아웃

    int num_attention_heads_;     // 어텐션 헤드의 수
    int attention_head_size_;     // 각 헤드의 차원 크기

public:
    /**
     * @brief BertSelfAttention 생성자.
     * @param config BertConfig 객체.
     * @param name_prefix 파라미터 이름에 사용될 접두사.
     */
    BertSelfAttention(const BertConfig& config, const std::string& name_prefix);

    /**
     * @brief 순전파를 수행합니다.
     * @param hidden_states 입력 텐서 (B, S, H).
     * @param attention_mask 어텐션 마스크.
     * @param cache 역전파에 사용될 중간 값들을 저장할 캐시 객체.
     * @param is_training 학습 모드 여부.
     */
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& hidden_states,
                 const GpuTensor& attention_mask,
                 SelfAttentionCache& cache,
                 bool is_training);

    /**
     * @brief 역전파를 수행하여 입력 및 파라미터에 대한 그래디언트를 계산합니다.
     * @param grad_context_layer_output 상위 계층에서 전달된 context_layer에 대한 그래디언트.
     * @param cache 순전파 시 저장된 캐시 객체.
     * @param grad_input_hidden_states 계산된 입력(hidden_states)에 대한 그래디언트 (출력).
     */
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_context_layer_output,
                  SelfAttentionCache& cache,
                  GpuTensor& grad_input_hidden_states);

    /**
     * @brief 이 계층에 속한 모든 파라미터의 포인터 리스트를 반환합니다.
     */
    std::vector<Parameter*> get_parameters();
};

/**
 * @class BertAttention
 * @brief Self-Attention 연산과 그 후의 Output 레이어(Dense, Dropout, Residual, LayerNorm)를 포함하는 전체 어텐션 모듈.
 */
class BertAttention {
private:
    BertSelfAttention self_attention_;   // Self-Attention 계산 파트
    DenseLayer output_dense_;          // Self-Attention 출력에 적용되는 Dense 레이어
    Dropout output_dropout_;           // 드롭아웃
    LayerNorm output_layernorm_;       // 잔차 연결 후 적용되는 Layer Normalization

public:
    BertAttention(const BertConfig& config, const std::string& name_prefix);

    /**
     * @brief BertAttention 모듈 전체의 순전파를 수행합니다.
     * @param input_tensor 모듈의 입력이자 잔차 연결에 사용될 텐서 (B, S, H).
     * @param attention_mask 어텐션 마스크.
     * @param cache 역전파를 위한 캐시 객체.
     * @param is_training 학습 모드 여부.
     */
    void forward(rocblas_handle blas_handle, hipStream_t stream,
                 const GpuTensor& input_tensor,
                 const GpuTensor& attention_mask,
                 GpuTensor& output_tensor, // Output tensor
                 BertAttentionCache& cache,
                 bool is_training);

    /**
     * @brief BertAttention 모듈 전체의 역전파를 수행합니다.
     * @param grad_output_layernorm 상위 계층에서 전달된 최종 출력에 대한 그래디언트.
     * @param cache 순전파 시 저장된 캐시 객체.
     * @param grad_input_tensor 계산된 모듈 입력(input_tensor)에 대한 그래디언트 (출력).
     */
    void backward(rocblas_handle blas_handle, hipStream_t stream,
                  const GpuTensor& grad_output_layernorm,
                  BertAttentionCache& cache,
                  GpuTensor& grad_input_tensor);

    /**
     * @brief 이 모듈 및 하위 모듈의 모든 파라미터 포인터 리스트를 반환합니다.
     */
    std::vector<Parameter*> get_parameters();
};

#endif // ATTENTION_HIP_HPP
