#ifndef COMMON_HIP_HPP
#define COMMON_HIP_HPP

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// --- 유틸리티 매크로 ---
#define HIP_CHECK(cmd) do { \
    hipError_t err = cmd; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s in %s at line %d\n", hipGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error("HIP error"); \
    } \
} while(0)

#define ROCBLAS_CHECK(cmd) do { \
    rocblas_status err = cmd; \
    if (err != rocblas_status_success) { \
        fprintf(stderr, "rocBLAS Error: %s (%d) in %s at line %d\n", rocblas_status_to_string(err), err, __FILE__, __LINE__); \
        throw std::runtime_error("rocBLAS error"); \
    } \
} while(0)

// --- 데이터 타입 ---
enum class DataType { FLOAT32, INT32 };

// --- 전방 선언 ---
class Parameter;
struct BertConfig;

// --- GpuTensor: GPU 메모리를 관리하는 텐서 클래스 ---
class GpuTensor {
public:
    void* d_ptr_ = nullptr;
    std::vector<int> dims_;
    size_t num_elements_ = 0;
    size_t element_size_ = 0;
    std::string name_;
    bool allocated_ = false;
    DataType dtype = DataType::FLOAT32;
    bool is_view_ = false; // 뷰(View) 여부 플래그 추가

    GpuTensor(const std::string& name = "");
    GpuTensor(const std::vector<int>& dimensions, const std::string& name = "", DataType type = DataType::FLOAT32);
    ~GpuTensor();

    // 복사 생성자 및 대입 연산자 삭제
    GpuTensor(const GpuTensor&) = delete;
    GpuTensor& operator=(const GpuTensor&) = delete;

    // 이동 생성자 및 대입 연산자
    GpuTensor(GpuTensor&& other) noexcept;
    GpuTensor& operator=(GpuTensor&& other) noexcept;

    void allocate(const std::vector<int>& new_dims);
    void set_view(void* ptr, const std::vector<int>& dims, DataType type = DataType::FLOAT32); // 뷰 설정 함수 추가
    void free();
    void zero_out(hipStream_t stream);

    template<typename T>
    void to_gpu(const std::vector<T>& data);
    template<typename T>
    std::vector<T> to_cpu() const;
    void copy_from_gpu(const GpuTensor& src, hipStream_t stream);

    size_t size_in_bytes() const { return num_elements_ * element_size_; }
    bool is_allocated() const { return allocated_; }
    int dim_size(int i) const { return dims_.at(i); }
};

// --- Parameter: 학습 가능한 가중치와 기울기를 관리하는 클래스 ---
class Parameter {
public:
    GpuTensor weights;
    GpuTensor bias;
    GpuTensor grad_weights;
    GpuTensor grad_bias;
    bool has_bias_;
    std::string name;

    Parameter(const std::vector<int>& weight_dims, const std::string& name);
    Parameter(const std::vector<int>& weight_dims, const std::vector<int>& bias_dims, const std::string& name);

    void initialize_random(float mean = 0.0f, float stddev = 0.02f);
    void allocate_gradients();
};

#endif 
