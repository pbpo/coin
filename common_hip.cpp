#include "common_hip.hpp"
#include <random>
#include <numeric>

// ============================================================================
// GpuTensor Class Implementations
// ============================================================================

GpuTensor::GpuTensor(const std::string& name) : name_(name) {}

GpuTensor::GpuTensor(const std::vector<int>& dimensions, const std::string& name, DataType type)
    : dims_(dimensions), name_(name), dtype(type) {
    allocate(dims_);
}

GpuTensor::~GpuTensor() {
    free();
}

GpuTensor::GpuTensor(GpuTensor&& other) noexcept {
    d_ptr_ = other.d_ptr_;
    dims_ = std::move(other.dims_);
    num_elements_ = other.num_elements_;
    element_size_ = other.element_size_;
    name_ = std::move(other.name_);
    allocated_ = other.allocated_;
    dtype = other.dtype;
    is_view_ = other.is_view_; // 뷰 플래그 이동
    
    other.d_ptr_ = nullptr;
    other.allocated_ = false;
    other.is_view_ = false;
    other.num_elements_ = 0;
}

GpuTensor& GpuTensor::operator=(GpuTensor&& other) noexcept {
    if (this != &other) {
        free();
        d_ptr_ = other.d_ptr_;
        dims_ = std::move(other.dims_);
        num_elements_ = other.num_elements_;
        element_size_ = other.element_size_;
        name_ = std::move(other.name_);
        allocated_ = other.allocated_;
        dtype = other.dtype;
        is_view_ = other.is_view_; // 뷰 플래그 이동

        other.d_ptr_ = nullptr;
        other.allocated_ = false;
        other.is_view_ = false;
        other.num_elements_ = 0;
    }
    return *this;
}

void GpuTensor::allocate(const std::vector<int>& new_dims) {
    dims_ = new_dims;
    is_view_ = false; // 새로 할당하는 것은 뷰가 아님
    size_t new_num_elements = 1;
    for (int dim : dims_) {
        if (dim <= 0) throw std::runtime_error("Tensor dimension must be positive for " + name_);
        new_num_elements *= dim;
    }

    if (new_num_elements == 0) {
        free();
        return;
    }

    element_size_ = (dtype == DataType::INT32) ? sizeof(int) : sizeof(float);
    size_t new_size_bytes = new_num_elements * element_size_;
    size_t current_size_bytes = allocated_ ? num_elements_ * element_size_ : 0;

    if (new_size_bytes != current_size_bytes || d_ptr_ == nullptr) {
        free();
        if (new_num_elements > 0) {
            num_elements_ = new_num_elements;
            HIP_CHECK(hipMalloc(&d_ptr_, new_size_bytes));
            allocated_ = true;
        }
    } else {
        num_elements_ = new_num_elements;
    }
}

// *** 추가된 뷰 설정 함수 ***
void GpuTensor::set_view(void* ptr, const std::vector<int>& new_dims, DataType type) {
    free(); // 기존 메모리가 있다면 해제
    d_ptr_ = ptr;
    dims_ = new_dims;
    is_view_ = true; // 뷰 플래그 설정
    dtype = type;
    element_size_ = (dtype == DataType::INT32) ? sizeof(int) : sizeof(float);
    num_elements_ = 1;
    for(int dim : dims_) {
        num_elements_ *= dim;
    }
    allocated_ = (d_ptr_ != nullptr);
}


void GpuTensor::free() {
    // *** 수정됨: 뷰가 아닐 경우에만 메모리 해제 ***
    if (allocated_ && d_ptr_ != nullptr && !is_view_) {
        hipFree(d_ptr_);
    }
    d_ptr_ = nullptr;
    allocated_ = false;
    is_view_ = false;
    num_elements_ = 0;
}

void GpuTensor::zero_out(hipStream_t stream) {
    if (allocated_ && d_ptr_ != nullptr) {
        HIP_CHECK(hipMemsetAsync(d_ptr_, 0, size_in_bytes(), stream));
    }
}

template<typename T>
void GpuTensor::to_gpu(const std::vector<T>& data) {
    if (!allocated_ || data.size() != num_elements_) {
        std::vector<int> current_dims = dims_.empty() ? std::vector<int>{(int)data.size()} : dims_;
        allocate(current_dims);
        if (data.size() != num_elements_) {
            throw std::runtime_error("Tensor size mismatch for " + name_);
        }
    }
    HIP_CHECK(hipMemcpy(d_ptr_, data.data(), size_in_bytes(), hipMemcpyHostToDevice));
}

template<typename T>
std::vector<T> GpuTensor::to_cpu() const {
    if (!allocated_ || d_ptr_ == nullptr) {
        throw std::runtime_error("Tensor " + name_ + " not allocated on GPU.");
    }
    std::vector<T> data(num_elements_);
    HIP_CHECK(hipMemcpy(data.data(), d_ptr_, size_in_bytes(), hipMemcpyDeviceToHost));
    return data;
}

// 명시적 템플릿 인스턴스화
template void GpuTensor::to_gpu<float>(const std::vector<float>& data);
template std::vector<float> GpuTensor::to_cpu<float>() const;
template void GpuTensor::to_gpu<int>(const std::vector<int>& data);
template std::vector<int> GpuTensor::to_cpu<int>() const;

void GpuTensor::copy_from_gpu(const GpuTensor& src, hipStream_t stream) {
    if (!src.allocated_) {
         throw std::runtime_error("Source tensor " + src.name_ + " is not allocated.");
    }
    if (!allocated_ || num_elements_ != src.num_elements_ || dtype != src.dtype) {
        dtype = src.dtype;
        allocate(src.dims_);
    }
    HIP_CHECK(hipMemcpyAsync(d_ptr_, src.d_ptr_, size_in_bytes(), hipMemcpyDeviceToDevice, stream));
}

// ============================================================================
// Parameter Class Implementations
// ============================================================================
Parameter::Parameter(const std::vector<int>& weight_dims, const std::string& name)
    : weights(weight_dims, name + "_w"), has_bias_(false), name(name) {}

Parameter::Parameter(const std::vector<int>& weight_dims, const std::vector<int>& bias_dims, const std::string& name)
    : weights(weight_dims, name + "_w"), bias(bias_dims, name + "_b"), has_bias_(true), name(name) {}

void Parameter::initialize_random(float mean, float stddev) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, stddev);

    if (weights.is_allocated()) {
        std::vector<float> w_data(weights.num_elements_);
        for(auto& val : w_data) val = dist(gen);
        weights.to_gpu(w_data);
    }

    if (has_bias_ && bias.is_allocated()) {
        std::vector<float> b_data(bias.num_elements_, 0.0f); // Bias는 0으로 초기화
        bias.to_gpu(b_data);
    }
}

void Parameter::allocate_gradients() {
    if (weights.is_allocated()) {
        grad_weights.allocate(weights.dims_);
        grad_weights.name_ = name + "_gw";
    }
    if (has_bias_ && bias.is_allocated()) {
        grad_bias.allocate(bias.dims_);
        grad_bias.name_ = name + "_gb";
    }
}
