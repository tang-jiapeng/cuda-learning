#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include "cuda_utils.cuh"

// All utilities live in the cuda_learning namespace to avoid polluting global scope.
namespace cuda_learning {

// ---------------------------------------------------------------------------
// DeviceBuffer<T> — RAII wrapper for device memory (cudaMalloc / cudaFree)
// ---------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : count_(count) {
        checkCudaErrors(cudaMalloc(&ptr_, count_ * sizeof(T)));
    }

    ~DeviceBuffer() {
        release();
    }

    // Non-copyable
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    // Movable
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() const noexcept {
        return ptr_;
    }
    std::size_t size() const noexcept {
        return count_;
    }
    std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }
    bool empty() const noexcept {
        return ptr_ == nullptr;
    }

    /// Upload count elements from a host pointer (defaults to the full buffer).
    void upload(const T* host_src, std::size_t count = 0) {
        const std::size_t n = count ? count : count_;
        checkCudaErrors(
            cudaMemcpy(ptr_, host_src, n * sizeof(T), cudaMemcpyHostToDevice));
    }

    /// Download count elements to a host pointer (defaults to the full buffer).
    void download(T* host_dst, std::size_t count = 0) const {
        const std::size_t n = count ? count : count_;
        checkCudaErrors(
            cudaMemcpy(host_dst, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost));
    }

    /// Zero-fill the entire buffer.
    void zero() {
        checkCudaErrors(cudaMemset(ptr_, 0, count_ * sizeof(T)));
    }

   private:
    void release() {
        if (ptr_) {
            cudaFree(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// HostBuffer<T> — RAII wrapper for pinned host memory (cudaMallocHost / cudaFreeHost)
//
// Pinned memory allows DMA transfers, giving higher PCIe bandwidth compared
// with ordinary malloc'd memory.
// ---------------------------------------------------------------------------

template <typename T>
class HostBuffer {
   public:
    HostBuffer() = default;

    explicit HostBuffer(std::size_t count) : count_(count) {
        checkCudaErrors(cudaMallocHost(&ptr_, count_ * sizeof(T)));
    }

    ~HostBuffer() {
        release();
    }

    // Non-copyable
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;

    // Movable
    HostBuffer(HostBuffer&& other) noexcept : ptr_(other.ptr_), count_(other.count_) {
        other.ptr_ = nullptr;
        other.count_ = 0;
    }
    HostBuffer& operator=(HostBuffer&& other) noexcept {
        if (this != &other) {
            release();
            ptr_ = other.ptr_;
            count_ = other.count_;
            other.ptr_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() noexcept {
        return ptr_;
    }
    const T* get() const noexcept {
        return ptr_;
    }
    std::size_t size() const noexcept {
        return count_;
    }
    std::size_t bytes() const noexcept {
        return count_ * sizeof(T);
    }
    bool empty() const noexcept {
        return ptr_ == nullptr;
    }

    T& operator[](std::size_t i) noexcept {
        return ptr_[i];
    }
    const T& operator[](std::size_t i) const noexcept {
        return ptr_[i];
    }

    // Pointer-iterator interface (enables range-based for loops)
    T* begin() noexcept {
        return ptr_;
    }
    T* end() noexcept {
        return ptr_ + count_;
    }
    const T* begin() const noexcept {
        return ptr_;
    }
    const T* end() const noexcept {
        return ptr_ + count_;
    }

   private:
    void release() {
        if (ptr_) {
            cudaFreeHost(ptr_);
            ptr_ = nullptr;
            count_ = 0;
        }
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace cuda_learning
