#pragma once

#include <iostream>
#include <numeric>
#include <vector>

namespace storage {

template <typename T>
class CircularBuffer {
 public:
  CircularBuffer() : max_len_(0), head_(0), tail_(0), full_(true) {}

  explicit CircularBuffer(size_t max_len_) : max_len_(max_len_), head_(0), tail_(0), full_(false) {
    this->buffer_.resize(this->max_len_);
  }

  void push(const T& value) {
    this->buffer_[this->head_] = value;
    if (this->full_) this->tail_ = (this->tail_ + 1) % this->max_len_;
    this->head_ = (this->head_ + 1) % this->max_len_;
    this->full_ = this->head_ == this->tail_;
  }

  void push(const std::vector<T>& values) {
    for (const auto& value : values) {
      this->push(value);
    }
  }

  size_t size() const {
    if (this->full_) return this->max_len_;
    if (this->head_ >= this->tail_) return this->head_ - this->tail_;
    return this->max_len_ + this->head_ - this->tail_;
  }

  float mean() const {
    if (size() == 0) return 0.0;

    float sum;
    if (this->full_)
      sum = static_cast<float>(std::accumulate(this->buffer_.begin(), this->buffer_.end(), T(0)));
    else
      sum = static_cast<float>(std::accumulate(this->buffer_.begin() + this->tail_,
                                               this->buffer_.begin() + this->head_, T(0)));
    return sum / size();
  }

  void clear() {
    this->head_ = 0;
    this->tail_ = 0;
    this->full_ = false;
  }

  friend std::ostream& operator<<(std::ostream& os, const CircularBuffer& cb) {
    if (cb.size() == 0) {
      os << "Buffer is empty.\n";
      return os;
    }

    for (size_t i = 0; i < cb.size(); ++i) {
      os << cb.buffer_[(cb.tail_ + i) % cb.max_len_] << " ";
    }

    return os;
  }

 private:
  std::vector<T> buffer_;
  size_t head_;
  size_t tail_;
  size_t max_len_;
  bool full_;
};

using CircularBufferFloat = CircularBuffer<float>;
using CircularBufferInt = CircularBuffer<int>;

using CircularBufferFloatPointer = std::unique_ptr<CircularBufferFloat>;
using CircularBufferIntPointer = std::unique_ptr<CircularBufferInt>;

}  // namespace storage
