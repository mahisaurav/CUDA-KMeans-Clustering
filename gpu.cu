#include "kmeans.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
void Check(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
  }
}

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) : bytes_(size * sizeof(T)) {
    Check(cudaMalloc(reinterpret_cast<void**>(&data), bytes_), "cudaMalloc");
  }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  void Upload(const T* src) {
    Check(cudaMemcpy(data, src, bytes_, cudaMemcpyHostToDevice), "upload");
  }
  void Download(T* dst) const {
    Check(cudaMemcpy(dst, data, bytes_, cudaMemcpyDeviceToHost), "download");
  }
  void Clear() { Check(cudaMemset(data, 0, bytes_), "clear"); }

  T* data = nullptr;

 private:
  size_t bytes_;
};

__global__ void Assign(const Point* points, int n, const Point* centers,
                       int k, ClusterSum* totals, int* labels) {
  extern __shared__ ClusterSum bins[];
  if (totals) {
    for (int c = threadIdx.x; c < k; c += blockDim.x) bins[c] = {};
    __syncthreads();
  }
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const Point point = points[i];
    const int c = Nearest(point, centers, k);
    if (totals) {
      atomicAdd(&bins[c].x, point.x);
      atomicAdd(&bins[c].y, point.y);
      atomicAdd(&bins[c].count, 1);
    } else {
      labels[i] = c;
    }
  }
  if (totals) {
    __syncthreads();
    for (int c = threadIdx.x; c < k; c += blockDim.x) {
      if (bins[c].count == 0) continue;
      atomicAdd(&totals[c].x, bins[c].x);
      atomicAdd(&totals[c].y, bins[c].y);
      atomicAdd(&totals[c].count, bins[c].count);
    }
  }
}
}  // namespace

void WarmGpu() { Check(cudaFree(nullptr), "initialize CUDA"); }

Result RunGpu(const std::vector<Point>& points, std::vector<Point> centers, int limit) {
  const int n = points.size();
  const int k = centers.size();
  DeviceBuffer<Point> device_points(n);
  DeviceBuffer<Point> device_centers(k);
  DeviceBuffer<ClusterSum> device_sums(k);
  DeviceBuffer<int> device_labels(n);
  std::vector<ClusterSum> sums(k);
  std::vector<int> labels(n);
  device_points.Upload(points.data());
  constexpr int threads = 256;
  const int blocks = (size_t(n) + threads - 1) / threads;
  int iterations = 0;
  while (iterations < limit) {
    ++iterations;
    device_sums.Clear();
    device_centers.Upload(centers.data());
    Assign<<<blocks, threads, k * sizeof(ClusterSum)>>>(
        device_points.data, n, device_centers.data, k, device_sums.data, nullptr);
    Check(cudaGetLastError(), "launch assignment and reduction");
    device_sums.Download(sums.data());
    if (Update(centers, sums)) break;
  }
  device_centers.Upload(centers.data());
  Assign<<<blocks, threads>>>(device_points.data, n, device_centers.data,
                              k, nullptr, device_labels.data);
  Check(cudaGetLastError(), "launch final assignment");
  device_labels.Download(labels.data());
  return {std::move(centers), std::move(labels), iterations};
}
