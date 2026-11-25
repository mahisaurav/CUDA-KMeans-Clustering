CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -pedantic
NVCC ?= nvcc
CUDA_ARCH ?= sm_75

.PHONY: cpu cuda clean
cpu: build/kmeans_cpu
cuda: build/kmeans

build:
	mkdir -p $@

build/kmeans_cpu: main.cc kmeans.h | build
	$(CXX) $(CXXFLAGS) $< -o $@

build/kmeans: main.cc gpu.cu kmeans.h | build
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) -DENABLE_CUDA main.cc gpu.cu -o $@

clean:
	rm -rf build
