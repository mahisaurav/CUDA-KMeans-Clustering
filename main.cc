#include "kmeans.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

bool Update(std::vector<Point>& centers, const std::vector<ClusterSum>& sums) {
  double max_shift = 0;
  for (size_t c = 0; c < centers.size(); ++c) {
    const ClusterSum sum = sums[c];
    if (sum.count == 0) continue;
    const Point next{sum.x / sum.count, sum.y / sum.count};
    max_shift = std::max(max_shift, std::hypot(next.x - centers[c].x,
                                               next.y - centers[c].y));
    centers[c] = next;
  }
  return max_shift <= 1e-6;
}

namespace {
std::vector<Point> Read(const std::string& path) {
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line) || (line != "x,y" && line != "x,y\r")) {
    throw std::runtime_error("Expected CSV header x,y in " + path);
  }
  std::vector<Point> points;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    Point p{};
    char comma;
    if (!(row >> p.x >> comma >> p.y) || comma != ',' ||
        !(row >> std::ws).eof() || std::abs(p.x) > 1e100 || std::abs(p.y) > 1e100 ||
        !std::isfinite(p.x) || !std::isfinite(p.y)) {
      throw std::runtime_error("Invalid coordinate on line " + std::to_string(points.size() + 2));
    }
    if (points.size() == std::numeric_limits<int>::max()) {
      throw std::runtime_error("Too many points");
    }
    points.push_back(p);
  }
  if (!input.eof()) throw std::runtime_error("Input read failed");
  return points;
}

std::vector<Point> Initialize(const std::vector<Point>& points, int k) {
  std::vector<Point> centers{points.front()};
  std::vector<double> nearest(points.size(), std::numeric_limits<double>::infinity());
  while (centers.size() < static_cast<size_t>(k)) {
    const Point last = centers.back();
    for (size_t i = 0; i < points.size(); ++i) {
      const double dx = points[i].x - last.x;
      const double dy = points[i].y - last.y;
      nearest[i] = std::min(nearest[i], dx * dx + dy * dy);
    }
    centers.push_back(points[std::max_element(nearest.begin(), nearest.end()) - nearest.begin()]);
  }
  return centers;
}

Result RunCpu(const std::vector<Point>& points, std::vector<Point> centers, int limit) {
  const int k = centers.size();
  std::vector<ClusterSum> sums(k);
  std::vector<int> labels(points.size());
  int iterations = 0;
  while (iterations < limit) {
    ++iterations;
    std::fill(sums.begin(), sums.end(), ClusterSum{});
    for (Point p : points) {
      ClusterSum& sum = sums[Nearest(p, centers.data(), k)];
      sum.x += p.x;
      sum.y += p.y;
      ++sum.count;
    }
    if (Update(centers, sums)) break;
  }
  for (size_t i = 0; i < points.size(); ++i) {
    labels[i] = Nearest(points[i], centers.data(), k);
  }
  return {std::move(centers), std::move(labels), iterations};
}

double Inertia(const std::vector<Point>& points, const Result& result) {
  double sum = 0;
  for (size_t i = 0; i < points.size(); ++i) {
    const Point center = result.centers[result.labels[i]];
    const double dx = points[i].x - center.x;
    const double dy = points[i].y - center.y;
    sum += dx * dx + dy * dy;
  }
  return sum;
}

void Write(const std::filesystem::path& directory, const std::vector<Point>& points,
           const Result& result) {
  std::filesystem::create_directories(directory);
  std::ofstream labels(directory / "clusters.csv");
  std::ofstream centers(directory / "centroids.csv");
  labels.exceptions(std::ios::failbit | std::ios::badbit);
  centers.exceptions(std::ios::failbit | std::ios::badbit);
  labels << std::setprecision(17) << "x,y,cluster\n";
  centers << std::setprecision(17) << "cluster,x,y\n";
  for (size_t i = 0; i < points.size(); ++i) {
    labels << points[i].x << ',' << points[i].y << ',' << result.labels[i] << '\n';
  }
  for (size_t c = 0; c < result.centers.size(); ++c) {
    centers << c << ',' << result.centers[c].x << ',' << result.centers[c].y << '\n';
  }
  labels.close();
  centers.close();
}

int Integer(const std::string& text, int maximum) {
  size_t end = 0;
  const int value = std::stoi(text, &end);
  if (end != text.size() || value < 1 || value > maximum) {
    throw std::runtime_error("Integer argument outside supported range: " + text);
  }
  return value;
}
}  // namespace

int main(int argc, char** argv) {
  const char* usage = "Usage: kmeans INPUT.csv K MAX_ITERATIONS {cpu|gpu} OUTPUT_DIR [REPEATS]\n";
  try {
    if (argc == 2 && std::string(argv[1]) == "--help") {
      std::cout << usage;
      return 0;
    }
    if (argc < 6 || argc > 7) throw std::runtime_error(usage);
    const int k = Integer(argv[2], 256);
    const int limit = Integer(argv[3], 10000);
    const std::string mode = argv[4];
    const int repeats = argc == 7 ? Integer(argv[6], 100) : 1;
    if (mode != "cpu" && mode != "gpu") throw std::runtime_error(usage);
#ifndef ENABLE_CUDA
    if (mode == "gpu") throw std::runtime_error("Build with make cuda to use GPU mode");
#endif
    const auto points = Read(argv[1]);
    if (points.size() < static_cast<size_t>(k)) throw std::runtime_error("K exceeds row count");
    const auto initial = Initialize(points, k);
    auto run = RunCpu;
#ifdef ENABLE_CUDA
    if (mode == "gpu") {
      WarmGpu();
      run = RunGpu;
    }
#endif
    if (repeats > 1) run(points, initial, limit);
    std::vector<double> times;
    Result result{};
    for (int trial = 0; trial < repeats; ++trial) {
      const auto start = std::chrono::steady_clock::now();
      result = run(points, initial, limit);
      const auto end = std::chrono::steady_clock::now();
      times.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    std::sort(times.begin(), times.end());
    const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
    Write(argv[5], points, result);
    std::cout << std::setprecision(12) << mode << ',' << points.size() << ','
              << k << ',' << result.iterations << ',' << repeats << ',' << median
              << ',' << Inertia(points, result) << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
