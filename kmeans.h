#ifndef KMEANS_H_
#define KMEANS_H_
#include <vector>

struct Point { double x, y; };
struct ClusterSum { double x, y; int count; };
struct Result {
  std::vector<Point> centers;
  std::vector<int> labels;
  int iterations;
};

#ifdef __CUDACC__
__host__ __device__
#endif
inline int Nearest(Point p, const Point* centers, int k) {
  int best = 0;
  double distance = 0;
  for (int c = 0; c < k; ++c) {
    const double dx = p.x - centers[c].x;
    const double dy = p.y - centers[c].y;
    const double squared = dx * dx + dy * dy;
    if (c == 0 || squared < distance) {
      distance = squared;
      best = c;
    }
  }
  return best;
}

bool Update(std::vector<Point>& centers, const std::vector<ClusterSum>& sums);
Result RunGpu(const std::vector<Point>& points, std::vector<Point> centers, int limit);
void WarmGpu();
#endif
