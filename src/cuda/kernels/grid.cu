#include "../includes/honeycomb.h"

extern "C" __global__ void generate_2d_grid_betaf(DartIdType* out, size_t n_x, size_t n_y, size_t n_out) {
    // cell coordinates in the generated grid
    uint64_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t iy = threadIdx.y + blockIdx.y * blockDim.y;
    // dart of the thread
    uint64_t dart = 1 + 4 * ix + 4 * n_x * iy + threadIdx.z;
    // beta images
    if (dart*3 + 2 < n_out) {
      switch ((1+threadIdx.z) % 4) {
          case 1: // d1
              out[dart*3] = dart + 3;
              out[dart*3+1] = dart + 1;
              out[dart*3+2] = (iy == 0) ? 0 : dart + 2 - 4 * n_x;
              break;
          case 2: // d2
              out[dart*3] = dart - 1;
              out[dart*3+1] = dart + 1;
              out[dart*3+2] = (ix == n_x - 1) ? 0 : dart + 6;
              break;
          case 3: // d3
              out[dart*3] = dart - 1;
              out[dart*3+1] = dart + 1;
              out[dart*3+2] = (iy == n_y - 1) ? 0 : dart - 2 + 4 * n_x;
              break;
          case 0: // d4
              out[dart*3] = dart - 1;
              out[dart*3+1] = dart - 3;
              out[dart*3+2] = (ix == 0) ? 0 : dart - 6;
              break;
      }
    }
}

extern "C" __global__ void generate_2d_grid_vertices(DVertex2* out, float lc_x, float lc_y, size_t n_x, size_t n_y, size_t n_out) {
    // cell coordinates in the generated grid
    uint64_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    uint64_t iy = threadIdx.y + blockIdx.y * blockDim.y;
    // dart of the thread
    uint64_t dart = 1 + 4 * ix + 4 * n_x * iy + threadIdx.z;
    // vertex associated to the dart; we'll filter useful values when building on the host
    if (dart < n_out) {
      switch ((1+threadIdx.z) % 4) {
          case 1: // d1 -> bottom left corner
              out[dart] = {{ lc_x * ix, lc_y * iy }};
              break;
          case 2: // d2 -> bottom right corner
              out[dart] = {{ lc_x * (ix + 1), lc_y * iy }};
              break;
          case 3: // d3 -> top right corner
              out[dart] = {{ lc_x * (ix + 1), lc_y * (iy + 1) }};
              break;
          case 0: // d4 -> top left corner
              out[dart] = {{ lc_x * ix, lc_y * (iy + 1) }};
              break;
      }
    }
}

