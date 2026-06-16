#include <cuda_runtime.h>

namespace {

constexpr int TILE_SIZE = 16;

__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C, int M,
                                             int N, int K) {
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

  const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int tile = 0; tile < N; tile += TILE_SIZE) {
    const int a_col = tile + threadIdx.x;
    const int b_row = tile + threadIdx.y;

    tile_a[threadIdx.y][threadIdx.x] =
        (row < M && a_col < N) ? A[row * N + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < N && col < K) ? B[b_row * K + col] : 0.0f;

    __syncthreads();

#pragma unroll
    for (int i = 0; i < TILE_SIZE; ++i) {
      sum += tile_a[threadIdx.y][i] * tile_b[i][threadIdx.x];
    }

    __syncthreads();
  }

  if (row < M && col < K) {
    C[row * K + col] = sum;
  }
}

}  // namespace

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N,
                      int K) {
  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}
