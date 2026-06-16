#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_M = 128;  // 一个 block 负责 C 的多少行
constexpr int BLOCK_N = 128;  // 一个 block 负责 C 的多少列
constexpr int BLOCK_K = 8;    // 每轮沿 K 方向处理多少个元素

constexpr int THREAD_TILE_M = 8;  // 一个 thread 负责 C 子块的多少行
constexpr int THREAD_TILE_N = 8;  // 一个 thread 负责 C 子块的多少列

__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C, int M,
                                             int N, int K) {
  __shared__ float s_a[BLOCK_M][BLOCK_K];
  __shared__ float s_b[BLOCK_K][BLOCK_N];

  const int tx = threadIdx.x;  // thread 在 block 内的列方向编号
  const int ty = threadIdx.y;  // thread 在 block 内的行方向编号
  const int tid = ty * blockDim.x + tx;
  const int num_threads = blockDim.x * blockDim.y;

  // 当前 thread 负责的 C 子块的位置
  const int c_row_start = blockIdx.y * BLOCK_M + ty * THREAD_TILE_M;
  const int c_col_start = blockIdx.x * BLOCK_N + tx * THREAD_TILE_N;

  float acc[THREAD_TILE_M][THREAD_TILE_N] = {0.0};  // 将累加缓存到寄存器中

  // 沿 K 维度分块
  for (int tile_k = 0; tile_k < K; tile_k += BLOCK_K) {
    // ------------------------------------------------------------
    // 1. scalar 加载 A tile 到 shared memory
    //    s_a 形状是 BLOCK_M × BLOCK_K
    // ------------------------------------------------------------
    for (int idx = tid; idx < BLOCK_M * BLOCK_K; idx += num_threads) {
      const int smem_m = idx / BLOCK_K;
      const int smem_k = idx % BLOCK_K;

      const int gmem_m = blockIdx.y * BLOCK_M + smem_m;
      const int gmem_k = tile_k + smem_k;

      if (gmem_m < M && gmem_k < K) {
        s_a[smem_m][smem_k] = A[gmem_m * K + gmem_k];
      } else {
        s_a[smem_m][smem_k] = 0.0f;
      }
    }

    // ------------------------------------------------------------
    // 2. scalar 加载 B tile 到 shared memory
    //    s_b 形状是 BLOCK_K × BLOCK_N
    // ------------------------------------------------------------
    for (int idx = tid; idx < BLOCK_K * BLOCK_N; idx += num_threads) {
      const int smem_k = idx / BLOCK_N;
      const int smem_n = idx % BLOCK_N;

      const int gmem_k = tile_k + smem_k;
      const int gmem_n = blockIdx.x * BLOCK_N + smem_n;

      if (gmem_k < K && gmem_n < N) {
        s_b[smem_k][smem_n] = B[gmem_k * N + gmem_n];
      } else {
        s_b[smem_k][smem_n] = 0.0f;
      }
    }

    __syncthreads();
    // ------------------------------------------------------------
    // 3. 当前 thread 计算自己负责的 8×8 C 子块
    // ------------------------------------------------------------
#pragma unroll
    for (int k = 0; k < BLOCK_K; ++k) {
      float a_frag[THREAD_TILE_M];
      float b_frag[THREAD_TILE_N];

#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
        const int smem_m = ty * THREAD_TILE_M + i;
        a_frag[i] = s_a[smem_m][k];
      }

#pragma unroll
      for (int j = 0; j < THREAD_TILE_N; ++j) {
        const int smem_n = tx * THREAD_TILE_N + j;
        b_frag[j] = s_b[k][smem_n];
      }

#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
          acc[i][j] += a_frag[i] * b_frag[j];
        }
      }
    }

    __syncthreads();
  }
  // ------------------------------------------------------------
  // 4. 写回 C
  // ------------------------------------------------------------
#pragma unroll
  for (int i = 0; i < THREAD_TILE_M; ++i) {
    const int row = c_row_start + i;

#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; ++j) {
      const int col = c_col_start + j;

      if (row < M && col < N) {
        C[row * N + col] = acc[i][j];
      }
    }
  }
}

}  // namespace

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N,
                      int K) {
  dim3 threadsPerBlock(BLOCK_N / THREAD_TILE_N, BLOCK_M / THREAD_TILE_M);
  dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N,
                     (M + BLOCK_M - 1) / BLOCK_M);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);
  cudaDeviceSynchronize();
}
