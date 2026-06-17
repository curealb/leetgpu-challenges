#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 128;
constexpr int BLOCK_K = 8;

constexpr int THREAD_TILE_M = 8;
constexpr int THREAD_TILE_N = 8;

constexpr int BLOCK_SIZE_X = BLOCK_N / THREAD_TILE_N;           // 16
constexpr int BLOCK_SIZE_Y = BLOCK_M / THREAD_TILE_M;           // 16
constexpr int THREADS_PER_BLOCK = BLOCK_SIZE_X * BLOCK_SIZE_Y;  // 256

__device__ __forceinline__ float4 make_f4(float x, float y, float z, float w) {
  float4 v;
  v.x = x;
  v.y = y;
  v.z = z;
  v.w = w;
  return v;
}

__global__ void matrix_multiplication_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C, int M,
                                             int N, int K) {
  // A 在 shared memory 中转置存储：
  // 原本 A tile 逻辑形状是 [BLOCK_M, BLOCK_K]
  // 实际存成 [BLOCK_K, BLOCK_M]
  //
  // s_a[k][m] = A[m][k]
  __shared__ float s_a[BLOCK_K * BLOCK_M];

  // B 正常存储：
  // s_b[k][n] = B[k][n]
  __shared__ float s_b[BLOCK_K * BLOCK_N];

  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int tid = ty * blockDim.x + tx;

  const int warp_id = tid >> 5;
  const int lane_id = tid & 31;

  const int block_m_start = blockIdx.y * BLOCK_M;
  const int block_n_start = blockIdx.x * BLOCK_N;

  // ------------------------------------------------------------
  // 1. 每个 thread 在 global -> shared 阶段负责搬 4 个 A 元素和 4 个 B 元素
  // ------------------------------------------------------------
  //
  // A tile 一共 128 * 8 = 1024 个 float
  // 256 个 thread，每个 thread 搬 4 个，刚好搬完
  //
  // rowA: 当前 thread 搬运的 A tile 行号，范围 0 ~ 127
  // colA: 当前 thread 搬运的 A tile 列号，只有 0 或 4
  const int rowA = tid >> 1;
  const int colA = (tid & 1) << 2;

  // B tile 一共 8 * 128 = 1024 个 float
  // 256 个 thread，每个 thread 搬 4 个，刚好搬完
  //
  // rowB: 当前 thread 搬运的 B tile 行号，范围 0 ~ 7
  // colB: 当前 thread 搬运的 B tile 列号，范围 0, 4, 8, ..., 124
  const int rowB = tid >> 5;
  const int colB = (tid << 2) & 127;

  // ------------------------------------------------------------
  // 2. 当前 thread 负责计算 C tile 里的哪个 8x8 子块
  // ------------------------------------------------------------
  //
  // 这里不用简单的：
  //   rowC = ty * 8
  //   colC = tx * 8
  //
  // 而是采用 warp 内重新排布的方式，目的是让后续 shared memory load 更友好。
  const int rowC = ((((warp_id >> 1) << 2) + (lane_id & 3)) << 3);

  const int colC = ((((warp_id & 1) << 3) + (lane_id >> 2)) << 3);

  float acc[THREAD_TILE_M * THREAD_TILE_N] = {0.0f};

  // ------------------------------------------------------------
  // 3. 沿 K 维度分块
  // ------------------------------------------------------------
  for (int tile_k = 0; tile_k < K; tile_k += BLOCK_K) {
    // ----------------------------------------------------------
    // 3.1 加载 A tile 到 shared memory
    //     global A: row-major [M, K]
    //     shared A: transposed [BLOCK_K, BLOCK_M]
    // ----------------------------------------------------------
    const int gmem_a_m = block_m_start + rowA;
    const int gmem_a_k = tile_k + colA;

    float4 a4;

    // 只有在完整 4 个 float 都合法，并且 K 是 4 的倍数时，才走 float4 global
    // load。 否则走 scalar fallback，避免越界。
    if (gmem_a_m < M && gmem_a_k + 3 < K && ((K & 3) == 0)) {
      a4 = *reinterpret_cast<const float4*>(A + gmem_a_m * K + gmem_a_k);
    } else {
      a4.x = (gmem_a_m < M && gmem_a_k + 0 < K) ? A[gmem_a_m * K + gmem_a_k + 0]
                                                : 0.0f;
      a4.y = (gmem_a_m < M && gmem_a_k + 1 < K) ? A[gmem_a_m * K + gmem_a_k + 1]
                                                : 0.0f;
      a4.z = (gmem_a_m < M && gmem_a_k + 2 < K) ? A[gmem_a_m * K + gmem_a_k + 2]
                                                : 0.0f;
      a4.w = (gmem_a_m < M && gmem_a_k + 3 < K) ? A[gmem_a_m * K + gmem_a_k + 3]
                                                : 0.0f;
    }

    // 注意这里是转置写入。
    //
    // 原来 A tile 逻辑上是：
    //   A[rowA][colA + 0]
    //   A[rowA][colA + 1]
    //   A[rowA][colA + 2]
    //   A[rowA][colA + 3]
    //
    // 写入 shared memory 后变成：
    //   s_a[colA + 0][rowA]
    //   s_a[colA + 1][rowA]
    //   s_a[colA + 2][rowA]
    //   s_a[colA + 3][rowA]
    s_a[(colA + 0) * BLOCK_M + rowA] = a4.x;
    s_a[(colA + 1) * BLOCK_M + rowA] = a4.y;
    s_a[(colA + 2) * BLOCK_M + rowA] = a4.z;
    s_a[(colA + 3) * BLOCK_M + rowA] = a4.w;

    // ----------------------------------------------------------
    // 3.2 加载 B tile 到 shared memory
    //     global B: row-major [K, N]
    //     shared B: normal [BLOCK_K, BLOCK_N]
    // ----------------------------------------------------------
    const int gmem_b_k = tile_k + rowB;
    const int gmem_b_n = block_n_start + colB;

    float4 b4;

    if (gmem_b_k < K && gmem_b_n + 3 < N && ((N & 3) == 0)) {
      b4 = *reinterpret_cast<const float4*>(B + gmem_b_k * N + gmem_b_n);
    } else {
      b4.x = (gmem_b_k < K && gmem_b_n + 0 < N) ? B[gmem_b_k * N + gmem_b_n + 0]
                                                : 0.0f;
      b4.y = (gmem_b_k < K && gmem_b_n + 1 < N) ? B[gmem_b_k * N + gmem_b_n + 1]
                                                : 0.0f;
      b4.z = (gmem_b_k < K && gmem_b_n + 2 < N) ? B[gmem_b_k * N + gmem_b_n + 2]
                                                : 0.0f;
      b4.w = (gmem_b_k < K && gmem_b_n + 3 < N) ? B[gmem_b_k * N + gmem_b_n + 3]
                                                : 0.0f;
    }

    *reinterpret_cast<float4*>(&s_b[rowB * BLOCK_N + colB]) = b4;

    __syncthreads();

    // ----------------------------------------------------------
    // 3.3 从 shared memory 读取 A/B fragment，计算 8x8 thread tile
    // ----------------------------------------------------------
#pragma unroll
    for (int kk = 0; kk < BLOCK_K; ++kk) {
      float4 regA[2];
      float4 regB[2];

      // A fragment: 当前 thread 需要 8 个 A 元素
      //
      // 因为 s_a 是转置后的 [K, M] 布局，
      // 所以固定 kk 后，rowC ~ rowC + 7 是连续的。
      regA[0] = *reinterpret_cast<float4*>(&s_a[kk * BLOCK_M + rowC]);
      regA[1] = *reinterpret_cast<float4*>(&s_a[kk * BLOCK_M + rowC + 4]);

      // B fragment: 当前 thread 需要 8 个 B 元素
      //
      // s_b 是正常 [K, N] 布局，
      // 固定 kk 后，colC ~ colC + 7 是连续的。
      regB[0] = *reinterpret_cast<float4*>(&s_b[kk * BLOCK_N + colC]);
      regB[1] = *reinterpret_cast<float4*>(&s_b[kk * BLOCK_N + colC + 4]);

      const float a_frag[8] = {regA[0].x, regA[0].y, regA[0].z, regA[0].w,
                               regA[1].x, regA[1].y, regA[1].z, regA[1].w};

      const float b_frag[8] = {regB[0].x, regB[0].y, regB[0].z, regB[0].w,
                               regB[1].x, regB[1].y, regB[1].z, regB[1].w};

#pragma unroll
      for (int i = 0; i < THREAD_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
          acc[i * THREAD_TILE_N + j] += a_frag[i] * b_frag[j];
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
    const int row = block_m_start + rowC + i;

#pragma unroll
    for (int j = 0; j < THREAD_TILE_N; j += 4) {
      const int col = block_n_start + colC + j;

      const float4 out = make_f4(
          acc[i * THREAD_TILE_N + j + 0], acc[i * THREAD_TILE_N + j + 1],
          acc[i * THREAD_TILE_N + j + 2], acc[i * THREAD_TILE_N + j + 3]);

      if (row < M && col + 3 < N && ((N & 3) == 0)) {
        *reinterpret_cast<float4*>(C + row * N + col) = out;
      } else if (row < M) {
        if (col + 0 < N) C[row * N + col + 0] = out.x;
        if (col + 1 < N) C[row * N + col + 1] = out.y;
        if (col + 2 < N) C[row * N + col + 2] = out.z;
        if (col + 3 < N) C[row * N + col + 3] = out.w;
      }
    }
  }
}

}  // namespace

extern "C" void solve(const float* A, const float* B, float* C, int M, int N,
                      int K) {
  dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);

  dim3 blocksPerGrid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

  matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M,
                                                                   N, K);

  cudaDeviceSynchronize();
}