import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
):
    pass


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_ak = K, 1
    stride_bk, stride_bn = N, 1
    stride_cm, stride_cn = N, 1

    grid = (M, N)
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn
    )
