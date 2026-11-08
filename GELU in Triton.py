import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    # so the overall idea of GELU is : 
    # GELU(x) = x * P(X<=x), where the second term is the cumulative distribution function of normal distribution
    # it is equal to x * Phi(x)
    tl.store(out_ptr + offsets, 1/2 * x * (1 + tl.erf(x/(2**(1/2)))), mask=mask)



def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gelu_kernel: out = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    n = x.numel()
    #tl.erf(x)
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)