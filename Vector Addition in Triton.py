import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # we need get the program id of an instance before running the programm
    # else we will be running the program on the same elements
    # axis=0 stands for dimension of the program to get ID from
    pid = tl.program_id(axis=0)

    # here we calculate the offset
    # offset means how far are we from the beginning of the memory
    # tl.arange creates a sequence of integers from start to end - 1. in this case from 0 to 
    # BLOCK_SIZE - 1
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # list to check which elements (by offset) are within the bounds.
    # e.g. if n = 6, then only the first 6 offsets are True, others are False
    mask = offsets < n

    # loading data from HBM into GPU registers
    # _ptr means the pointer of the first memory address of a tensor
    # x_ptr + offsets basically means the following: 
    # x_ptr: 0x1000, offsets[0, 1, 2, 3] -> result [0x1000, 0x1004, 0x1009, 0x100C]
    # it is a standard way to load a tensor into memory
    # we use mask=mask here to prevent accessing memory outside of tensor's bounds 
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Loading data back from GPU registers to HBM, already with vector addition 
    tl.store(out_ptr + offsets, x + y, mask=mask)


def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    # number of elements
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)