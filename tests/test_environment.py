# environment checks for the training box
import math
import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

device = "cuda"
torch.manual_seed(0)

# test 1: cuda visible and device is 3090
def test_cuda_device():

    name = torch.cuda.get_device_name(0)
    
    assert torch.cuda.is_available()
    assert "3090" in name

# test 2: bf16 supported and matmul close to fp32
def test_bf16_matmul():

    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)

    reference = a @ b
    result = (a.bfloat16() @ b.bfloat16()).float()
    relative_error = (result - reference).norm() / reference.norm()

    assert torch.cuda.is_bf16_supported()
    assert relative_error < 1e-2

# test 3: flash sdpa casual matches manual masked softmax
def test_flash_attention():
    
    batch_size = 2
    num_heads = 4
    sequence_length = 128
    head_size = 64

    shape = (batch_size, num_heads, sequence_length, head_size)
    query = torch.randn(shape, device=device, dtype=torch.bfloat16)
    key = torch.randn(shape, device=device, dtype=torch.bfloat16)
    value = torch.randn(shape, device=device, dtype=torch.bfloat16)
    
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        
    scores = (query.float() @ key.float().transpose(-2, -1)) / math.sqrt(head_size)
    mask = torch.ones(sequence_length, sequence_length, device=device).tril().bool()
    scores = scores.masked_fill(~mask, float("-inf"))
    reference = torch.softmax(scores, dim=-1) @ value.float()

    assert torch.allclose(result.float(), reference, atol=2e-2)

# test 4: torch.compile matches eager
def test_compile():

    def gelu_mlp(x, weight):

        return F.gelu(x @ weight)
    
    x = torch.randn(256, 256, device=device)
    weight = torch.randn(256, 256, device=device)
    compiled = torch.compile(gelu_mlp)

    assert torch.allclose(compiled(x, weight), gelu_mlp(x, weight), atol=1e-4)