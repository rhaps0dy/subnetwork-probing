import pytest
from transformer_lens.HookedTransformer import MaskedHookedTransformer
import torch
import math


@pytest.fixture
def gpt2():
    return MaskedHookedTransformer.from_pretrained("gpt2")


def test_regularizer_computation(gpt2):
    from train import regularizer

    regularizer_loss = regularizer(gpt2, beta=0)
    assert regularizer_loss.detach().cpu().item() - (1 / (1 + math.exp(-1.0))) < 1e-5


def test_regularizer_can_optimize(gpt2):
    from train import regularizer

    initial_loss = regularizer(gpt2, beta=0)
    opt = torch.optim.Adam(gpt2.parameters(), lr=1e-3)
    opt.zero_grad()
    initial_loss.backward()
    opt.step()

    opt.zero_grad()
    next_loss = regularizer(gpt2, beta=0)
    assert next_loss.detach().cpu().numpy() < initial_loss.detach().cpu().numpy()
