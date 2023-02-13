import pytest
from transformer_lens.HookedTransformer import (
    MaskedHookedTransformer,
    HookedTransformer,
)
import torch
import math


@pytest.fixture
def masked_gpt2():
    return MaskedHookedTransformer.from_pretrained("gpt2")


@pytest.fixture
def gpt2():
    return HookedTransformer.from_pretrained("gpt2")


def test_ioi_logit_diff(gpt2):
    from transformer_lens.ioi_dataset import IOIDataset
    from train import logit_diff_from_ioi_dataset

    ioi_dataset = IOIDataset(prompt_type="ABBA", N=100, nb_templates=1,)
    train_data = ioi_dataset.toks.long()
    logits = gpt2(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    assert logit_diff.detach().cpu().item() > 0.0


def test_regularizer_computation(masked_gpt2):
    from train import regularizer

    regularizer_loss = regularizer(masked_gpt2, beta=0)
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
