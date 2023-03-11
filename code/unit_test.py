import math

import numpy as np
import pytest
import torch
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.ioi_dataset import IOIDataset


@pytest.fixture
def masked_gpt2():
    return HookedTransformer.from_pretrained(is_masked=True, model_name="gpt2")


@pytest.fixture
def gpt2():
    return HookedTransformer.from_pretrained(is_masked=False, model_name="gpt2")


@pytest.fixture
def ioi_data():
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=100, nb_templates=1)
    train_data = ioi_dataset.toks.long()
    return train_data


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
    param_val = 1.6094379425048828
    assert regularizer_loss.detach().cpu().item() - param_val < 1e-5


def test_regularizer_can_optimize(masked_gpt2):
    from train import regularizer

    initial_loss = regularizer(masked_gpt2, beta=0)
    opt = torch.optim.Adam(masked_gpt2.parameters(), lr=1e-3)
    opt.zero_grad()
    initial_loss.backward()
    opt.step()

    opt.zero_grad()
    next_loss = regularizer(masked_gpt2, beta=0)
    assert next_loss.detach().cpu().numpy() < initial_loss.detach().cpu().numpy()


def test_mask_is_binary(masked_gpt2):
    mask_sample = masked_gpt2.blocks[
        0
    ].attn.hook_v.sample_mask()  # TODO: check all blocks eventually, importantly this fails mask != binary
    for i in range(mask_sample.shape[0]):
        assert mask_sample[i].cpu().item() == 1.0 or mask_sample[i].cpu().item() == 0.0


def test_log_percentage_binary():
    from train import log_percentage_binary

    mask_dict = {
        "0.1.k": 0.0,
        "0.1.v": 0.0,
        "2.1.q": 1.0,
        "1.1.v": 0.0,
        "1.2.v": 0.02304837,
    }
    assert log_percentage_binary(mask_dict) == 4 / 5

    mask_dict = {
        "0.1.k": 0.0,
        "0.1.v": 1,
        "2.1.q": 0.0,
        "1.1.v": 0.0,
        "1.2.v": 0,
    }
    assert log_percentage_binary(mask_dict) == 1

    mask_dict = {
        "0.1.k": 0.111,
        "0.1.v": 0.99999,
        "2.1.q": 0.1,
        "1.1.v": 0.2,
        "1.4.v": 0.3,
    }
    assert log_percentage_binary(mask_dict) == 0


def test_do_random_resample_caching(gpt2, masked_gpt2, ioi_data):
    from train import do_random_resample_caching

    for a_block in masked_gpt2.blocks:
        assert a_block.attn.hook_v.cache is None
        assert a_block.attn.hook_q.cache is None
        assert a_block.attn.hook_k.cache is None

    masked_gpt2 = do_random_resample_caching(masked_gpt2, ioi_data)

    def check_cache_shape(cache: torch.Tensor) -> bool:
        return cache.shape == (
            ioi_data.shape[0],
            ioi_data.shape[1],
            masked_gpt2.cfg.n_heads,
            masked_gpt2.cfg.d_head,
        )

    for a_block in masked_gpt2.blocks:
        assert check_cache_shape(a_block.attn.hook_v.cache)
        assert check_cache_shape(a_block.attn.hook_q.cache)
        assert check_cache_shape(a_block.attn.hook_k.cache)

    def cache_representations_equal(
        our_cache: torch.Tensor, transformer_lens_cache: torch.Tensor
    ) -> bool:
        assert our_cache.shape == transformer_lens_cache.shape
        random_representation = our_cache[np.random.choice(our_cache.shape[0])]
        mse_list = []
        for _, representation in enumerate(transformer_lens_cache):
            mse_list.append(
                torch.nn.functional.mse_loss(random_representation, representation)
            )
            if torch.equal(random_representation, representation.detach()):
                return True
            else:

                print("Not equal", random_representation, representation)

        import ipdb

        ipdb.set_trace()
        return False

    _, gpt2_cache = gpt2.run_with_cache(ioi_data)

    example_cache_location = "blocks.8.attn.hook_q"
    transformer_lens_cache = gpt2_cache[example_cache_location].cpu().detach()
    assert cache_representations_equal(
        transformer_lens_cache, masked_gpt2.blocks[8].attn.hook_q.cache
    )

    example_cache_location = "blocks.2.attn.hook_v"
    transformer_lens_cache = gpt2_cache[example_cache_location].cpu().detach()
    assert cache_representations_equal(
        transformer_lens_cache, masked_gpt2.blocks[2].attn.hook_v.cache
    )

    example_cache_location = "blocks.10.attn.hook_k"
    transformer_lens_cache = gpt2_cache[example_cache_location].cpu().detach()
    assert cache_representations_equal(
        transformer_lens_cache, masked_gpt2.blocks[10].attn.hook_k.cache
    )

def test_

# def sanity_check_with_transformer_lens(nodes_to_mask, gpt2):
#     from train import logit_diff_from_ioi_dataset, make_forward_hooks, train_ioi

#     gpt2.freeze_weights()
#     print("Finding subnetwork...")
#     log, gpt2, number_of_nodes, logit_diff, nodes_to_mask = train_ioi(
#         gpt2, lambda_reg=1
#     )

#     ioi_dataset = IOIDataset(prompt_type="ABBA", N=100, nb_templates=1,)
#     train_data = ioi_dataset.toks.long()
#     gpt2 = HookedTransformer.from_pretrained(is_masked=False, model_name="gpt2")
#     gpt2.freeze_weights()
#     logits = gpt2(train_data)
#     logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

#     fwd_hooks = make_forward_hooks(nodes_to_mask)
#     logits = gpt2.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
#     logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
#     assert logit_diff_masked > logit_diff
