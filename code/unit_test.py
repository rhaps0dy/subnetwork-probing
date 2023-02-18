import pytest
from transformer_lens.HookedTransformer import HookedTransformer
import torch
import math
from transformer_lens.ioi_dataset import IOIDataset


@pytest.fixture
def masked_gpt2():
    return HookedTransformer.from_pretrained(is_masked=True, model_name="gpt2")


@pytest.fixture
def gpt2():
    return HookedTransformer.from_pretrained(is_masked=False, model_name="gpt2")


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
