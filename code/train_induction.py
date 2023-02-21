import os

os.chdir(os.path.expanduser("~/mlab2/"))
import IPython
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.indexer import TORCH_INDEXER as I

from induction_utils import get_induction_model, get_induction_dataset

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore
import pickle
from functools import partial
from typing import Dict, List

import IPython
import rust_circuit as rc
import transformer_lens.utils as utils
import wandb
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr
from interp.tools.data_loading import get_val_seqs
from tqdm import tqdm
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.ioi_dataset import IOIDataset


SEQ_LEN = 300
NUM_EXAMPLES = 40


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def visualize_mask(gpt2: HookedTransformer) -> None:
    node_name = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask = []
    for layer_index, layer in enumerate(gpt2.blocks):
        for head in range(12):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                if q_k_v == "q":

                    if layer.attn.hook_q.sample_mask()[layer_index].cpu().item() < 0.5:
                        nodes_to_mask.append(f"{layer_index}.{head}.{q_k_v}")
                if q_k_v == "k":
                    if layer.attn.hook_k.sample_mask()[layer_index].cpu().item() < 0.5:
                        nodes_to_mask.append(f"{layer_index}.{head}.{q_k_v}")
                if q_k_v == "v":
                    if layer.attn.hook_v.sample_mask()[layer_index].cpu().item() < 0.5:
                        nodes_to_mask.append(f"{layer_index}.{head}.{q_k_v}")

    log_plotly_bar_chart(x=node_name, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask


def regularizer(
    gpt2: HookedTransformer,
    gamma: float = -0.1,
    zeta: float = 1.1,
    beta: float = 2 / 3,
) -> torch.Tensor:
    # TODO: globally read hyperparams from config
    # need to also do this in the masked hook point so
    # the hyperparams are the same
    def regularization_term(mask: torch.nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(mask - beta * np.log(-gamma / zeta)).mean()

    mask_scores = [
        regularization_term(p)
        for (n, p) in gpt2.named_parameters()
        if "mask_scores" in n
    ]
    return torch.mean(torch.stack(mask_scores))


def negative_log_probs(
    dataset: Dataset, logits: torch.Tensor, mask_reshaped: torch.Tensor
) -> float:
    """NOTE: this average over all sequence positions, I'm unsure why..."""
    labels = dataset.arrs["labels"].evaluate()
    probs = F.softmax(logits, dim=-1)

    assert probs.min() >= 0.0
    assert probs.max() <= 1.0

    log_probs = probs[
        torch.arange(NUM_EXAMPLES).unsqueeze(-1),
        torch.arange(SEQ_LEN).unsqueeze(0),
        labels,
    ].log()

    assert mask_reshaped.shape == log_probs.shape, (
        mask_reshaped.shape,
        log_probs.shape,
    )
    denom = mask_reshaped.int().sum().item()

    masked_log_probs = log_probs * mask_reshaped.int()
    result = (-1.0 * float(masked_log_probs.sum())) / denom

    print("Result", result, denom)
    return result


def do_random_resample_caching(
    gpt2: HookedTransformer, train_data: torch.Tensor
) -> HookedTransformer:
    for layer in gpt2.blocks:
        layer.attn.hook_q.is_caching = True
        layer.attn.hook_k.is_caching = True
        layer.attn.hook_v.is_caching = True

    _ = gpt2(train_data)

    for layer in gpt2.blocks:
        layer.attn.hook_q.is_caching = False
        layer.attn.hook_k.is_caching = False
        layer.attn.hook_v.is_caching = False

    return gpt2


def train_induction(
    induction_model, mask_lr=0.01, epochs=100, verbose=True, lambda_reg=100,
):
    wandb.init(
        project="subnetwork-probing",
        entity="acdcremix",
        config={"epochs": epochs, "mask_lr": mask_lr, "lambda_reg": lambda_reg},
    )
    train_data_tensor, dataset, _, _, mask_reshaped = get_induction_dataset()

    # one parameter per thing that is masked
    mask_params = [
        p
        for n, p in induction_model.named_parameters()
        if "mask_scores" in n and p.requires_grad
    ]
    # parameters for the probe (we don't use a probe)
    gpt2_params = [
        p
        for n, p in induction_model.named_parameters()
        if "mask_scores" not in n and p.requires_grad
    ]
    assert len(gpt2_params) == 0, ("GPT2 should be empty", gpt2_params)
    trainer = torch.optim.Adam(mask_params, lr=mask_lr)
    log = []

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        induction_model = do_random_resample_caching(induction_model, train_data_tensor)
        induction_model.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        logit_diff_term = negative_log_probs(
            dataset, induction_model(train_data_tensor), mask_reshaped
        )
        regularizer_term = regularizer(induction_model)
        loss = logit_diff_term + lambda_reg * regularizer_term
        loss.backward()

        wandb.log(
            {
                "regularisation_loss": regularizer_term,
                "log_probs_loss": logit_diff_term,
                "total_loss": loss,
            }
        )
        trainer.step()

        log.append({"loss_val": loss.item()})
        if epoch % 10 == 0:
            number_of_nodes, nodes_to_mask = visualize_mask(induction_model)
    # wandb.finish()
    # torch.save(gpt2.state_dict(), "masked_gpt2.pt")
    return log, induction_model, number_of_nodes, logit_diff_term, nodes_to_mask


def sanity_check_with_transformer_lens(mask_dict):
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1)
    train_data = ioi_dataset.toks.long()
    gpt2 = HookedTransformer.from_pretrained(is_masked=False, model_name="gpt2")
    gpt2.freeze_weights()
    logits = gpt2(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

    fwd_hooks = make_forward_hooks(mask_dict)
    logits = gpt2.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
    logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    print("original logit diff", logit_diff)
    print("masked logit diff", logit_diff_masked)


def make_forward_hooks(mask_dict):
    forward_hooks = []
    for layer in range(12):
        for head in range(12):
            for qkv in ["q", "k", "v"]:
                mask_value = mask_dict[f"{layer}.{head}.{qkv}"]

                def head_ablation_hook(
                    value, hook, head_idx, layer_idx, qkv_val, mask_value
                ):
                    value[:, :, head_idx, :] *= mask_value
                    return value

                a_hook = (
                    utils.get_act_name(qkv, int(layer)),
                    partial(
                        head_ablation_hook,
                        head_idx=head,
                        layer_idx=layer,
                        qkv_val=qkv,
                        mask_value=mask_value,
                    ),
                )
                forward_hooks.append(a_hook)
    return forward_hooks


def log_percentage_binary(mask_val_dict: Dict) -> float:
    binary_count = 0
    total_count = 0
    for _, v in mask_val_dict.items():
        total_count += 1
        if v == 0 or v == 1:
            binary_count += 1
    return binary_count / total_count


def get_nodes_mask_dict(gpt2: HookedTransformer):
    mask_value_dict = {}
    for layer_index, layer in enumerate(gpt2.blocks):
        for head_index in range(12):
            for q_k_v in ["q", "k", "v"]:
                # total_nodes += 1
                if q_k_v == "q":
                    mask_value = (
                        layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "k":
                    mask_value = (
                        layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                    )
                if q_k_v == "v":
                    mask_value = (
                        layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                    )
                mask_value_dict[f"{layer_index}.{head_index}.{q_k_v}"] = mask_value
    return mask_value_dict


if __name__ == "__main__":
    model = get_induction_model()
    regularization_params = [
        1e1,
        1e2,
        300,
        500,
        700,
        1e3,
    ]

    is_masked = True
    logit_diff_list = []
    number_of_nodes_list = []
    percentage_binary_list = []

    for a_regulation_param in regularization_params:
        for task in ["IOI"]:
            model.freeze_weights()
            print("Finding subnetwork...")
            assert task == "IOI"
            log, model, number_of_nodes, logit_diff, nodes_to_mask = train_induction(
                model, lambda_reg=a_regulation_param
            )
            print("nodes to mask", nodes_to_mask)
            logit_diff_list.append(logit_diff)
            number_of_nodes_list.append(number_of_nodes)
            # mask_val_dict = get_nodes_mask_dict(model)
            # percentage_binary = log_percentage_binary(mask_val_dict)
            # wandb.log({"percentage_binary": percentage_binary})
            # percentage_binary_list.append(percentage_binary)
            # sanity_check_with_transformer_lens(mask_val_dict)
            wandb.finish()

    wandb.init(project="pareto-subnetwork-probing", entity="acdcremix")
    import plotly.express as px

    df = pd.DataFrame(
        {
            "x": number_of_nodes_list,
            "y": [i for i in logit_diff_list],
            "regularization_params": regularization_params,
            # "percentage_binary": percentage_binary_list,
        }
    )
    plt = px.scatter(
        df, x="x", y="y", hover_data=["regularization_params"]  # , "percentage_binary"]
    )
    plt.update_layout(xaxis_title="Number of Nodes", yaxis_title="Log Probs")
    wandb.log({"number_of_nodes": plt})
    wandb.finish()
