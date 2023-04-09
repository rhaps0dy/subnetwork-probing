# import os

# os.chdir("/home/ubuntu/mlab2_https/mlab2/")

import argparse
from copy import deepcopy
from functools import partial
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformer_lens.utils as utils
import wandb
from tqdm import tqdm
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.ioi_dataset import IOIDataset

from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.acdc_utils import EdgeType, TorchIndex

import networkx as nx

def corr_graph_from_mask(model: HookedTransformer, nodes_to_mask: list[TLACDCInterpNode]) -> nx.DiGraph:
    corr = TLACDCCorrespondence.setup_from_model(model)

    # Build DiGraph from correspondence
    graph = nx.DiGraph()
    for node in corr.nodes():
        graph.add_node((node.name, node.index))
    for cn, ci, pn, pi in corr.all_edges().keys():
        graph.add_edge((pn, pi), (cn, ci))

    # Remove masked nodes
    for node in nodes_to_mask:
        graph.remove_node((node.name, node.index))

    # Remove nodes which don't communicate with the output
    logits_node = (f"blocks.{model.cfg.n_layers-1}.hook_resid_post", TorchIndex([None]))
    assert logits_node in graph.nodes
    for node in list(graph.nodes):
        if not nx.has_path(graph, node, logits_node):
            graph.remove_node(node)
    return graph


SEQ_LEN = 300
NUM_EXAMPLES = 40
NUMBER_OF_HEADS = 8
NUMBER_OF_LAYERS = 2
BASE_MODEL_LOGPROBS = torch.zeros([0])  # zero-size tensor so we error


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


def visualize_mask(model: HookedTransformer) -> tuple[int, list[TLACDCInterpNode]]:
    node_name_list = []
    mask_scores_for_names = []
    total_nodes = 0
    nodes_to_mask: list[TLACDCInterpNode] = []
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(NUMBER_OF_HEADS):
            for q_k_v in ["q", "k", "v"]:
                total_nodes += 1
                if q_k_v == "q":
                    mask_sample = (
                        layer.attn.hook_q.sample_mask()[head_index].cpu().item()
                    )
                elif q_k_v == "k":
                    mask_sample = (
                        layer.attn.hook_k.sample_mask()[head_index].cpu().item()
                    )
                elif q_k_v == "v":
                    mask_sample = (
                        layer.attn.hook_v.sample_mask()[head_index].cpu().item()
                    )
                else:
                    raise ValueError(f"{q_k_v=} must be q, k, or v")

                node_name = f"blocks.{layer_index}.attn.hook_{q_k_v}"
                node_name_with_index = f"{node_name}[{head_index}]"
                node_name_list.append(node_name_with_index)
                node = TLACDCInterpNode(node_name, TorchIndex((None, None, head_index)), EdgeType.ADDITION)

                mask_scores_for_names.append(mask_sample)
                if mask_sample < 0.5:
                    nodes_to_mask.append(node)

    assert len(mask_scores_for_names) == 3 * NUMBER_OF_HEADS * NUMBER_OF_LAYERS
    log_plotly_bar_chart(x=node_name_list, y=mask_scores_for_names)
    node_count = total_nodes - len(nodes_to_mask)
    return node_count, nodes_to_mask


def regularizer(
    model: HookedTransformer,
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
        for (n, p) in model.named_parameters()
        if "mask_scores" in n
    ]
    return torch.mean(torch.stack(mask_scores))


def negative_log_probs(
    labels: torch.Tensor, logits: torch.Tensor, mask_reshaped: torch.Tensor
) -> float:
    """NOTE: this average over all sequence positions, I'm unsure why..."""
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
    result = (-1.0 * (masked_log_probs.sum())) / denom

    print("Result", result, denom)
    return result


def kl_divergence(logits: torch.Tensor, mask_reshaped: torch.Tensor):
    """Compute KL divergence between base_model_probs and probs, taken from Arthur's ACDC code"""
    # labels = dataset.arrs["labels"].evaluate()
    probs = F.log_softmax(logits, dim=-1)

    denom = mask_reshaped.int().sum().item()
    kl_div = F.kl_div(probs, BASE_MODEL_LOGPROBS, log_target=True, reduction="none").sum(dim=-1)
    assert kl_div.shape == mask_reshaped.shape, (kl_div.shape, mask_reshaped.shape)
    kl_div = kl_div * mask_reshaped

    return kl_div.sum() / denom


def do_random_resample_caching(
    model: HookedTransformer, train_data: torch.Tensor
) -> HookedTransformer:
    for layer in model.blocks:
        layer.attn.hook_q.is_caching = True
        layer.attn.hook_k.is_caching = True
        layer.attn.hook_v.is_caching = True

    _ = model(train_data)

    for layer in model.blocks:
        layer.attn.hook_q.is_caching = False
        layer.attn.hook_k.is_caching = False
        layer.attn.hook_v.is_caching = False

    return model


def train_induction(args, induction_model):
    epochs = args.epochs
    mask_lr = args.lr
    lambda_reg = args.lambda_reg
    verbose = args.verbose

    wandb.init(
        project="subnetwork-probing",
        entity=args.wandb_entity,
        config={"epochs": epochs, "mask_lr": mask_lr, "lambda_reg": lambda_reg},
    )
    base_dir = Path(__file__).parent.parent / "data" / "induction"
    train_data_tensor = torch.load(base_dir / "train.pt").to(args.device)
    patch_data_tensor = torch.load(base_dir / "patch.pt").to(args.device)
    mask_reshaped = torch.load(base_dir / "mask_reshaped.pt").to(args.device)


    global BASE_MODEL_LOGPROBS
    with torch.no_grad():
        for layer in induction_model.blocks:
            layer.attn.hook_q.mask_scores[:] = torch.inf
            layer.attn.hook_k.mask_scores[:] = torch.inf
            layer.attn.hook_v.mask_scores[:] = torch.inf

        BASE_MODEL_LOGPROBS = F.log_softmax(induction_model(train_data_tensor), dim=-1)

        for layer in induction_model.blocks:
            layer.attn.hook_q.mask_scores[:] = 1.6094
            layer.attn.hook_k.mask_scores[:] = 1.6094
            layer.attn.hook_v.mask_scores[:] = 1.6094

    # one parameter per thing that is masked
    mask_params = [
        p
        for n, p in induction_model.named_parameters()
        if "mask_scores" in n and p.requires_grad
    ]
    # parameters for the probe (we don't use a probe)
    model_params = [
        p
        for n, p in induction_model.named_parameters()
        if "mask_scores" not in n and p.requires_grad
    ]
    assert len(model_params) == 0, ("MODEL should be empty", model_params)
    trainer = torch.optim.Adam(mask_params, lr=mask_lr)
    log = []

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        induction_model = do_random_resample_caching(induction_model, patch_data_tensor)
        induction_model.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        # logit_diff_term = negative_log_probs(
        #     dataset, induction_model(train_data_tensor), mask_reshaped
        # )
        logit_diff_term = kl_divergence(
            induction_model(train_data_tensor), mask_reshaped
        )
        regularizer_term = regularizer(induction_model)
        loss = logit_diff_term + regularizer_term * lambda_reg
        loss.backward()

        wandb.log(
            {
                "regularisation_loss": regularizer_term,
                "KL_loss": logit_diff_term,
                "total_loss": loss,
            }
        )
        trainer.step()

        log.append({"loss_val": loss.item()})

        if epoch % 10 == 0:
            number_of_nodes, nodes_to_mask = visualize_mask(induction_model)
    # wandb.finish()
    # torch.save(model.state_dict(), "masked_model.pt")
    return log, induction_model, number_of_nodes, logit_diff_term, nodes_to_mask


# check regularizer can set all the
def sanity_check_with_transformer_lens(mask_dict):
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1)
    train_data = ioi_dataset.toks.long()
    model = HookedTransformer.from_pretrained(is_masked=False, model_name="model")
    model.freeze_weights()
    logits = model(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

    fwd_hooks = make_forward_hooks(mask_dict)
    logits = model.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
    logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    print("original logit diff", logit_diff)
    print("masked logit diff", logit_diff_masked)


def make_forward_hooks(mask_dict):
    forward_hooks = []
    for layer in range(NUMBER_OF_LAYERS):
        for head in range(NUMBER_OF_HEADS):
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


def get_nodes_mask_dict(model: HookedTransformer):
    mask_value_dict = {}
    for layer_index, layer in enumerate(model.blocks):
        for head_index in range(NUMBER_OF_HEADS):
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


parser = argparse.ArgumentParser("train_induction")
parser.add_argument("--wandb-entity", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--lambda-reg", type=float, default=100)


if __name__ == "__main__":
    args = parser.parse_args()
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=256,
        n_ctx=2048,  # chekc pos embed size
        n_heads=8,
        d_head=32,
        # model_name : str = "custom"
        # d_mlp: Optional[int] = None
        # act_fn: Optional[str] = None
        d_vocab=50259,
        # eps: float = 1e-5
        use_attn_result=True,
        use_attn_scale=True,  # divide by sqrt(d_head)
        # use_local_attn: bool = False
        # original_architecture: Optional[str] = None
        # from_checkpoint: bool = False
        # checkpoint_index: Optional[int] = None
        # checkpoint_label_type: Optional[str] = None
        # checkpoint_value: Optional[int] = None
        # tokenizer_name: Optional[str] = None
        # window_size: Optional[int] = None
        # attn_types: Optional[List] = None
        # init_mode: str = "gpt2"
        # normalization_type: Optional[str] = "LN"
        # device: Optional[str] = None
        # attention_dir: str = "causal"
        attn_only=True,
        # seed: Optional[int] = None
        # initializer_range: float = -1.0
        # init_weights: bool = True
        # scale_attn_by_inverse_layer_idx: bool = False
        positional_embedding_type="shortformer",
        # final_rms: bool = False
        # d_vocab_out: int = -1
        # parallel_attn_mlp: bool = False
        # rotary_dim: Optional[int] = None
        # n_params: Optional[int] = None
        # use_hook_tokens: bool = False
    )
    model = HookedTransformer(cfg, is_masked=True)
    state_dict = torch.load(Path(__file__).parent.parent / "data" / "induction" / "model.pt")
    model.load_state_dict(state_dict)
    model = model.to(args.device)

    regularization_params = [args.lambda_reg]

    is_masked = True

    model.freeze_weights()
    print("Finding subnetwork...")
    log, model, number_of_nodes, logit_diff, nodes_to_mask = train_induction(args=args, induction_model=model)

    graph = corr_graph_from_mask(model, nodes_to_mask)
    mask_val_dict = get_nodes_mask_dict(model)
    percentage_binary = log_percentage_binary(mask_val_dict)

    print(dict(
        logit_diff=logit_diff,
        number_of_nodes=number_of_nodes,
        nodes_to_mask=nodes_to_mask,
        number_of_edges=len(graph.edges),
        percentage_binary=percentage_binary,
    ))

    # sanity_check_with_transformer_lens(mask_val_dict)
    wandb.finish()

    # make sure that regularizer can be optimized DONE
    # make sure logit diff can be optimized DONE
    # make sure that the mask is the right shape HOLD
    # reimplement all the diagnostics that are commented out TODO
    # reimplement sanity  check with transformer lens TODO
    # make sure that the input data makes sense
    # make sure that the model makes correct predictions
    # brainstorm more
    #
    # wandb.init(project="pareto-subnetwork-probing", entity=args.wandb_entity)
    # import plotly.express as px

    # df = pd.DataFrame(
    #     {
    #         "x": number_of_edges,
    #         "y": [i.detach().cpu().item() for i in logit_diff_list],
    #         "regularization_params": regularization_params,
    #         "percentage_binary": percentage_binary_list,
    #     }
    # )
    # plt = px.scatter(
    #     df, x="x", y="y", hover_data=["regularization_params", "percentage_binary"]
    # )
    # plt.update_layout(xaxis_title="Number of Nodes", yaxis_title="KL")
    # wandb.log({"number_of_nodes": plt})
    # wandb.finish()
