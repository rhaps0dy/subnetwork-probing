# import os

# os.chdir("/home/ubuntu/mlab2_https/mlab2/")

import argparse
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple
import collections

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformer_lens.utils as utils
from acdc.acdc_utils import EdgeType, TorchIndex
from acdc.TLACDCCorrespondence import TLACDCCorrespondence
from acdc.TLACDCInterpNode import TLACDCInterpNode
from acdc.induction.utils import get_all_induction_things, get_mask_repeat_candidates
from tqdm import tqdm
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.HookedTransformerConfig import HookedTransformerConfig
from transformer_lens.ioi_dataset import IOIDataset

import wandb


def correspondence_from_mask(model: HookedTransformer, nodes_to_mask: list[TLACDCInterpNode]) -> TLACDCCorrespondence:
    corr = TLACDCCorrespondence.setup_from_model(model)

    # If all of {qkv} is masked, also add its head child
    # to the list of nodes to mask
    head_parents = collections.defaultdict(lambda: 0)
    for node in nodes_to_mask:
        child_name = node.name.replace("_q", "_result").replace("_k", "_result").replace("_v", "_result")
        head_parents[(child_name, node.index)] += 1

    assert all([v <= 3 for v in head_parents.values()])

    for (child_name, child_index), count in head_parents.items():
        if count == 3:
            nodes_to_mask.append(TLACDCInterpNode(child_name, child_index, EdgeType.ADDITION))

    for node in nodes_to_mask:
        # Mark edges where this is child as not present
        rest2 = corr.edges[node.name][node.index]
        for rest3 in rest2.values():
            for edge in rest3.values():
                edge.present = False

        # Mark edges where this is parent as not present
        for rest1 in corr.edges.values():
            for rest2 in rest1.values():
                try:
                    rest2[node.name][node.index].present = False
                except KeyError:
                    pass
    return corr


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
) -> torch.Tensor:
    logprobs = F.log_softmax(logits, dim=-1)

    nll_all = F.nll_loss(
        logprobs.view(-1, logprobs.size(-1)), labels.view(-1), reduction="none"
    ).view_as(mask_reshaped)

    denom = mask_reshaped.int().sum().item()

    out = (nll_all * mask_reshaped).sum() / denom
    return out


def kl_divergence(logits: torch.Tensor, mask_reshaped: torch.Tensor) -> torch.Tensor:
    """Compute KL divergence between base_model_probs and probs, taken from Arthur's ACDC code"""
    # labels = dataset.arrs["labels"].evaluate()
    logprobs = F.log_softmax(logits, dim=-1)

    denom = mask_reshaped.int().sum().item()
    kl_div = F.kl_div(logprobs, BASE_MODEL_LOGPROBS, log_target=True, reduction="none").sum(dim=-1)
    assert kl_div.shape == mask_reshaped.shape, (kl_div.shape, mask_reshaped.shape)
    kl_div = kl_div * mask_reshaped

    return kl_div.sum() / denom


def do_random_resample_caching(
    model: HookedTransformer, train_data: torch.Tensor
) -> torch.Tensor:
    for layer in model.blocks:
        layer.attn.hook_q.is_caching = True
        layer.attn.hook_k.is_caching = True
        layer.attn.hook_v.is_caching = True

    with torch.no_grad():
        outs = model(train_data)

    for layer in model.blocks:
        layer.attn.hook_q.is_caching = False
        layer.attn.hook_k.is_caching = False
        layer.attn.hook_v.is_caching = False

    return outs

def do_zero_caching(model: HookedTransformer) -> None:
    for layer in model.blocks:
        layer.attn.hook_q.cache = None
        layer.attn.hook_k.cache = None
        layer.attn.hook_v.cache = None


def train_induction(
    args,
    induction_model,
    train_data_tensor,
    patch_data_tensor,
    labels_tensor,
    train_candidates_mask,
    test_data_tensor,
    test_patch_data_tensor,
    test_labels_tensor,
    test_candidates_mask,
):
    epochs = args.epochs
    mask_lr = args.lr
    lambda_reg = args.lambda_reg
    verbose = args.verbose

    torch.manual_seed(args.seed)

    wandb.init(
        project="subnetwork-probing",
        entity=args.wandb_entity,
        group=args.wandb_group,
        config=args,
    )
    base_dir = Path(__file__).parent.parent / "data" / "induction"

    target_model = HookedTransformer(induction_model.cfg, is_masked=True).to(torch.device(args.device))

    global BASE_MODEL_LOGPROBS
    print("Reset target: ", args.reset_target)
    with torch.no_grad():
        if not args.reset_target:
            target_model.load_state_dict(induction_model.state_dict())

        BASE_MODEL_LOGPROBS = F.log_softmax(
            do_random_resample_caching(target_model, train_data_tensor), dim=-1)

        BASE_NLL = negative_log_probs(labels_tensor, BASE_MODEL_LOGPROBS, train_candidates_mask)
        assert not BASE_NLL.requires_grad and BASE_NLL.shape == ()
        print("Base NLL: ", BASE_NLL.item())

    print("Reset subject:", args.reset_subject)
    if args.reset_subject:
        reset_state_dict = torch.load(base_dir / "random_model.pt")
        induction_model.load_state_dict(reset_state_dict)
        del reset_state_dict
        induction_model.freeze_weights()

        reset_logits = do_random_resample_caching(induction_model, train_data_tensor)
        induction_model_kl = kl_divergence(reset_logits, train_candidates_mask)
        induction_model_nll = negative_log_probs(labels_tensor, reset_logits, train_candidates_mask)
        print("Reset NLL: ", induction_model_nll.item())
        print("Reset KL: ", induction_model_kl.item())

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

    if args.zero_ablation:
        do_zero_caching(induction_model)
    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        if not args.zero_ablation:
            do_random_resample_caching(induction_model, patch_data_tensor)
        induction_model.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        # logit_diff_term = negative_log_probs(
        #     dataset, induction_model(train_data_tensor), mask_reshaped
        # )
        if args.loss_type == "kl_div":
            logit_diff_term = kl_divergence(
                induction_model(train_data_tensor), train_candidates_mask
            )
        elif args.loss_type == "nll":
            logit_diff_term = negative_log_probs(
                labels_tensor, induction_model(train_data_tensor), train_candidates_mask
            )
        elif args.loss_type == "match_nll":
            nll = negative_log_probs(
                labels_tensor, induction_model(train_data_tensor), train_candidates_mask
            )
            logit_diff_term = torch.abs(nll - BASE_NLL)
        else:
            raise ValueError(f"Unknown loss type {args.loss_type}")

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

        if epoch % 10 == 0:
            number_of_nodes, nodes_to_mask = visualize_mask(induction_model)

    print("Calculating metrics on a test set")
    with torch.no_grad():
        BASE_MODEL_LOGPROBS = F.log_softmax(
            do_random_resample_caching(target_model, test_data_tensor), dim=-1)

        BASE_NLL = negative_log_probs(labels_tensor, BASE_MODEL_LOGPROBS, test_candidates_mask)
        assert not BASE_NLL.requires_grad and BASE_NLL.shape == ()
        print("Base NLL: ", BASE_NLL.item())


        if args.zero_ablation:
            do_zero_caching(induction_model)
        else:
            do_random_resample_caching(induction_model, test_patch_data_tensor)

        logits, mask = induction_model(test_data_tensor), test_candidates_mask
        if args.loss_type == "kl_div":
            test_logit_diff_term = kl_divergence(logits, mask)
        elif args.loss_type == "nll":
            test_logit_diff_term = negative_log_probs(labels_tensor, logits, mask)
        elif args.loss_type == "match_nll":
            nll = negative_log_probs(labels_tensor, logits, mask)
            test_logit_diff_term = torch.abs(nll - BASE_NLL)
        else:
            raise ValueError(f"Unknown loss type {args.loss_type}")

    to_log_dict = dict(
        number_of_nodes=number_of_nodes,
        logit_diff=logit_diff_term,
        nodes_to_mask=nodes_to_mask,
        test_logit_diff=test_logit_diff_term,
    )
    return induction_model, to_log_dict


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
parser.add_argument("--wandb-group", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--loss-type", type=str, default="kl_div", choices=["kl_div", "nll", "match_nll"])
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--lambda-reg", type=float, default=100)
parser.add_argument("--zero-ablation", type=int, required=True)
parser.add_argument("--reset-target", type=int, required=True)
parser.add_argument("--reset-subject", type=int, default=0)
parser.add_argument("--seed", type=int, default=random.randint(0, 2 ** 31 - 1), help="Random seed (default: random)")
parser.add_argument("--num-examples", type=int, default=100)
parser.add_argument("--seq-len", type=int, default=300)



def get_transformer_config():
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
    return cfg

if __name__ == "__main__":
    args = parser.parse_args()
    cfg = get_transformer_config()
    model = HookedTransformer(cfg, is_masked=True)

    _acdc_model, train_data_orig, patch_data_orig, _acdc_metric = get_all_induction_things(
        args.num_examples*2, args.seq_len+1, torch.device(args.device), randomize_data=False
    )
    train_candidates_mask_orig = get_mask_repeat_candidates(
        num_examples=args.num_examples * 2, seq_len=args.seq_len
    )

    # Partition data in train and test
    train_data = train_data_orig[:args.num_examples, :-1].contiguous()
    labels_data = train_data_orig[:args.num_examples, 1:].contiguous()
    patch_data = patch_data_orig[:args.num_examples, :-1].contiguous()
    train_candidates_mask = train_candidates_mask_orig[:args.num_examples, :].contiguous()


    test_data = train_data_orig[args.num_examples:, :-1].contiguous()
    test_labels_data = train_data_orig[args.num_examples:, 1:].contiguous()
    test_patch_data = patch_data_orig[args.num_examples:, :-1].contiguous()
    test_candidates_mask = train_candidates_mask_orig[args.num_examples:, :].contiguous()


    model.load_state_dict(_acdc_model.state_dict(), strict=False)
    model = model.to(args.device)
    # Check that the model's outputs are the same
    # torch.testing.assert_allclose(do_random_resample_caching(model, train_data), _acdc_model(train_data))
    del _acdc_model

    regularization_params = [args.lambda_reg]

    is_masked = True

    model.freeze_weights()
    print("Finding subnetwork...")
    model, to_log_dict = train_induction(
        args=args,
        induction_model=model,
        train_data_tensor=train_data,
        patch_data_tensor=patch_data,
        labels_tensor=labels_data,
        train_candidates_mask=train_candidates_mask,
        test_data_tensor=test_data,
        test_patch_data_tensor=test_patch_data,
        test_labels_tensor=test_labels_data,
        test_candidates_mask=test_candidates_mask,
    )

    corr = correspondence_from_mask(model, to_log_dict["nodes_to_mask"])
    mask_val_dict = get_nodes_mask_dict(model)
    percentage_binary = log_percentage_binary(mask_val_dict)

    # Update dict with some different things
    to_log_dict["nodes_to_mask"] = list(map(str, to_log_dict["nodes_to_mask"]))
    to_log_dict["number_of_edges"] = corr.count_no_edges()
    to_log_dict["percentage_binary"] = percentage_binary

    wandb.log(to_log_dict)
    # sanity_check_with_transformer_lens(mask_val_dict)
    wandb.finish()

    # make sure that regularizer can be optimized DONE
    # make sure logit diff can be optimized DONE
    # make sure that the mask is the right shape HOLD
    # reimplement all the diagnostics that are commented out TODO
    # reimplement sanity  check with transformer lens TODO
    # make sure that the input data makes sense
    # make sure that the model makes correct predictions
