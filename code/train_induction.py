import os
import time

os.chdir(os.path.expanduser("~/mlab2/"))
import sys
import pandas as pd
import uuid
from typing import Optional, Tuple
import interp.tools.optional as op
import numpy as np
import rust_circuit as rc
import torch
from interp.circuit.causal_scrubbing.experiment import (
    Experiment,
    ExperimentCheck,
    ExperimentEvalSettings,
    ScrubbedExperiment,
)
import warnings
from interp.circuit.causal_scrubbing.hypothesis import (
    Correspondence,
    CondSampler,
    ExactSampler,
    FuncSampler,
    InterpNode,
    UncondSampler,
    chain_excluding,
    corr_root_matcher,
)
import einops
import pandas as pd
from interp.circuit.interop_rust.algebric_rewrite import (
    residual_rewrite,
    split_to_concat,
)
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.indexer import TORCH_INDEXER as I
from torch.nn.functional import binary_cross_entropy_with_logits
import torch.nn.functional as F
import wandb
import datetime

import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore
    IPython.get_ipython().run_line_magic("autoreload", "2")  # type: ignore
from copy import deepcopy
from typing import List
from tqdm import tqdm
from interp.tools.data_loading import get_val_seqs
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import make_arr
import rust_circuit as rc
import torch
from interp.circuit.interop_rust.model_rewrites import To, configure_transformer
from interp.circuit.interop_rust.module_library import load_model_id
from interp.tools.data_loading import get_val_seqs
from interp.tools.indexer import SLICER as S
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR
from torch.testing import assert_close
import random
import pickle
from functools import partial
from typing import Dict, List

import IPython
import numpy as np
import plotly
import torch
import torch.nn.functional as F
import transformer_lens.utils as utils
import wandb
from transformer_lens.HookedTransformer import HookedTransformer
from transformer_lens.ioi_dataset import IOIDataset

from classifiers import NERModel, POSModel, UDModel
from subnetwork_datasets import (
    build_vocab,
    evaluate,
    load_conllu,
    load_ner,
    masked_loss,
    sent_avgs,
)
from util import from_numpy, partial_state_dict
N = 100
DEVICE = "cuda:0"
SEQ_LEN = 300
NUM_EXAMPLES = 40
MODEL_ID = "attention_only_2"
PRINT_CIRCUITS = True
ACTUALLY_RUN = True
SLOW_EXPERIMENTS = True
DEFAULT_CHECKS: ExperimentCheck = True
EVAL_DEVICE = "cuda:0"
MAX_MEMORY = 20000000000
# BATCH_SIZE = 2000
USING_WANDB = True
MONOTONE_METRIC = "maximize"
START_TIME = datetime.datetime.now().strftime("%a-%d%b_%H%M%S")
PROJECT_NAME = f"acdc_induction_again"

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


def negative_log_probs(dataset: Dataset, logits: torch.Tensor,) -> float:
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

def get_induction_dataset() -> torch.Tensor:
    N = 100
    DEVICE = "cuda:0"
    SEQ_LEN = 300
    NUM_EXAMPLES = 40
    MODEL_ID = "attention_only_2"
    PRINT_CIRCUITS = True
    ACTUALLY_RUN = True
    SLOW_EXPERIMENTS = True
    DEFAULT_CHECKS: ExperimentCheck = True
    EVAL_DEVICE = "cuda:0"
    MAX_MEMORY = 20000000000
    # BATCH_SIZE = 2000
    USING_WANDB = True
    MONOTONE_METRIC = "maximize"
    START_TIME = datetime.datetime.now().strftime("%a-%d%b_%H%M%S")
    PROJECT_NAME = f"acdc_induction_again"
    # longer seq len is better, but short makes stuff a bit easier...
    n_files = 1
    reload_dataset = False
    toks_int_values: rc.Array

    (circ_dict, tokenizer, model_info) = load_model_id(MODEL_ID)
    if reload_dataset:
        dataset_toks = torch.tensor(
            get_val_seqs(n_files=n_files, files_start=0, max_size=SEQ_LEN + 1)
        ).cuda()
        NUM_EXAMPLES, _ = dataset_toks.shape
        toks_int_values = rc.Array(dataset_toks.float(), name="toks_int_vals")
        print(f'new dataset "{toks_int_values.repr()}"')
    else:
        P = rc.Parser()
        toks_int_values_raw = P(
            f"'toks_int_vals' [104091,301] Array 3f36c4ca661798003df14994"
        ).cast_array()

    CACHE_DIR = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
    good_induction_candidate = torch.load(
        f"{CACHE_DIR}/induction_candidates_2022-10-15 04:48:29.970735.pt"
    ).to(device=DEVICE, dtype=torch.float32)
    assert (
        toks_int_values_raw.shape[0] >= SEQ_LEN
    ), f"toks_int_values_raw.shape[0] = {toks_int_values_raw.shape[0]} < {SEQ_LEN} - you could try increasing `n_files`"
    assert (
        toks_int_values_raw.shape[1] >= SEQ_LEN + 1
    ), f"toks_int_values_raw.shape[1] = {toks_int_values_raw.shape[1]} < {SEQ_LEN + 1}"

    tokens_device_dtype = rc.TorchDeviceDtype(device="cuda", dtype="int64")

    toks_int_values = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, :SEQ_LEN],
        name="toks_int_vals",
        device_dtype=tokens_device_dtype,
    )
    mask_reshaped = mask_repeat_candidates[
        :NUM_EXAMPLES, :SEQ_LEN
    ]  # only used for loss
    denom = mask_reshaped.int().sum().item()
    print("We're going to study", denom, "examples...")
    assert denom == 172, (denom, "was not expected")

    toks_int_labels = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, 1 : SEQ_LEN + 1],
        name="toks_int_labels",
        device_dtype=tokens_device_dtype,
    )

    def shuffle_tensor(tens):
        """Shuffle tensor along first dimension"""
        return tens[torch.randperm(tens.shape[0])]

    toks_int_values_other = make_arr(
        shuffle_tensor(toks_int_values.value[:NUM_EXAMPLES, : SEQ_LEN + 1]),
        name="toks_int_vals_other",
        device_dtype=tokens_device_dtype,
    )

    toks = tokenizer.batch_decode(
        good_induction_candidate.nonzero().flatten().view(-1, 1)
    )
    maxlen_tok = max((len(tok), tok) for tok in toks)

    circ_dict = {
        s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device="cuda"))
        for s, c in circ_dict.items()
    }

    orig_circuit = circ_dict["t.bind_w"]
    tok_embeds = circ_dict["t.w.tok_embeds"]
    pos_embeds = circ_dict["t.w.pos_embeds"]

    default_input = toks_int_values.rename("tokens")
    default_output = toks_int_labels.rename("labels")

    print("\ntokens of input and output")
    print(tokenizer.batch_decode(default_input.evaluate()[0, :10]))
    print(tokenizer.batch_decode(default_output.evaluate()[0, :10]))

    patch_input = toks_int_values_other.rename("tokens")
    # make_arr(
    #     toks_int_values_other[:NUM_EXAMPLES, :SEQ_LEN],
    #     "tokens",
    #     device_dtype=tokens_device_dtype,
    # )
    patch_output = default_output  # oo cares..

    default_ds = Dataset({"tokens": default_input, "labels": default_output})
    patch_ds = Dataset({"tokens": patch_input, "labels": patch_output})
    return default_ds.tokens.evaluate(), default_ds, 

def train_induction(
    induction_model, mask_lr=0.01, epochs=100, verbose=True, lambda_reg=100,
):
    wandb.init(
        project="subnetwork-probing",
        entity="acdcremix",
        config={"epochs": epochs, "mask_lr": mask_lr, "lambda_reg": lambda_reg},
    )
    train_data_tensor, dataset = get_induction_dataset()

    # one parameter per thing that is masked
    mask_params = [
        p for n, p in induction_model.named_parameters() if "mask_scores" in n and p.requires_grad
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
    from tqdm import tqdm

    for epoch in tqdm(range(epochs)):  # tqdm.notebook.tqdm(range(epochs)):
        induction_model = do_random_resample_caching(induction_model, train_data_tensor)
        induction_model.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        logit_diff_term = negative_log_probs(
            dataset, induction_model(train_data_tensor)
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
    # %% [markdown]

    """
    Edited version of the hierarchy tutorial that runs a number of threshold runs, and then uses the results to plot a pareto frontier.
    """

    # %%


    MAIN = __name__ == "__main__"

    # %%

    try:
        with open(
            os.path.expanduser("~/induction/data/masks/mask_repeat_candidates.pkl"),
            "rb",
        ) as f:
            mask_repeat_candidates = pickle.load(f)
            # t[1] has 132, 136, 176 available...
            # if induction is AB...AB, B tokens are on 133(OK...), 137, 177
            # OK so this is where we punish losses
    except:
        raise Exception(
            "Have you cloned https://github.com/aryamanarora/induction ??? It is where all the masks are kept !!!"
        )


    # %%

    (circ_dict, tokenizer, model_info) = load_model_id(MODEL_ID)
    # %%

    """
    Get toks and data
    """

    # longer seq len is better, but short makes stuff a bit easier...
    n_files = 1
    reload_dataset = False
    toks_int_values: rc.Array

    if reload_dataset:
        dataset_toks = torch.tensor(
            get_val_seqs(n_files=n_files, files_start=0, max_size=SEQ_LEN + 1)
        ).cuda()
        NUM_EXAMPLES, _ = dataset_toks.shape
        toks_int_values = rc.Array(dataset_toks.float(), name="toks_int_vals")
        print(f'new dataset "{toks_int_values.repr()}"')
    else:
        P = rc.Parser()
        toks_int_values_raw = P(
            f"'toks_int_vals' [104091,301] Array 3f36c4ca661798003df14994"
        ).cast_array()

    CACHE_DIR = f"{RRFS_DIR}/ryan/induction_scrub/cached_vals"
    good_induction_candidate = torch.load(
        f"{CACHE_DIR}/induction_candidates_2022-10-15 04:48:29.970735.pt"
    ).to(device=DEVICE, dtype=torch.float32)
    assert (
        toks_int_values_raw.shape[0] >= SEQ_LEN
    ), f"toks_int_values_raw.shape[0] = {toks_int_values_raw.shape[0]} < {SEQ_LEN} - you could try increasing `n_files`"
    assert (
        toks_int_values_raw.shape[1] >= SEQ_LEN + 1
    ), f"toks_int_values_raw.shape[1] = {toks_int_values_raw.shape[1]} < {SEQ_LEN + 1}"

    tokens_device_dtype = rc.TorchDeviceDtype(device="cuda", dtype="int64")

    toks_int_values = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, :SEQ_LEN],
        name="toks_int_vals",
        device_dtype=tokens_device_dtype,
    )
    mask_reshaped = mask_repeat_candidates[
        :NUM_EXAMPLES, :SEQ_LEN
    ]  # only used for loss
    denom = mask_reshaped.int().sum().item()
    print("We're going to study", denom, "examples...")
    assert denom == 172, (denom, "was not expected")

    toks_int_labels = make_arr(
        toks_int_values_raw.value[:NUM_EXAMPLES, 1 : SEQ_LEN + 1],
        name="toks_int_labels",
        device_dtype=tokens_device_dtype,
    )

    def shuffle_tensor(tens):
        """Shuffle tensor along first dimension"""
        return tens[torch.randperm(tens.shape[0])]

    toks_int_values_other = make_arr(
        shuffle_tensor(toks_int_values.value[:NUM_EXAMPLES, : SEQ_LEN + 1]),
        name="toks_int_vals_other",
        device_dtype=tokens_device_dtype,
    )

    toks = tokenizer.batch_decode(
        good_induction_candidate.nonzero().flatten().view(-1, 1)
    )
    maxlen_tok = max((len(tok), tok) for tok in toks)

    circ_dict = {
        s: rc.cast_circuit(c, rc.TorchDeviceDtypeOp(device="cuda"))
        for s, c in circ_dict.items()
    }

    orig_circuit = circ_dict["t.bind_w"]
    tok_embeds = circ_dict["t.w.tok_embeds"]
    pos_embeds = circ_dict["t.w.pos_embeds"]

    default_input = toks_int_values.rename("tokens")
    default_output = toks_int_labels.rename("labels")

    print("\ntokens of input and output")
    print(tokenizer.batch_decode(default_input.evaluate()[0, :10]))
    print(tokenizer.batch_decode(default_output.evaluate()[0, :10]))

    patch_input = toks_int_values_other.rename("tokens")
    # make_arr(
    #     toks_int_values_other[:NUM_EXAMPLES, :SEQ_LEN],
    #     "tokens",
    #     device_dtype=tokens_device_dtype,
    # )
    patch_output = default_output  # oo cares..

    default_ds = Dataset({"tokens": default_input, "labels": default_output})
    patch_ds = Dataset({"tokens": patch_input, "labels": patch_output})

    #%%

    # from transformer_lens import HookedTransformer
    # et_model = HookedTransformer.from_pretrained("gpt2-medium")

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer("Hello world")["input_ids"]
    [15496, 995]

    # %%

    """
    Create metric
    As mentioned in the AF post, the UNSCRUBBED output = 0.179,
    Induction heads scrubbed = 0.24
    """


    # %%

    """
    Create model
    """

    assert (
        model_info.causal_mask
    ), "Should not apply causal mask if the transformer doesn't expect it!"

    # TODO: may be an issue later!
    causal_mask = rc.Array(
        (torch.arange(SEQ_LEN)[:, None] >= torch.arange(SEQ_LEN)[None, :]).to(
            tok_embeds.cast_array().value
        ),
        f"t.a.c.causal_mask",
    )
    assert model_info.pos_enc_type == "shortformer"
    pos_embeds = pos_embeds.index(I[:SEQ_LEN], name="t.w.pos_embeds_idxed")

    tokens_arr = rc.cast_circuit(
        rc.Array(torch.zeros(SEQ_LEN).to(torch.long), name="tokens"),
        device_dtype=tokens_device_dtype.op(),
    )
    idxed_embeds = rc.GeneralFunction.gen_index(
        tok_embeds, tokens_arr, index_dim=0, name="idxed_embeds"
    )

    # CHECK
    model = rc.module_new_bind(
        orig_circuit,
        ("t.input", idxed_embeds),
        ("a.mask", causal_mask),
        ("a.pos_input", pos_embeds),
        name="t.call",
    )

    # model = model_info.bind_to_input(
    #     orig_circuit,
    #     idxed_embeds,
    #     pos_embeds,
    #     causal_mask,
    # )

    # CHECK
    model = model.update(
        "t.bind_w",
        lambda c: configure_transformer(
            c,
            To.ATTN_HEAD_MLP_NORM,
            split_by_head_config="full",
            use_pull_up_head_split=True,
            use_flatten_res=True,
            flatten_components=True,
        ),
    )
    model = model.cast_module().substitute()
    model.print_html()
    model = rc.Index(model, I[0]).rename("model")
    model = rc.conform_all_modules(model)
    model = model.update("t.call", lambda c: c.rename("logits"))
    model = model.update("t.call", lambda c: c.rename("logits_with_bias"))
    model = model.update(
        rc.Regex("[am]\\d(.h\\d)?$"), lambda c: c.rename(c.name + ".inner")
    )
    model = model.update("t.inp_tok_pos", lambda c: c.rename("embeds"))
    model = model.update("t.a.mask", lambda c: c.rename("padding_mask"))
    for l in range(model_info.params.num_layers):
        for h in range(8):
            model = model.update(f"b{l}.a.h{h}", lambda c: c.rename(f"a{l}.h{h}"))
        next = "final" if l == model_info.params.num_layers - 1 else f"a{l + 1}"
        model = model.update(f"b{l}", lambda c: c.rename(f"{next}.input"))

    def create_path_matcher(
        start_node: rc.MatcherIn, path: list[str], max_distance=10
    ) -> rc.IterativeMatcher:
        """
        Creates a matcher that matches a path of nodes, given in a list of names, where the maximum distance between each node on the path is max_distance
        """

        initial_matcher = rc.IterativeMatcher(start_node)
        max_dis_path_matcher = lambda name: rc.restrict(
            rc.Matcher(name), end_depth=max_distance
        )
        chain_matcher = initial_matcher.chain(max_dis_path_matcher(path[0]))
        for i in range(1, len(path)):
            chain_matcher = chain_matcher.chain(max_dis_path_matcher(path[i]))
        return chain_matcher

    q_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.q",
    ]
    k_path = [
        "a.comb_v",
        "a.attn_probs",
        "a.attn_scores",
        "a.attn_scores_raw",
        "a.k",
    ]
    v_path = ["a.comb_v", "a.v"]
    qkv_paths = {"q": q_path, "k": k_path, "v": v_path}
    attention_head_name = "a{layer}.h{head}"
    qkv_node_name = "a{layer}.h{head}.{qkv}"
    embed_name = "idxed_embeds"
    root_name = "final.input"
    no_layers = 2
    no_heads = 8
    new_circuit = model
    for l in range(no_layers):
        for h in range(no_heads):
            for qkv in ["q", "k", "v"]:
                qkv_matcher = create_path_matcher(f"a{l}.h{h}", qkv_paths[qkv])
                new_circuit = new_circuit.update(
                    qkv_matcher, lambda c: c.rename(f"a{l}.h{h}.{qkv}")
                )

    printer = rc.PrintHtmlOptions(
        shape_only_when_necessary=False,
        traversal=rc.restrict(
            rc.IterativeMatcher(
                "embeds", "padding_mask", "final.norm", rc.Regex("^[am]\\d(.h\\d)?$")
            ),
            term_if_matches=True,
        ),
    )
    new_circuit = rc.substitute_all_modules(new_circuit)
    new_circuit.get_unique("final.input").print_html()
    new_circuit = new_circuit.get_unique("logits")

    tokens_arr = make_arr(
        torch.arange(300), name="tokens", device_dtype=tokens_device_dtype,
    )

    new_circuit = new_circuit.update(rc.Matcher("tokens"), lambda _: tokens_arr,)

    new_logits = new_circuit.evaluate()
    print(torch.norm(new_logits))

    #%% [markdown]

    """
    Now get on to porting this to TransformerLens
    """

    import transformer_lens
    from transformer_lens.HookedTransformerConfig import HookedTransformerConfig

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

    #%%

    et_model = transformer_lens.HookedTransformer(cfg, is_masked=True)

    # embed.W_E torch.Size([50259, 256]) True
    et_model.embed.W_E.data = new_circuit.get_unique("t.w.tok_embeds").evaluate()

    # pos_embed.W_pos torch.Size([2048, 256]) True
    et_model.pos_embed.W_pos.data = new_circuit.get_unique("t.w.pos_embeds").evaluate()

    # blocks.0.ln1.w torch.Size([256]) True
    et_model.blocks[0].ln1.w.data = new_circuit.get_unique("a0.ln.w.scale").evaluate()

    # blocks.0.ln1.b torch.Size([256]) True
    et_model.blocks[0].ln1.b.data = new_circuit.get_unique("a0.ln.w.bias").evaluate()

    # blocks.0.attn.W_Q torch.Size([8, 256, 32]) True
    et_model.blocks[0].attn.W_Q.data = einops.rearrange(
        new_circuit.get_unique("a0.w.q").evaluate(), "a b c -> a c b"
    )

    # blocks.0.attn.W_K torch.Size([8, 256, 32]) True
    et_model.blocks[0].attn.W_K.data = einops.rearrange(
        new_circuit.get_unique("a0.w.k").evaluate(), "a b c -> a c b"
    )

    # blocks.0.attn.W_V torch.Size([8, 256, 32]) True
    et_model.blocks[0].attn.W_V.data = einops.rearrange(
        new_circuit.get_unique("a0.w.v").evaluate(), "a b c -> a c b"
    )  # .reshape(8, 256, 32)

    # blocks.0.attn.W_O torch.Size([8, 32, 256]) True
    et_model.blocks[0].attn.W_O.data = einops.rearrange(
        new_circuit.get_unique("a0.w.o").evaluate(), "a b c -> a c b"
    )

    # blocks.0.attn.b_Q torch.Size([8, 32]) True
    et_model.blocks[0].attn.b_Q.data *= 0.0

    # blocks.0.attn.b_K torch.Size([8, 32]) True
    et_model.blocks[0].attn.b_K.data *= 0.0

    # blocks.0.attn.b_V torch.Size([8, 32]) True
    et_model.blocks[0].attn.b_V.data *= 0.0

    # blocks.0.attn.b_O torch.Size([256]) True
    et_model.blocks[0].attn.b_O.data *= 0.0

    # blocks.1.ln1.w torch.Size([256]) True
    et_model.blocks[1].ln1.w.data = new_circuit.get_unique("a1.ln.w.scale").evaluate()

    # blocks.1.ln1.b torch.Size([256]) True
    et_model.blocks[1].ln1.b.data = new_circuit.get_unique("a1.ln.w.bias").evaluate()

    # blocks.1.attn.W_Q torch.Size([8, 256, 32]) True
    et_model.blocks[1].attn.W_Q.data = einops.rearrange(
        new_circuit.get_unique("a1.w.q").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.W_K torch.Size([8, 256, 32]) True
    et_model.blocks[1].attn.W_K.data = einops.rearrange(
        new_circuit.get_unique("a1.w.k").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.W_V torch.Size([8, 256, 32]) True
    et_model.blocks[1].attn.W_V.data = einops.rearrange(
        new_circuit.get_unique("a1.w.v").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.W_O torch.Size([8, 32, 256]) True
    et_model.blocks[1].attn.W_O.data = einops.rearrange(
        new_circuit.get_unique("a1.w.o").evaluate(), "a b c -> a c b"
    )

    # blocks.1.attn.b_Q torch.Size([8, 32]) True
    et_model.blocks[1].attn.b_Q.data *= 0.0

    # blocks.1.attn.b_K torch.Size([8, 32]) True
    et_model.blocks[1].attn.b_K.data *= 0.0

    # blocks.1.attn.b_V torch.Size([8, 32]) True
    et_model.blocks[1].attn.b_V.data *= 0.0

    # blocks.1.attn.b_O torch.Size([256]) True
    et_model.blocks[1].attn.b_O.data *= 0.0

    # ln_final.w torch.Size([256]) True
    et_model.ln_final.w.data = new_circuit.get_unique("final.ln.w.scale").evaluate()

    # ln_final.b torch.Size([256]) True
    et_model.ln_final.b.data = new_circuit.get_unique("final.ln.w.bias").evaluate()

    # unembed.W_U torch.Size([256, 50259]) True
    et_model.unembed.W_U.data = new_circuit.get_unique("t.w.unembed").evaluate().T

    # unembed.b_U torch.Size([50259]) True
    et_model.unembed.b_U.data *= 0.0

    #%%

    # cache = {}
    # et_model.cache_all(cache)
    # ans = et_model(torch.arange(300))

    # print(
    #     torch.allclose(ans, new_logits, atol=1e-4, rtol=1e-4),
    #     "<- YES THEY ARE THE SAME!!!",
    # )

    from transformer_lens.HookedTransformer import HookedTransformer

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
            et_model.freeze_weights()
            print("Finding subnetwork...")
            assert task == "IOI"
            log, model, number_of_nodes, logit_diff, nodes_to_mask = train_induction(
                et_model, lambda_reg=a_regulation_param
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
    import pandas as pd
    import plotly.express as px

    try:
        df = pd.DataFrame(
            {
                "x": number_of_nodes_list,
                "y": [i for i in logit_diff_list],
                "regularization_params": regularization_params,
                # "percentage_binary": percentage_binary_list,
            }
        )
        plt = px.scatter(
            df, x="x", y="y", hover_data=["regularization_params"]#, "percentage_binary"]
        )
        plt.update_layout(xaxis_title="Number of Nodes", yaxis_title="Log Probs")
        wandb.log({"number_of_nodes": plt})
        wandb.finish()
    except:
        import pdb; pdb.set_trace()
