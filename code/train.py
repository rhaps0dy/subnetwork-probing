import IPython
import torch
import torch.nn.functional as F
import numpy as np
<<<<<<< Updated upstream
from transformer_lens.HookedTransformer import HookedTransformer
=======

>>>>>>> Stashed changes
from util import from_numpy, partial_state_dict
from classifiers import POSModel, NERModel, UDModel
from subnetwork_datasets import (
    load_conllu,
    build_vocab,
    sent_avgs,
    masked_loss,
    evaluate,
    load_ner,
)
<<<<<<< Updated upstream
from transformer_lens.ioi_dataset import IOIDataset
import wandb
import plotly
from typing import List
import transformer_lens.utils as utils

N = 100


def log_plotly_bar_chart(x: List[str], y: List[float]) -> None:
    import plotly.graph_objects as go

    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    wandb.log({"mask_scores": fig})


# TODO: make sure this is good
# TODO: implement a sample mask binary function to remove this as a possible issue -> confirmed its not this
# TODO: maybe if lots of things are turned off, then slight non-binary things can change logit diff a lot? -> confirmed its not this
# TODO: make sure that the weights are not changing -> confirmed its not this
# TODO: is it using layernorm? make a new model and attach mask parameters to it everytime
# TODO: MLPs are the same


def visualize_mask(gpt2: HookedTransformer) -> None:
    # node_name = []
    # mask_scores_for_names = []
    # total_nodes = 0
    # nodes_to_mask = []
    mask_value_dict = {}
    for layer_index, layer in enumerate(gpt2.blocks):
        for head in range(12):
            for q_k_v in ["q", "k", "v"]:
                # total_nodes += 1
                if q_k_v == "q":
                    mask_value = (
                        layer.attn.hook_q.sample_mask()[layer_index].cpu().item()
                    )
                if q_k_v == "k":
                    mask_value = (
                        layer.attn.hook_k.sample_mask()[layer_index].cpu().item()
                    )
                if q_k_v == "v":
                    mask_value = (
                        layer.attn.hook_v.sample_mask()[layer_index].cpu().item()
                    )
                mask_value_dict[f"{layer_index}.{head}.{q_k_v}"] = mask_value

    # log_plotly_bar_chart(x=node_name, y=mask_scores_for_names)
    # node_count = total_nodes - len(nodes_to_mask)
    return mask_value_dict  # node_count, nodes_to_mask


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


def logit_diff_from_ioi_dataset(
    logits: torch.Tensor, tokens: torch.Tensor, mean=False,
):
    assert tokens.shape == (
        N,
        16,
    ), tokens.shape  # TODO check this is not breaking things...
    assert len(logits.shape) == 3, logits.shape

    io_labels = tokens[:, 2]
    s_labels = tokens[:, 4]

    io_logits = logits[torch.arange(N), -2, io_labels]
    s_logits = logits[torch.arange(N), -2, s_labels]

    logit_diff = io_logits - s_logits
    if mean:
        return logit_diff.mean()
    else:
        return logit_diff


def train_ioi(
    gpt2, mask_lr=0.01, epochs=10000, verbose=True, lambda_reg=100,
=======


def train_ioi(
    gpt2,
    train_path,
    dev_path,
    lambda_init=1000,
    lambda_final=10000,
    lr_base=3e-5,
    mask_lr_base=0.1,
    lr_warmup_frac=0.1,
    epochs=3,
    batch_size=32,
    verbose=True,
    lambda_startup_frac=0.25,
    lambda_warmup_frac=0.5,
    subbatch_size=8,
    masked=False,
>>>>>>> Stashed changes
):
    def calc_dev():
        gpt2.eval()
        all_preds = np.array([])
        all_labels = pack_labels([exmp["labels"] for exmp in dev_data])
        for j in range(0, len(dev_data), batch_size):
            exmps = dev_data[j : j + batch_size]
            sents = [exmp["sent"] for exmp in exmps]
            with torch.no_grad():
                pred = gpt2.predict_batch(sents).cpu().numpy()  # numsent x 3
            pred = np.argmax(pred, axis=1)
            all_preds = np.concatenate((all_preds, pred))
        return calc_f1(all_preds, all_labels)

    def pack_labels(labels):
        return np.array(sum(labels, []))

    def calc_f1(preds, labels):
        assert len(preds) == len(labels)
        preds = [gpt2.i2tag[pred] for pred in preds]
        labels = [gpt2.i2tag[label] for label in labels]
        result = evaluate(labels, preds)
        return result[2] / 100

    def converged(log, k=10, thresh=0.01, min_epochs=25):
        if len(log) < min_epochs:
            return False
        accs = [l["dev_acc"] for l in log[-k:]]
        accs_converged = (np.max(accs) - np.min(accs)) < thresh
        if masked:
            sparsity = [l["reg_val"] for l in log[-k:]]
            sparsity_converged = (np.max(sparsity) - np.min(sparsity)) < thresh
            return accs_converged and sparsity_converged
        return accs_converged

    train_data = load_ner(train_path, gpt2.tag2i)
    dev_data = load_ner(dev_path, gpt2.tag2i)

    print(
        "lambda_init: {}, lambda_final: {}, lambda_startup_frac: {}, lambda_warmup_frac: {}".format(
            lambda_init, lambda_final, lambda_startup_frac, lambda_warmup_frac
        )
    )
    print(
        "lr_base: {}, mask_lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}".format(
            lr_base, mask_lr_base, lr_warmup_frac, epochs, batch_size
        )
    )
<<<<<<< Updated upstream
    # blackbox this bit
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1,)
    train_data = ioi_dataset.toks.long()
=======
>>>>>>> Stashed changes

    # group parameters for different learning rates
    mask_params = [
        p for n, p in gpt2.named_parameters() if "mask_score" in n and p.requires_grad
    ]
    gpt2_params = [
        p
        for n, p in gpt2.named_parameters()
        if "mask_score" not in n and p.requires_grad
    ]

<<<<<<< Updated upstream
        log.append({"loss_val": loss.item()})
        if epoch % 10 == 0:
            # number_of_nodes, nodes_to_mask = visualize_mask(gpt2)
            mask_value_dict = visualize_mask(gpt2)
    wandb.finish()
    return log, gpt2, logit_diff_term, mask_value_dict


def sanity_check_with_transformer_lens(nodes_to_mask):
    import ipdb

    ipdb.set_trace()
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1)
    train_data = ioi_dataset.toks.long()
    gpt2 = HookedTransformer.from_pretrained(is_masked=False, model_name="gpt2")
    gpt2.freeze_weights()
    logits = gpt2(train_data)
    logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)

    fwd_hooks = make_forward_hooks(nodes_to_mask)
    logits = gpt2.run_with_hooks(train_data, return_type="logits", fwd_hooks=fwd_hooks)
    logit_diff_masked = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
    print("original logit diff", logit_diff)
    print("masked logit diff", logit_diff_masked)


def make_forward_hooks(nodes_to_mask):
    forward_hooks = []
    for node in nodes_to_mask:
        layer = int(node.split(".")[0])
        head = int(node.split(".")[1])
        qkv = node.split(".")[2]

        def head_ablation_hook(value, hook):
            print(f"Shape of the value tensor: {value.shape}")
            value[:, :, layer, :] = 0.0
            return value

        a_hook = (utils.get_act_name(qkv, int(head)), head_ablation_hook)
        forward_hooks.append(a_hook)
    return forward_hooks


if __name__ == "__main__":
    from transformer_lens.HookedTransformer import (
        HookedTransformer,
        # MaskedHookedTransformer,
    )

    regularization_params = [
        # 1e4,
        # 1e3,
        1e2,
        # 1e1,
        # 1e0,
        # 0.1,
        # 0.01,
        # 0.001,
    ]
    is_masked = True
    logit_diff_list = []
    number_of_nodes_list = []

    for a_regulation_param in regularization_params:
        for task in ["IOI"]:
            gpt2 = HookedTransformer.from_pretrained(
                is_masked=is_masked, model_name="gpt2"
            )
            gpt2.freeze_weights()
            print("Finding subnetwork...")
            assert task == "IOI"
            log, gpt2, logit_diff_term, mask_value_dict = train_ioi(
                gpt2, lambda_reg=a_regulation_param
            )
            # log, model, number_of_nodes, logit_diff, nodes_to_mask = train_ioi(
            #     gpt2, lambda_reg=a_regulation_param
            # )
            # print("nodes to mask", nodes_to_mask)
            # logit_diff_list.append(logit_diff * -1)
            # number_of_nodes_list.append(number_of_nodes)
            sanity_check_with_transformer_lens(mask_value_dict)
=======
    trainer = torch.optim.Adam(
        [
            {"params": mask_params, "lr": 0.0, "lr_base": mask_lr_base, "name": "mask"},
            {"params": gpt2_params, "lr": 0.0, "lr_base": lr_base, "name": "gpt2"},
        ],
        lr=0.0,
    )

    def set_lr(lr_ratio):
        for param_group in trainer.param_groups:
            param_group["lr"] = param_group["lr_base"] * lr_ratio
>>>>>>> Stashed changes

    log = []
    processed = 0
    check_processed = 0
    check_every = 2048
    lambda_reg = lambda_init
    for epoch in range(epochs):  # tqdm.notebook.tqdm(range(epochs)):
        np.random.shuffle(train_data)
        for i in range(
            0, len(train_data), batch_size
        ):  # tqdm.notebook.tqdm(range(0, len(train_data), batch_size)):
            examples = train_data[i : i + batch_size]
            gpt2.train()
            trainer.zero_grad()
            if len(examples) == batch_size:
                # compute loss, also log other metrics
                for i in range(0, len(examples), subbatch_size):
                    examples_subbatch = examples[i : i + subbatch_size]
                    sents = [exmp["sent"] for exmp in examples]
                    labels = [exmp["labels"] for exmp in examples]
                    loss = F.cross_entropy(
                        gpt2.predict_batch(sents),
                        from_numpy(pack_labels(labels)).long(),
                    )
                    (loss * len(examples_subbatch) / len(examples)).backward()
                if masked:
                    reg = gpt2.gpt2.compute_total_regularizer()
                    (lambda_reg * reg).backward()
                # mask_grad_norm = torch.nn.utils.clip_grad_norm_(mask_params, np.inf)
                # gpt2_grad_norm = torch.nn.utils.clip_grad_norm_(gpt2_params, np.inf)
                trainer.step()

                processed += len(examples)
                check_processed += len(examples)

                # warmup from 0 to lr_base for lr_warmup_frac
                lr_ratio = min(
                    1, processed / (lr_warmup_frac * epochs * len(train_data))
                )
                set_lr(lr_ratio)

                # schedule lambda_reg - constant, then linear, then constant
                lambda_ratio = max(
                    0,
                    min(
                        1,
                        (processed - lambda_startup_frac * epochs * len(train_data))
                        / (lambda_warmup_frac * epochs * len(train_data)),
                    ),
                )
                lambda_reg = lambda_init + (lambda_final - lambda_init) * lambda_ratio

                if check_processed >= check_every:
                    if masked:
                        log.append(
                            {
                                "dev_acc": calc_dev(),
                                "loss_val": loss.item(),
                                "reg_val": reg.item(),
                                #'mask_grad_norm': mask_grad_norm.item(),
                                #'gpt2_grad_norm': gpt2_grad_norm.item(),
                                "pct_binary": gpt2.gpt2.compute_binary_pct(),
                            }
                        )
                    else:
                        log.append({"dev_acc": calc_dev(), "loss_val": loss.item()})
                    check_processed -= check_every
                    if converged(log):
                        break
                    if verbose:
                        print("Log: {}".format(log[-1]))
        if converged(log):
            break
    return log, gpt2


def save_mask_scores(model, log, base="../test/mask_scores"):
    keys = ["dev_acc", "reg_val"]
    fname = base
    for key in keys:
        if key in log[-1].keys():
            fname = fname + "_{}={:.4f}".format(key, log[-1][key] * 100)
    fname = fname + ".pt"

    print("Saving to {}...".format(fname))
    torch.save(partial_state_dict(model.state_dict()), fname)
    print("Done saving")
