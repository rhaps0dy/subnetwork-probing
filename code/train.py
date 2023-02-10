import IPython
import torch
import torch.nn.functional as F
import numpy as np
from transformer_lens.HookedTransformer import MaskedHookedTransformer
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
from transformer_lens.ioi_dataset import IOIDataset

N = 100


def regularizer(
    gpt2: MaskedHookedTransformer,
    gamma: float = -0.1,
    zeta: float = 1.1,
    beta: float = 2 / 3,
) -> torch.Tensor:
    # TODO: globally read hyperparams from config
    # need to also do this in the masked hook point so
    # the hyperparams are the same
    def regularization_term(mask: torch.nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(mask - beta * np.log(-gamma / zeta)).sum() / mask.numel()

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
    gpt2,
    lambda_init=1000,
    lambda_final=10000,
    # lr_base=3e-5, #TODO: just going to do simple lr stuff for now
    # mask_lr_base=0.1,
    mask_lr=1e-3,
    lr_warmup_frac=0.1,
    epochs=3,
    batch_size=32,
    verbose=True,
    lambda_startup_frac=0.25,
    lambda_warmup_frac=0.5,
    subbatch_size=8,
    masked=True,
):
    # blackbox this bit
    ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1,)
    train_data = ioi_dataset.toks.long()

    # print(
    #     "lambda_init: {}, lambda_final: {}, lambda_startup_frac: {}, lambda_warmup_frac: {}".format(
    #         lambda_init, lambda_final, lambda_startup_frac, lambda_warmup_frac
    #     )
    # )
    # print(
    #     "lr_base: {}, mask_lr_base: {}, lr_warmup_frac: {}, epochs: {}, batch_size: {}".format(
    #         lr_base, mask_lr_base, lr_warmup_frac, epochs, batch_size
    #     )
    # )

    # group parameters for different learning rates

    # one parameter per thing that is masked
    mask_params = [
        p for n, p in gpt2.named_parameters() if "mask_scores" in n and p.requires_grad
    ]
    # parameters for the probe (we don't use a probe)
    gpt2_params = [
        p
        for n, p in gpt2.named_parameters()
        if "mask_scores" not in n and p.requires_grad
    ]

    assert len(gpt2_params) == 0, ("GPT2 should be empty", gpt2_params)

    # trainer = torch.optim.Adam(
    #     [
    #         {"params": mask_params, "lr": 1, "lr_base": mask_lr_base, "name": "mask",},
    #         {"params": gpt2_params, "lr": 0.0, "lr_base": lr_base, "name": "gpt2"},
    #     ],
    #     lr=0,
    # )

    # def set_lr(lr_ratio):
    #     for param_group in trainer.param_groups:
    #         param_group["lr"] = param_group["lr_base"] * lr_ratio

    trainer = torch.optim.Adam(mask_params, lr=mask_lr)
    log = []
    lambda_reg = lambda_init
    for epoch in range(epochs):  # tqdm.notebook.tqdm(range(epochs)):
        gpt2.train()
        trainer.zero_grad()
        # compute loss, also log other metrics
        logit_diff_term = -1.0 * logit_diff_from_ioi_dataset(
            gpt2(train_data), train_data, mean=True
        )
        regularizer_term = regularizer(gpt2)
        loss = logit_diff_term + lambda_reg * regularizer_term
        loss.backward()

        if masked:
            reg = gpt2.gpt2.compute_total_regularizer()
            (lambda_reg * reg).backward()
        # mask_grad_norm = torch.nn.utils.clip_grad_norm_(mask_params, np.inf)
        # gpt2_grad_norm = torch.nn.utils.clip_grad_norm_(gpt2_params, np.inf)
        trainer.step()

        # processed += len(examples)
        # check_processed += len(examples)

        # warmup from 0 to lr_base for lr_warmup_frac
        # lr_ratio = min(1, epoch / (lr_warmup_frac * epochs * len(train_data)))
        # set_lr(lr_ratio)

        # schedule lambda_reg - constant, then linear, then constant
        lambda_ratio = max(
            0,
            min(
                1,
                (epoch - lambda_startup_frac * epochs * len(train_data))
                / (lambda_warmup_frac * epochs * len(train_data)),
            ),
        )
        lambda_reg = lambda_init + (lambda_final - lambda_init) * lambda_ratio

        if masked:
            log.append(
                {
                    "loss_val": loss.item(),
                    # 'reg_val': reg.item(), # TODO add reg
                    # 'mask_grad_norm': mask_grad_norm.item(),
                    # 'gpt2_grad_norm': gpt2_grad_norm.item(),
                    # 'pct_binary': gpt2.gpt2.compute_binary_pct()
                }
            )
        else:
            log.append({"loss_val": loss.item()})
        if verbose:
            print("Log: {}".format(log[-1]))
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
