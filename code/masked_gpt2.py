from transformer_lens.HookedTransformer import MaskedHookedTransformer
from transformer_lens.ioi_dataset import IOIDataset
import torch
import numpy as np

# import IPython

# if IPython.get_ipython() is None:
#     IPython.get_ipython().run_line_magic("load_ext", "autoreload")
#     IPython.get_ipython().run_line_magic("autoreload", "2")

N = 100
# blackbox this bit
ioi_dataset = IOIDataset(prompt_type="ABBA", N=N, nb_templates=1,)
patch_dataset = (
    ioi_dataset.gen_flipped_prompts(("IO", "RAND"))
    .gen_flipped_prompts(("S", "RAND"))
    .gen_flipped_prompts(("S1", "RAND"))
)


# use these two!!!
print(ioi_dataset.toks.long().shape)
assert ioi_dataset.toks.long().shape == patch_dataset.toks.long().shape

gpt2 = MaskedHookedTransformer.from_pretrained("gpt2")

print(gpt2.is_masked)


def regularizer(
    gpt2: MaskedHookedTransformer,
    gamma: float = -0.1,
    zeta: float = 1.1,
    beta: float = 2 / 3,
) -> torch.Tensor:
    def regularization_term(mask: torch.nn.Parameter) -> torch.Tensor:
        return torch.sigmoid(mask - beta * np.log(-gamma / zeta)).sum() / mask.numel()

    mask_scores = [
        regularization_term(p)
        for (n, p) in gpt2.named_parameters()
        if "mask_scores" in n
    ]
    return torch.mean(torch.stack(mask_scores))


regularizer_loss = regularizer(gpt2)
import ipdb

ipdb.set_trace()
