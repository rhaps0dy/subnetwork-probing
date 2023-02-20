from transformer_lens.HookedTransformer import (
    MaskedHookedTransformer,
    HookedTransformer,
)
from transformer_lens.ioi_dataset import IOIDataset
from train import logit_diff_from_ioi_dataset

gpt2 = HookedTransformer.from_pretrained("gpt2")


ioi_dataset = IOIDataset(prompt_type="ABBA", N=100, nb_templates=1,)
train_data = ioi_dataset.toks.long()
logits = gpt2(train_data)
logit_diff = logit_diff_from_ioi_dataset(logits, train_data, mean=True)
assert logit_diff.detach().cpu().item() > 0.0
