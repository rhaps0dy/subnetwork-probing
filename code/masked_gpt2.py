#%%
# import os 
# os.chdir("/home/ubuntu/TransformerLens/transformer_lens/")

from transformer_lens.HookedTransformer import MaskedHookedTransformer

gpt2 = MaskedHookedTransformer.from_pretrained("gpt2")

print(gpt2.is_masked)