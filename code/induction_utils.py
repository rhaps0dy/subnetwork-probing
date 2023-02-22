import os
import pickle
from typing import Tuple

import einops
import rust_circuit as rc
import torch
from interp.circuit.causal_scrubbing.dataset import Dataset
from interp.circuit.interop_rust.model_rewrites import (To,
                                                        configure_transformer)
from interp.circuit.interop_rust.module_library import load_model_id
from interp.circuit.projects.gpt2_gen_induction.rust_path_patching import \
    make_arr
from interp.tools.data_loading import get_val_seqs
from interp.tools.indexer import TORCH_INDEXER as I
from interp.tools.rrfs import RRFS_DIR
from transformer_lens import HookedTransformer


def get_induction_dataset():
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
    DEVICE = "cuda:0"
    SEQ_LEN = 300
    NUM_EXAMPLES = 40
    MODEL_ID = "attention_only_2"
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

    default_input = toks_int_values.rename("tokens")
    default_output = toks_int_labels.rename("labels")

    print("\ntokens of input and output")
    print(tokenizer.batch_decode(default_input.evaluate()[0, :10]))
    print(tokenizer.batch_decode(default_output.evaluate()[0, :10]))

    patch_input = toks_int_values_other.rename("tokens")
    patch_output = default_output  # oo cares..

    default_ds = Dataset({"tokens": default_input, "labels": default_output})
    # patched_ds = Dataset({"tokens": patch_input, "labels": patch_output})
    return (
        default_ds.tokens.evaluate(),
        default_ds,
        circ_dict,
        model_info,
        mask_reshaped,
    )


def get_induction_model() -> HookedTransformer:
    SEQ_LEN = 300
    tokens_device_dtype = rc.TorchDeviceDtype(device="cuda", dtype="int64")

    _, default_ds, circ_dict, model_info, _ = get_induction_dataset()
    orig_circuit = circ_dict["t.bind_w"]
    tok_embeds = circ_dict["t.w.tok_embeds"]
    pos_embeds = circ_dict["t.w.pos_embeds"]

    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer("Hello world")["input_ids"]

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

    """
    Now get on to porting this to TransformerLens
    """

    import transformer_lens
    from transformer_lens.HookedTransformerConfig import \
        HookedTransformerConfig

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
    return et_model
