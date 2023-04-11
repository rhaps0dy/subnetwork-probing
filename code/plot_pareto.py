import argparse

import pandas as pd
import plotly.express as px
from acdc.acdc_utils import EdgeType, TorchIndex
from acdc.TLACDCInterpNode import TLACDCInterpNode
from transformer_lens.HookedTransformer import HookedTransformer

import wandb
from train_induction import get_transformer_config, correspondence_from_mask

parser = argparse.ArgumentParser("make pareto plot")
parser.add_argument("--wandb-entity", type=str, required=True)
parser.add_argument("--wandb-group", type=str, required=True)


def parse_interpnode(s: str) -> TLACDCInterpNode:
    name, idx = s.split("[")
    idx = int(idx[-2])
    return TLACDCInterpNode(name, TorchIndex([None, None, idx]), EdgeType.ADDITION)


def main(args):
    api = wandb.Api()
    runs = api.runs(
        f"{args.wandb_entity}/subnetwork-probing", filters={"group": args.wandb_group}
    )

    cfg = get_transformer_config()
    model = HookedTransformer(cfg, is_masked=True)

    df_rows = []
    for run in runs:
        try:
            nodes_to_mask_str = run.summary["nodes_to_mask"]
        except KeyError:
            print("Skipping run", run.name, 'because "nodes_to_mask" not found')
            continue

        nodes_to_mask = list(map(parse_interpnode, nodes_to_mask_str))
        corr = correspondence_from_mask(model, nodes_to_mask)

        try:
            row = {
                "number_of_edges": corr.count_no_edges(),
                "logit_diff": run.summary["logit_diff"],
                "percentage_binary": run.summary["percentage_binary"],
                "lambda_reg": run.config["lambda_reg"],
                "zero_ablation": run.config["zero_ablation"],
                "reset_target": run.config["reset_target"],
                "reset_subject": run.config["reset_subject"],
            }
        except KeyError as e:
            print("Skipping run", run.name, "because", e, "not found")
            continue

        row["color"] = px.colors.qualitative.G10[
            int(row["zero_ablation"]) * 4
            + int(row["reset_target"]) * 2
            + int(row["reset_subject"])
        ]

        df_rows.append(row)
    plt = px.scatter(
        pd.DataFrame(df_rows),
        x="number_of_edges",
        y="logit_diff",
        hover_data=[
            "lambda_reg",
            "percentage_binary",
            "zero_ablation",
            "reset_target",
            "reset_subject",
        ],
        color_discrete_map="identity",
        color="color",
    )
    plt.update_layout(xaxis_title="Number of Edges", yaxis_title="KL")
    plt.write_html(f"pareto_{args.wandb_group}.html")
    return plt


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
