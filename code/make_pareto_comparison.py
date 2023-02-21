#%%

"""
File for doing misc plotting
"""
import wandb
import numpy as np


def get_col_from_df(df, col_name):
    return df[col_name].values


def df_to_np(df):
    return df.values


def get_time_diff(run_name):
    """Get the difference between first log and last log of a WANBB run"""
    api = wandb.Api()
    run = api.run(run_name)
    df = run.history()["_timestamp"]
    arr = df_to_np(df)
    n = len(arr)
    for i in range(n - 1):
        assert arr[i].item() < arr[i + 1].item()
    print(arr[-1].item() - arr[0].item())


def get_nonan(arr, last=True):
    """Get last non nan by default (or first if last=False)"""

    indices = list(range(len(arr) - 1, -1, -1)) if last else list(range(len(arr)))

    for i in indices:  # range(len(arr)-1, -1, -1):
        if not np.isnan(arr[i]):
            return arr[i]

    return np.nan


def get_corresponding_element(
    df, col1_name, col1_value, col2_name,
):
    """Get the corresponding element of col2_name for a given element of col1_name"""
    col1 = get_col_from_df(df, col1_name)
    col2 = get_col_from_df(df, col2_name)
    for i in range(len(col1)):
        if col1[i] == col1_value and not np.isnan(col2[i]):
            return col2[i]
    assert False, "No corresponding element found"


def get_first_element(
    df, col, last=False,
):
    col1 = get_col_from_df(df, "_step")
    col2 = get_col_from_df(df, col)

    cur_step = 1e30 if not last else -1e30
    cur_ans = None

    for i in range(len(col1)):
        if not last:
            if col1[i] < cur_step and not np.isnan(col2[i]):
                cur_step = col1[i]
                cur_ans = col2[i]
        else:
            if col1[i] > cur_step and not np.isnan(col2[i]):
                cur_step = col1[i]
                cur_ans = col2[i]

    assert cur_ans is not None
    return cur_ans


def get_longest_float(s):
    ans = None

    for i in range(len(s) - 1, -1, -1):
        try:
            ans = float(s[i:])
        except:
            pass
        else:
            ans = float(s[i:])
    assert ans is not None
    return ans


import warnings
import torch
import os
import pandas as pd
import wandb
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import IPython

if IPython.get_ipython() is not None:
    IPython.get_ipython().run_line_magic("load_ext", "autoreload")
    IPython.get_ipython().run_line_magic("autoreload", "2")

#%%

project_names = [
    "induction_arthur",
]

api = wandb.Api()
ALL_COLORS = [
    "blue",
    "red",
    "green",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

final_edges = []
final_metric = []
names = []
COLS = [
    "black",
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]
col_dict = {
    "geometric": "blue",
    "harmonic": "red",
    "off": "black",
}
colors = []
_initial_losses = []  # [1.638 for _ in range(len(names))]
_initial_edges = []  # [11663 for _ in range(len(names))]
histories = []
for pi, project_name in enumerate(project_names):
    runs = api.runs(f"remix_school-of-rock/{project_name}")
    for run in runs:
        print(run.name, run.state)
        if run.state == "finished" or run.state == "failed":

            history = pd.DataFrame(run.scan_history())
            histories.append(history)

            min_edges = history["num_edges_total"].min()
            max_edges = history["num_edges_total"].max()
            assert 0 < min_edges, min_edges
            assert 1e30 > max_edges, max_edges

            start_metric = get_first_element(history, "self.cur_metric")
            end_metric = get_first_element(history, "self.cur_metric", last=True)

            if (
                "num_edges_total" in history.keys()
                and "self.cur_metric" in history.keys()
                and run.name not in names
            ):
                names.append(run.name)

                final_edges.append(min_edges)
                _initial_edges.append(max_edges)

                final_metric.append(end_metric)
                _initial_losses.append(start_metric)

                colors.append("black")

if torch.norm(torch.tensor(_initial_losses) - _initial_losses[0]) > 1e-5:
    warnings.warn(
        f"Initial losses are not the same, so this may be an unfair comparison of {_initial_losses=}"
    )
if torch.norm(torch.tensor(_initial_edges).float() - _initial_edges[0]) > 1e-5:
    warnings.warn(
        f"Initial edges are not the same, so this may be an unfair comparison of {_initial_edges=}"
    )

added_final_edges = False
thresholds = [get_longest_float(name) for name in names]

#%%

if not added_final_edges:
    final_edges.append(_initial_edges[0])
    final_metric.append(_initial_losses[0])  # from just including the whole graph
    names.append("The whole graph")
    colors.append("black")
    added_final_edges = True

fig = go.Figure()

# scatter plot with names as labels and thresholds as colors

fig.add_trace(
    go.Scatter(
        x=final_edges,
        y=final_metric,
        mode="markers",
        marker=dict(
            size=10,
            color=thresholds,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="Threshold",
                titleside="right",
                tickmode="array",
                tickvals=np.arange(0, 10) / 10,
                ticktext=np.arange(0, 10) / 10,
            ),
        ),
        text=names,
    )
)

for y_val, text in zip(
    [_initial_losses[0], 4.493981649709302],
    ["The whole graph", "Induction heads scrubbed"],
):
    # add a dotted line y=WHOLE_LOSS
    fig.add_shape(
        type="line",
        x0=0,
        x1=_initial_edges[0],
        y0=y_val,
        y1=y_val,
        line=dict(color="Black", width=1, dash="dot",),
    )
    # add label to this
    fig.add_annotation(
        x=0,
        y=y_val,
        text=text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        xshift=10,
        yshift=2,
        font=dict(family="Courier New, monospace", size=16, color="Black"),
    )

# add legend for colors
fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))

# rescale
# fig.update_xaxes(range=[0, 1500])
# fig.update_yaxes(range=[0, max(final_metric)+0.01])

# add axis labels
fig.update_xaxes(title_text="Number of edges")
fig.update_yaxes(title_text="Induction metric")

# add title
fig.update_layout(title_text="Docstring circuit results")

#%%
