"""Evaluate results."""

import ast

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 


def plot_loss_curves_llm(
        file: str,
        attn_type: str,
        feature_map_qkv: str | None = None,
        feature_map_attn: str | None = None,
        use_out_proj: bool | None = None,
        identity_weight: float | None = None,
        to_plot: str = "val_loss",
) -> None:
    assert to_plot in ("train_loss", "val_loss", "val_acc")

    filters = pl.col("attn_type") == attn_type
    if feature_map_qkv is not None:
        filters &= pl.col("feature_map_qkv") == feature_map_qkv
    if feature_map_attn is not None:
        filters &= pl.col("feature_map_attn") == feature_map_attn
    if use_out_proj is not None:
        filters &= (pl.col("use_out_proj") == use_out_proj)
    if identity_weight is not None:
        filters &= (pl.col("identity_weight") == identity_weight)

    df = pl.scan_csv(file).filter(filters).collect()

    def to_array(series: pl.Series) -> np.ndarray:
        return np.array(ast.literal_eval(series[0]))

    columns_list = [c for c in df.columns if (to_plot in c) and ("avg" not in c)]
    ys = np.array([to_array(df[c]) for c in columns_list])
    num_datapoints = len(ys[0])

    if "train" in to_plot:
        xs = (np.arange(num_datapoints) + 1) * 10
    elif "val" in to_plot:
        xs = (np.arange(num_datapoints) + 1) * 50

    avg_ys = np.zeros_like(ys[0])
    for y in ys:
        avg_ys += y
    avg_ys /= num_datapoints

    # data = {
    #     "x": xs,
    #     **{
    #         c: y
    #         for c, y in zip(columns_list, ys, strict=True)
    #     }
    # }
    # for col in columns_list:
    #     sns.lineplot(data, x="x", y=col, palette=sns.color_palette(['#ffb3b3']), hue=None)

    data_avg = {
        "x": xs,
        "y": avg_ys
    }
    sns.lineplot(data_avg, x="x", y="y")
    plt.show()
    

if __name__ == "__main__":
    plot_loss_curves_llm(
        file="../results/results_llm_1500_steps.csv",
        attn_type="vanilla",
        
    )
