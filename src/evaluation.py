"""Evaluate results."""

import ast

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 


def series_to_array(series: pl.Series) -> np.ndarray:
    return np.array(ast.literal_eval(series[0]))


def load_xs_ys_avg_y(
        file: str,
        attn_type: str,
        feature_map_qkv: str | None = None,
        feature_map_attn: str | None = None,
        use_out_proj: bool | None = None,
        identity_weight: float | None = None,
        to_plot: str = "val_loss",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
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

    columns_list = [c for c in df.columns if (to_plot in c) and ("avg" not in c)]
    ys = np.array([series_to_array(df[c]) for c in columns_list])
    num_datapoints = len(ys[0])

    if "train" in to_plot:
        xs = (np.arange(num_datapoints) + 1) * 10
    elif "val" in to_plot:
        xs = (np.arange(num_datapoints) + 1) * 50

    avg_ys = np.mean(ys, axis=0)

    return xs, ys, avg_ys


def plot_loss_curves_llm_single_attn(
        file: str,
        attn_type: str,
        feature_map_qkv: str | None = None,
        feature_map_attn: str | None = None,
        use_out_proj: bool | None = None,
        identity_weight: float | None = None,
        to_plot: str = "val_loss",
) -> None:
    """Plot loss curves."""
    xs, ys, avg_ys = load_xs_ys_avg_y(
        file=file,
        attn_type=attn_type,
        feature_map_qkv=feature_map_qkv,
        feature_map_attn=feature_map_attn,
        use_out_proj=use_out_proj,
        identity_weight=identity_weight,
        to_plot=to_plot,
    )

    plt.figure(figsize=(8, 6))
    for y in ys:
        plt.plot(xs, y, "k", alpha=0.1)
    plt.plot(xs, avg_ys, "r", label="Average")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"{attn_type} {to_plot}")
    plt.legend()
    plt.show()


def plot_loss_curves_avg_contrast_1500_steps(to_plot: str = "val_loss") -> None:
    for attn_type, color in zip(
            ("identity", "hlb-gpt", "torchMHA", "vanilla", "hydra", "hercules", "zeus"),
            ("blue", "violet", "brown", "orange", "green", "red", "purple"),
            strict=True,
    ):
        xs, ys, avg_y = load_xs_ys_avg_y(
            file = "../results/results_llm_1500_steps.csv",
            attn_type = attn_type,
            to_plot=to_plot,
        )
        plt.plot(xs, avg_y, color=color, label=attn_type)
        for y in ys:
            plt.plot(xs, y, color, alpha=0.1)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Average {to_plot}")
    plt.legend()
    plt.show()
    

def plot_loss_curves_feature_maps(
        attn_type: str,
        to_plot: str = "val_loss",
) -> None:
    """Plot loss curves."""
    # The given attn type is plotted thrice:
    # - once with feature_map_qkv="identity" and feature_map_attn="cos_sim" -> as in paper
    # - once with the feature map combination leading to the lowest loss    -> compare to late activation
    # - once with feature_map_qkv="cos_sim" and feature_map_attn="identity" -> show result of sweep
    # Two other attn types are plotted:
    # - "identity" -> as low baseline (only MLP)
    # - "vanilla"  -> as high baseline
    # All curves are in the same plot to make them more comparable
    file = "../results/results_llm_feature_maps.csv"

    # 1. Find the feature map combination leading to the lowest loss
    filters = pl.col("attn_type") == attn_type
    df = pl.scan_csv(file).filter(filters).collect()
    df = df.sort("avg_val_loss")
    best_feature_map_qkv = df["feature_map_qkv"][0]
    best_feature_map_attn = df["feature_map_attn"][0]

    for attn_type, feature_map_qkv, feature_map_attn in zip(
            (attn_type, attn_type, attn_type, "identity", "vanilla"),
            ("identity", best_feature_map_qkv, "cos_sim", "identity", "identity"),
            ("cos_sim", best_feature_map_attn, "identity", "identity", "identity"),
    ):
        xs, _, avg_y = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            feature_map_qkv=feature_map_qkv,
            feature_map_attn=feature_map_attn,
            to_plot=to_plot,
        )
        label = f"{attn_type}"
        if attn_type not in ("identity", "vanilla"):
            label += f"-{feature_map_qkv}-{feature_map_attn}"
        plt.plot(xs, avg_y, label=label)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"{attn_type} {to_plot}")
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    plot_loss_curves_feature_maps(
        attn_type="hydra",
        to_plot="val_loss",
    )
