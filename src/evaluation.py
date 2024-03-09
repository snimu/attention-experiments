"""Evaluate results."""

import ast
import copy
import itertools
import math

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
import torch
import colorsys
from scipy.stats import pearsonr


def series_to_array(series: pl.Series) -> np.ndarray:
    return np.array(ast.literal_eval(series[0]))


def load_xs_ys_avg_y(
        file: str,
        attn_type: str,
        feature_map_qkv: str | None = None,
        feature_map_attn: str | None = None,
        use_out_proj: bool | None = None,
        use_x_norm: bool | None = None,
        use_qkv_norm: bool | None = None,
        use_qkv_weight: bool | None = None,
        identity_weight: float | None = None,
        residual_depth: int | None = None,
        logit_scalar: str | None = None,
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
    if use_x_norm is not None:
        filters &= (pl.col("use_x_norm") == use_x_norm)
    if use_qkv_norm is not None:
        filters &= (pl.col("use_qkv_norm") == use_qkv_norm)
    if use_qkv_weight is not None:
        filters &= (pl.col("use_qkv_weight") == use_qkv_weight)
    if residual_depth is not None:
        filters &= (pl.col("residual_depth") == residual_depth)
    if logit_scalar is not None:
        filters &= (pl.col("logit_scalar") == logit_scalar)

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


def get_attn_types(file: str) -> list[str]:
    return (
        pl.scan_csv(file)
        .select(pl.col("attn_type"))
        .collect()
        ["attn_type"]
        .unique()
        .to_list()
    )


def get_rand_colors(n: int, random: bool = False) -> list:
    if n <= 8:
        colors = mcolors.BASE_COLORS
    elif n <= 10:
        colors = mcolors.TABLEAU_COLORS
    else:
        colors = mcolors.CSS4_COLORS

    colors = list(colors)
    if random:
        colors = [colors[i] for i in torch.randperm(n)]

    colors = colors[:n]

    return colors


def generate_distinct_colors(n):
    """
    Generates n visually distinct colors.

    Parameters:
        n (int): The number of distinct colors to generate.

    Returns:
        list: A list of n visually distinct colors in hex format.
    """
    colors = []
    for i in range(n):
        hue = i / n
        # Fixing saturation and lightness/value to 0.9 for bright colors
        # You can adjust these values for different color variations
        lightness = 0.5
        saturation = 0.9
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    
    return colors


def get_unique_settings(
        file: str,
        targets: list[str],
        attn_type: str | None = None,
) -> list[str | bool | float]:
    attn_types = get_attn_types(file) if attn_type is None else [attn_type]
    settings = []
    for attn_type in attn_types:
        combinations = (
            pl.scan_csv(file)
            .filter(pl.col("attn_type") == attn_type)
            .select(*[pl.col(target) for target in targets])
            .collect()
            .unique()
        )
        for features in zip(
                *[combinations[target] for target in targets]
        ):
            settings.append((attn_type,) + tuple(features))
    return settings


def plot_llm_1500_steps_by_norm_position(
        file: str = "../results/results_llm_1500_steps.csv",
        to_plot: str = "val_loss",
        attn_type: str | None  = "vanilla",
        show_all_plots: bool = False,
) -> None:
    settings = get_unique_settings(
        file, 
        targets=["use_x_norm", "use_qkv_norm"], 
        attn_type=attn_type
    )
    colors = get_rand_colors(len(settings))
    for (attn_type, use_x_norm, use_qkv_norm), color in zip(settings, colors, strict=True):
        xs, ys, avg_y = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            to_plot=to_plot,
        )
        label = f"{attn_type}"
        if use_x_norm:
            label += " x-norm"
        if use_qkv_norm:
            label += " qkv-norm"
        plt.plot(xs, avg_y, color=color, label=label)
        if show_all_plots:
            for y in ys:
                plt.plot(xs, y, color, alpha=0.2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Average {to_plot}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_llm_1000_steps_100_tries_by_norm_position(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        to_plot: str = "val_loss",
        attn_type: str | None  = "vanilla",
        show_all_plots: bool = False,
) -> None:
    settings = get_unique_settings(
        file, 
        targets=["use_x_norm", "use_qkv_norm"], 
        attn_type=attn_type
    )
    colors = generate_distinct_colors(len(settings))
    for (attn_type, use_x_norm, use_qkv_norm), color in zip(settings, colors, strict=True):
        xs, ys, avg_y = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            logit_scalar="sqrt_dh",
            to_plot=to_plot,
        )
        label = f"{attn_type}"
        if use_x_norm:
            label += " x-norm"
        if use_qkv_norm:
            label += " qkv-norm"
        if show_all_plots:
            for y in ys:
                plt.plot(xs, y, color, alpha=0.3/math.sqrt(len(ys)))
        plt.plot(xs, avg_y, color=color, label=label, linewidth=2)

    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"Average {to_plot}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_llm_1000_steps_100_tries_by_norm_position_single_setting(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        attn_type: str = "vanilla",
        to_plot: str = "val_loss",
        show_all_plots: bool = True,
        use_x_norm: bool = False,
        use_qkv_norm: bool = False,
        logit_scalar: str = "sqrt_dh",
) -> None:
    xs, ys, avg_y = load_xs_ys_avg_y(
        file=file,
        attn_type=attn_type,
        use_x_norm=use_x_norm,
        use_qkv_norm=use_qkv_norm,
        logit_scalar=logit_scalar,
        to_plot=to_plot,
    )
    if show_all_plots:
        for y in ys:
            plt.plot(xs, y, alpha=0.3, color="pink")
    plt.plot(xs, avg_y, label=f"{attn_type}: mean", color="red", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel(to_plot)
    plt.title(f"Average {to_plot}")
    plt.grid()
    plt.show()


def plot_metric_variance(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        attn_type: str = "vanilla",
        to_plot: str = "val_loss",
) -> None:
    settings = get_unique_settings(
        file, 
        targets=["use_x_norm", "use_qkv_norm"], 
        attn_type=attn_type
    )
    for (attn_type, use_x_norm, use_qkv_norm) in settings:
        xs, ys, avg_y = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            logit_scalar="sqrt_dh",
            to_plot=to_plot,
        )
        label = f"{attn_type}"
        if use_x_norm:
            label += " x-norm"
        if use_qkv_norm:
            label += " qkv-norm"
        plt.plot(xs, np.std(ys, axis=0), label=label)
    plt.xlabel("Steps")
    plt.ylabel("Standard deviation")
    plt.title(f"Standard deviation of {to_plot}")
    plt.legend()
    plt.grid()
    plt.show()


def print_loss_acc_correlation(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        attn_type: str = "vanilla",
        train: bool = False,
) -> None:
    settings = get_unique_settings(
        file, 
        targets=["use_x_norm", "use_qkv_norm"], 
        attn_type=attn_type
    )
    for (attn_type, use_x_norm, use_qkv_norm) in settings:
        _, _, avg_loss = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            logit_scalar="sqrt_dh",
            to_plot=("train" if train else "val") + "_loss",
        )
        avg_loss = avg_loss.max() - avg_loss  # Invert loss to make it line up with accuracy
        _, _, avg_acc = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            logit_scalar="sqrt_dh",
            to_plot=("train" if train else "val") + "_acc",
        )
        correlation = pearsonr(avg_loss, avg_acc)[0]
        label = f"{attn_type}"
        if use_x_norm:
            label += " x-norm"
        if use_qkv_norm:
            label += " qkv-norm"
        print(f"{label}: {correlation:.4f}")



def plot_loss_curves_avg_contrast_1500_steps(
        file: str = "../results/results_llm_1500_steps.csv",
        to_plot: str = "val_loss",
) -> None:
    attn_types = get_attn_types(file)
    settings = []
    for attn_type in attn_types:
        feature_map_combinations = (
            pl.scan_csv(file)
            .filter(pl.col("attn_type") == attn_type)
            .select(pl.col("feature_map_qkv"), pl.col("feature_map_attn"))
            .collect()
            .unique()
        )
        for (feature_map_qkv, feature_map_attn) in zip(
                feature_map_combinations["feature_map_qkv"], feature_map_combinations["feature_map_attn"]
        ):
            settings.append((attn_type, feature_map_qkv, feature_map_attn))
    colors = ("blue", "violet", "brown", "orange", "green", "red", "purple")[:len(settings)]
    for (attn_type, feature_map_qkv, feature_map_attn), color in zip(settings, colors, strict=True):
        xs, ys, avg_y = load_xs_ys_avg_y(
            file = file,
            attn_type = attn_type,
            to_plot=to_plot,
            feature_map_attn=feature_map_attn,
            feature_map_qkv=feature_map_qkv,
        )
        plt.plot(xs, avg_y, color=color, label=f"{attn_type} ({feature_map_qkv}-{feature_map_attn})")
        for y in ys:
            plt.plot(xs, y, color, alpha=0.2)
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

    for color, attn_type, feature_map_qkv, feature_map_attn in zip(
            ("red", "purple", "green", "blue", "orange"),
            (attn_type, attn_type, attn_type, "identity", "vanilla"),
            ("identity", best_feature_map_qkv, "cos_sim", "identity", "identity"),
            ("cos_sim", best_feature_map_attn, "identity", "identity", "identity"),
    ):
        xs, ys, avg_y = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            feature_map_qkv=feature_map_qkv,
            feature_map_attn=feature_map_attn,
            to_plot=to_plot,
        )
        label = f"{attn_type}"
        if attn_type not in ("identity", "vanilla"):
            label += f" ({feature_map_qkv}-{feature_map_attn})"
        plt.plot(xs, avg_y, label=label, color=color)

        for y in ys:
            plt.plot(xs, y, alpha=0.1, color=color)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title(f"{attn_type} {to_plot}")
    plt.legend()
    plt.grid()
    plt.show()


def get_outputs_diffusion(
        file: str,
        in_attn: str,
        mid_attn: str,
        out_attn: str,
        to_plot: str,
        from_step: int = 0,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    filters = pl.col("in_attn") == in_attn
    filters &= pl.col("mid_attn") == mid_attn
    filters &= pl.col("out_attn") == out_attn

    df = pl.scan_csv(file).filter(filters).collect()
    trial_nums = df["trial_num"].to_numpy()
    outputs = [
        df.filter(pl.col("trial_num") == trial_num)[to_plot].item()
        for trial_num in trial_nums
    ]
    xs = np.arange(len(outputs[0].split(","))) + from_step
    # Remove "nan" from outputs by only using outputs up to the first "nan"
    for i, output in enumerate(outputs):
        if "nan" in output:
            first_nan_pos = output.index("nan")
            output = output[:first_nan_pos] + "]"
        output = ast.literal_eval(output)[from_step:]
        outputs[i] = np.array(output)

    # Calculate the average output by 
        # 1. Calculating the lenghts of all outputs 
        # 2. The avg_output from 0 to the minimum length is the mean of all output in that range
        # 2. The avg_output from the smallest to the next smallest length is the mean of all output in that range
        # 3. Repeat until the maximum length
    working_outputs = copy.deepcopy(outputs)  # copy to not change the original list
    working_outputs.sort(key=len)
    avg_outputs = np.zeros(len(working_outputs[-1]))
    lengths = [0] + [len(output) for output in working_outputs]
    for i, _ in enumerate(lengths):
        if i == len(lengths) - 1:
            break
        start_len, end_len = lengths[i], lengths[i+1]
        outputs_slice = [o[start_len:end_len] for o in working_outputs]
        avg_outputs[start_len:end_len] = np.mean(outputs_slice, axis=0)
        working_outputs.pop(0)

    return xs, outputs, avg_outputs



def plot_loss_curves_diffusion(
        file: str,
        in_attns: list[str], 
        mid_attns: list[str], 
        out_attns: list[str],
        to_plot: str = "losses",  # could also be the times taken per step
        show_all_trials: bool = False,
        from_step: int = 0,
) -> None:
    colors = plt.cm.tab10.colors[:len(in_attns)]

    for in_attn, mid_attn, out_attn, color in zip(
            in_attns, mid_attns, out_attns, colors
    ):
        xs, ys, avg_y = get_outputs_diffusion(
            file=file,
            in_attn=in_attn,
            mid_attn=mid_attn,
            out_attn=out_attn,
            to_plot=to_plot,
            from_step=from_step,
        )
        label = f"{in_attn}-{mid_attn}-{out_attn}"
        plt.plot(xs, avg_y, label=label, color=color)

        if show_all_trials:
            for y in ys:
                plt.plot(xs, y, color=color, alpha=0.1)

    plt.xlabel("Steps")
    plt.ylabel(to_plot)
    plt.title(f"Diffusion {to_plot}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_loss_curves_diffusion_single_attn(
        file: str,
        attn_type: str,
        to_plot: str = "losses",
        show_all_trials: bool = False,
        from_step: int = 0,
) -> None:
    xs, ys, avg_y = get_outputs_diffusion(
        file=file,
        in_attn=attn_type,
        mid_attn=attn_type,
        out_attn=attn_type,
        to_plot=to_plot,
        from_step=from_step,
    )

    if show_all_trials:
        for i, y in enumerate(ys):
            plt.plot(xs[:len(y)], y, alpha=0.3, color="blue", label=f"{attn_type}: trials" if i == 0 else None)
    plt.plot(xs[:len(avg_y)], avg_y, label=f"{attn_type}: mean", color="red")
    plt.xlabel("Steps")
    plt.ylabel(to_plot)
    plt.title(f"Diffusion {to_plot}")
    plt.legend()
    plt.grid()
    plt.show()


def find_best_attn_setting_diffusion(file: str) -> None:
    df = pl.scan_csv(file).collect()
    
    in_attns = df["in_attn"].unique()
    mid_attns = df["mid_attn"].unique()
    out_attns = df["out_attn"].unique()

    best_loss = float("inf")

    for in_attn, mid_attn, out_attn in itertools.product(in_attns, mid_attns, out_attns):
        xs, _, avg_y = get_outputs_diffusion(
            file=file,
            in_attn=in_attn,
            mid_attn=mid_attn,
            out_attn=out_attn,
            to_plot="losses",
        )
        if avg_y.min().item() < best_loss:
            best_loss = avg_y[-1]
            best_in_attn = in_attn
            best_mid_attn = mid_attn
            best_out_attn = out_attn

    print(f"Best attn setting: {best_in_attn}-{best_mid_attn}-{best_out_attn}")
    

if __name__ == "__main__":
    # plot_loss_curves_diffusion(
    #     file="../results/results_diffusion_20_epochs.csv",
    #     in_attns=["identity", "linear", "hydra", "hercules", "zeus"],
    #     mid_attns=["identity", "linear", "hydra", "hercules", "zeus"],
    #     out_attns=["identity", "linear", "hydra", "hercules", "zeus"],
    #     to_plot="losses",
    #     from_step=0,
    # )
    # plot_loss_curves_feature_maps("hydra")
    # plot_loss_curves_diffusion_single_attn(
    #     file="../results/results_diffusion_20_epochs.csv",
    #     attn_type="hydra",
    #     to_plot="losses",
    #     show_all_trials=True,
    #     from_step=0,
    # )
    # plot_loss_curves_avg_contrast_1500_steps(
    #     file="../results/results_llm_hydra_feature_maps.csv",
    #     to_plot="val_loss",
    # )
    # plot_llm_1500_steps_by_norm_position(attn_type="vanilla", to_plot="val_acc")
    to_plot = "val_loss"
    plot_llm_1000_steps_100_tries_by_norm_position(
        attn_type="vanilla", 
        to_plot=to_plot,
        show_all_plots=False,
    )
    # plot_llm_1000_steps_100_tries_by_norm_position_single_setting(
    #     attn_type="vanilla",
    #     to_plot=to_plot,
    #     show_all_plots=True,
    #     use_x_norm=False,
    #     use_qkv_norm=False,
    #     logit_scalar="sqrt_dh",
    # )
    plot_metric_variance(to_plot=to_plot)
    print_loss_acc_correlation(attn_type="vanilla", train="train" in to_plot)
