"""Evaluate results."""

import ast
import copy
import itertools
import math
import os
from typing import Literal

import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
import torch
import colorsys
from scipy.stats import pearsonr


def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


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
        plot_over: Literal["step", "epoch", "token"] = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    assert to_plot in ("train_loss", "train_acc", "val_loss", "val_acc", "val_pplx"), f"Invalid to_plot: {to_plot}"

    filters = (pl.col("attn_type") == attn_type)
    if feature_map_qkv is not None:
        filters &= (pl.col("feature_map_qkv") == feature_map_qkv)
    if feature_map_attn is not None:
        filters &= (pl.col("feature_map_attn") == feature_map_attn)
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
    columns_list.sort(key=lambda x: int(x.split("_")[-1]))
    arrays = [series_to_array(df[c]) for c in columns_list]

    if plot_over == "step":
        return load_steps_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "epoch":
        return load_epochs_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "token":
        return load_tokens_ys_avg_ys(df, arrays, to_plot)
    else:
        raise ValueError(f"{plot_over} not a valid x-value")


def load_steps_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_len = min([len(a) for a in arrays])
    ys = np.array([list(a[:min_len]) for a in arrays])
    num_datapoints = len(ys[0])

    if "train" in to_plot:
        xs = ((np.arange(num_datapoints) + 1) * 12.5).astype(int)
    elif "val" in to_plot:
        xs = (np.arange(num_datapoints) + 1) * 50

    avg_ys = np.mean(ys, axis=0)

    return xs, ys, avg_ys


def load_epochs_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    epochs_str = "epochs_train_" if "train" in to_plot else "epochs_val_"
    epoch_cols = [c for c in df.columns if epochs_str in c]
    epoch_cols.sort(key=lambda x: int(x.split("_")[-1]))
    xs = [series_to_array(df[c]) for c in epoch_cols]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_str = "tokens_seen_train_" if "train" in to_plot else "tokens_seen_val_"
    tokens_cols = [c for c in df.columns if tokens_str in c]
    tokens_cols.sort(key=lambda x: int(x.split("_")[-1]))
    xs = [series_to_array(df[c]) for c in tokens_cols]
    return interpolate_linearly(xs, arrays)


def interpolate_linearly(
        xs: list[np.ndarray], ys: list[np.ndarray], num_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Determine the maximum x value across all datasets
    max_x = max(x_vals.max() for x_vals in xs)
    
    # Generate a single set of new x values for all datasets
    new_x_vals = np.linspace(0, max_x, num_samples)

    new_ys = []
    for x_vals, y_vals in zip(xs, ys):
        # Interpolate y to the common set of new x values
        new_y_vals = np.interp(new_x_vals, x_vals, y_vals)
        new_ys.append(new_y_vals)

    # Convert new_ys to a 2D numpy array for easy manipulation
    new_ys = np.array(new_ys)
    
    # Calculate the average y values across all datasets
    avg_ys = np.nanmean(new_ys, axis=0)

    return new_x_vals, new_ys, avg_ys



def plot_loss_curves_llm_single_attn(
        file: str,
        attn_type: str,
        feature_map_qkv: str | None = None,
        feature_map_attn: str | None = None,
        use_out_proj: bool | None = None,
        identity_weight: float | None = None,
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
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
        plot_over=plot_over,
    )

    plt.figure(figsize=(8, 6))
    for y in ys:
        plt.plot(xs, y, "k", alpha=0.1)
    plt.plot(xs, avg_ys, "r", label="Average")
    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
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
        # Load the unique combinations of the targets
        combinations = (
            pl.scan_csv(file)
            .filter(pl.col("attn_type") == attn_type)
            .select(*[pl.col(target) for target in targets])
            .collect()
            .unique()
        )
        # Sort combinations alphabetically by content, target by target (for consistency in plotting)
        for target in targets:
            combinations = combinations.sort(target)
        # Create a list of settings
        for features in zip(
                *[combinations[target] for target in targets]
        ):
            settings.append((attn_type,) + tuple(features))
    return settings


def plot_llm_1500_steps_by_norm_position(
        file: str = "../results/results_llm_1500_steps.csv",
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
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
            plot_over=plot_over,
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
    plt.xlabel(plot_over)
    plt.ylabel("Loss")
    plt.title(f"Average {to_plot}")
    plt.legend()
    plt.grid()
    plt.show()


def plot_llm_1000_steps_100_tries_by_norm_position(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
        attn_type: str | None  = "vanilla",
        show_all_plots: bool = False,
        from_step: int = 0,
        save: bool = False,
        logit_scalar: str = "sqrt_dh",
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
            logit_scalar=logit_scalar,
            to_plot=to_plot,
            plot_over=plot_over,
        )
        label = f"{attn_type}"
        if use_x_norm:
            label += " x-norm"
        if use_qkv_norm:
            label += " qkv-norm"
        mask = xs >= from_step
        if show_all_plots:
            for y in ys:
                plt.plot(xs[mask], y[mask], color, alpha=0.3/math.sqrt(len(ys)))
        plt.plot(xs[mask], avg_y[mask], color=color, label=label, linewidth=2)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.title(f"Average {to_plot}")
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(13, 7)
    plt.tight_layout()
    plt.grid()

    if save:
        name = f"{to_plot}_{attn_type}_" + file.split("/")[-1].split(".")[0] + "_from_step_" + str(from_step)
        os.makedirs("../results/plots", exist_ok=True)
        plt.savefig(f"../results/plots/{name}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    close_plt()


def plot_llm_1000_steps_100_tries_by_norm_position_multiplot(
        file: str = "../results/results_llm_1000_steps_100_tries_sqrt_dh.csv",
        to_plot_set: list[str] | str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
        attn_type: str | None  = "vanilla",
        show_all_plots: bool = False,
        from_step_set: list[str] | int = 0,
        save: bool = False,
        logit_scalar: str = "sqrt_dh",
) -> None:
    if isinstance(to_plot_set, str):
        to_plot_set = [to_plot_set]
    if isinstance(from_step_set, int):
        from_step_set = [from_step_set]

    num_plots = int(len(to_plot_set) * len(from_step_set))
    nrows = len(to_plot_set)
    ncols = math.ceil(num_plots / nrows)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

    row = col = 0
    for i, (to_plot, from_step) in enumerate(itertools.product(to_plot_set, from_step_set)):
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
                logit_scalar=logit_scalar,
                to_plot=to_plot,
                plot_over=plot_over,
            )
            label = f"{attn_type}"
            if use_x_norm:
                label += " x-norm"
            if use_qkv_norm:
                label += " qkv-norm"
            mask = xs >= from_step
            if show_all_plots:
                for y in ys:
                    axs[row, col].plot(xs[mask], y[mask], color, alpha=0.3/math.sqrt(len(ys)))
            axs[row, col].plot(xs[mask], avg_y[mask], color=color, label=label, linewidth=2)

        if row == nrows - 1:
            axs[row, col].set_xlabel(plot_over)
        axs[row, col].set_ylabel(to_plot)
        axs[row, col].grid()

        to_next_row = ((i != 0) or (ncols == 1)) and ((i+1) % ncols == 0)
        if to_next_row:
            row += 1
            col = 0
        else:
            col += 1

    fig.set_size_inches(13, 9)
    axs[0, 0].legend()#ncol=2, loc='lower center', bbox_to_anchor=(0.5, -1.0))
    plt.tight_layout()

    if save:
        name = f"multiplot_{'_'.join(to_plot_set)}_{attn_type}_" + file.split("/")[-1].split(".")[0] + f"_from_step_{'_'.join(str(from_step) for from_step in from_step_set)}"
        os.makedirs("../results/plots", exist_ok=True)
        plt.savefig(f"../results/plots/{name}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()
    close_plt()


def plot_llm_1000_steps_100_tries_by_norm_position_single_setting(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        attn_type: str = "vanilla",
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
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
        plot_over=plot_over,
    )
    if show_all_plots:
        for y in ys:
            plt.plot(xs, y, alpha=0.3, color="pink")
    plt.plot(xs, avg_y, label=f"{attn_type}: mean", color="red", linewidth=2)
    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.title(f"Average {to_plot}")
    plt.grid()
    plt.show()


def plot_metric_variance(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        attn_type: str = "vanilla",
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
        from_step: int = 0,
        save: bool = False,
) -> None:
    settings = get_unique_settings(
        file, 
        targets=["use_x_norm", "use_qkv_norm"], 
        attn_type=attn_type
    )
    colors = generate_distinct_colors(len(settings))
    for (attn_type, use_x_norm, use_qkv_norm), color in zip(settings, colors):
        xs, ys, avg_y = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            logit_scalar="sqrt_dh",
            to_plot=to_plot,
            plot_over=plot_over,
        )
        label = f"{attn_type}"
        if use_x_norm:
            label += " x-norm"
        if use_qkv_norm:
            label += " qkv-norm"
        plt.plot(xs[xs >= from_step], np.std(ys, axis=0)[xs >= from_step], label=label, color=color)
    plt.xlabel(plot_over)
    plt.ylabel("Standard deviation")
    plt.title(f"Standard deviation of {to_plot}")

    plt.legend()
    
    fig = plt.gcf()
    fig.set_size_inches(13, 7)
    plt.tight_layout()
    plt.grid()

    if save:
        name = f"std_dev_{to_plot}_{attn_type}_" + file.split("/")[-1].split(".")[0] + "_from_step_" + str(from_step)
        os.makedirs("../results/plots", exist_ok=True)
        plt.savefig(f"../results/plots/{name}.png", dpi=300, bbox_inches="tight")
    else:
        plt.show()

    close_plt()


def get_loss_acc_correlation(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        attn_type: str = "vanilla",
        train: bool = False,
        from_step: int = 0,
        verbose: bool = True,
        logit_scalar: str = "sqrt_dh",
        plot_over: Literal["step", "epoch", "token"] = "step",
) -> dict[str, float]:
    settings = get_unique_settings(
        file, 
        targets=["use_x_norm", "use_qkv_norm"], 
        attn_type=attn_type
    )
    if verbose:
        print(f"Correlation between loss and accuracy for {attn_type}")
    correlations = {}
    for (attn_type, use_x_norm, use_qkv_norm) in settings:
        xsl, _, avg_loss = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            logit_scalar=logit_scalar,
            to_plot=("train" if train else "val") + "_loss",
            plot_over=plot_over,
        )
        xsa, _, avg_acc = load_xs_ys_avg_y(
            file=file,
            attn_type=attn_type,
            use_x_norm=use_x_norm,
            use_qkv_norm=use_qkv_norm,
            logit_scalar="sqrt_dh",
            to_plot=("train" if train else "val") + "_acc",
            plot_over=plot_over,
        )
        assert (xsl == xsa).all()
        mask = xsl >= from_step
        avg_loss = avg_loss.max() - avg_loss  # Invert loss to make it line up with accuracy
        avg_loss = avg_loss[mask]
        avg_acc = avg_acc[mask]

        correlation = pearsonr(avg_loss, avg_acc)[0]
        label = f"{attn_type}"
        if use_x_norm:
            label += " x-norm"
        if use_qkv_norm:
            label += " qkv-norm"
        correlations[label] = correlation

    # Sort correlations by value
    correlations = dict(sorted(correlations.items(), key=lambda item: item[1], reverse=True))
    if verbose:
        for label, correlation in correlations.items():
            print(f"- {correlation:.4f} ({label})")
    
    return correlations


def plot_correlations(
        file: str = "../results/results_llm_1000_steps_100_tries.csv",
        attn_type: str = "vanilla",
        logit_scalar: str = "sqrt_dh",
        from_step_list: list[int] | None = None,
) -> None:
    if from_step_list is None:
        from_step_list = [0, 800]

    results = {}

    for train in (True, False):
        for from_step in from_step_list:
            correlations = get_loss_acc_correlation(
                file=file,
                attn_type=attn_type,
                train=train,
                from_step=from_step,
                verbose=False,
                logit_scalar=logit_scalar,
            )
            for label, correlation in correlations.items():
                if label not in results:
                    results[label] = {"train": [train], "from_step": [from_step], "correlation": [correlation]}
                else:
                    results[label]["train"].append(train)
                    results[label]["from_step"].append(from_step)
                    results[label]["correlation"].append(correlation)

    # Make and print a table from this - each column is a 'label' and each row is a 'train' and 'from_step' combination
    # The table is then sorted by the correlation values
    table1 = []
    table2 = []
    for label, data in results.items():
        for train, from_step, correlation in zip(data["train"], data["from_step"], data["correlation"]):
            if train:
                table1.append([label, train, from_step, round(correlation, 4)])
            else:
                table2.append([label, train, from_step, round(correlation, 4)])

    def print_table(table):
        table = sorted(table, key=lambda x: x[3], reverse=True)
        print("Correlation table:")
        columns = ["attn_type", "train", "from_step", "correlation"]
        column_widths = [max(max(len(str(row[i])) for row in table), len(columns[i])) for i in range(len(columns))]

        column_str = "| "
        for i, column in enumerate(columns):
            column_str += column.ljust(column_widths[i]) + " | "
        print(column_str)
        print(f"|{'---|' * len(columns)}")
        for row in table:
            print("| ", end="")
            for i, value in enumerate(row):
                print(str(value).ljust(column_widths[i]), end=" | ")
            print()

    print_table(table1)
    print()
    print_table(table2)


def plot_loss_curves_avg_contrast_1500_steps(
        file: str = "../results/results_llm_1500_steps.csv",
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
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
            plot_over=plot_over,
            feature_map_attn=feature_map_attn,
            feature_map_qkv=feature_map_qkv,
        )
        plt.plot(xs, avg_y, color=color, label=f"{attn_type} ({feature_map_qkv}-{feature_map_attn})")
        for y in ys:
            plt.plot(xs, y, color, alpha=0.2)
    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.title(f"Average {to_plot}")
    plt.legend()
    plt.show()
    

def plot_loss_curves_feature_maps(
        attn_type: str,
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
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
            plot_over=plot_over,
        )
        label = f"{attn_type}"
        if attn_type not in ("identity", "vanilla"):
            label += f" ({feature_map_qkv}-{feature_map_attn})"
        plt.plot(xs, avg_y, label=label, color=color)

        for y in ys:
            plt.plot(xs, y, alpha=0.1, color=color)
    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
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
    to_plot_list = ["val_pplx", "val_loss", "val_acc"]
    from_step_list = [0]
    attn_types_list = ["vanilla"]
    save = False
    file_1000 = "../results/results_llm_1000_steps_100_tries_sqrt_dh.csv"
    file_10e = "../results/results_llm_10_epochs_10_tries_sqrt_dh.csv"

    # plot_llm_1000_steps_100_tries_by_norm_position_multiplot(
    #     file=file_10e,
    #     to_plot_set=["train_loss", "val_loss", "train_acc", "val_acc"],
    #     attn_type="vanilla",
    #     show_all_plots=False,
    #     from_step_set=[0, 800],
    #     save=save,
    #     logit_scalar="sqrt_dh",
    #     plot_over="token",
    # )

    for to_plot, from_step, attn_type in itertools.product(to_plot_list, from_step_list, attn_types_list):
        print(f"Plotting {to_plot} for {attn_type} from step {from_step}")
        plot_llm_1000_steps_100_tries_by_norm_position(
            file=file_1000,
            attn_type=attn_type, 
            to_plot=to_plot,
            plot_over="token",
            show_all_plots=False,
            from_step=from_step,
            save=save,
            logit_scalar="sqrt_dh" if attn_type == "vanilla" else None,
        )
        # print(f"Plotting variance of {to_plot} for {attn_type} from step {from_step}\n")
        # plot_metric_variance(file=file_1000, to_plot=to_plot, from_step=from_step, save=save)
    # for to_plot in ("val_loss", "train_loss"):
    #     print(f"Plotting {to_plot} for vanilla from step 0")
    #     get_loss_acc_correlation(file=file_1000, attn_type="vanilla", train="train" in to_plot, from_step=800)
    # plot_correlations(file=file_10e, attn_type="vanilla", from_step_list=[0, 800], logit_scalar="sqrt_dh")
