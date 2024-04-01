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


def close_plt() -> None:
    plt.cla()
    plt.clf()
    plt.close()


def series_to_array(series: pl.Series) -> np.ndarray:
    try:
        return np.array(ast.literal_eval(series[0]))
    except SyntaxError:
        return np.array(ast.literal_eval(series))


def load_xs_ys_avg_y(
        file: str,
        linear: bool,
        use_x_norm: bool | None = None,
        use_qk_norm: bool | None = None,
        residual_depth: int | None = None,
        model_scale: float | None = None,
        model_scale_method: Literal["depth", "width", "both"] | None = None,
        depth: int | None = None,
        width: int | None = None,
        num_params: int | None = None,
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token"] = "step",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load x, y, and average y from a CSV file."""
    assert to_plot in ("train_loss", "train_acc", "val_loss", "val_acc", "val_pplx"), f"Invalid to_plot: {to_plot}"

    filters = (pl.col("linear") == linear)
    if use_x_norm is not None:
        filters &= (pl.col("use_x_norm") == use_x_norm)
    if use_qk_norm is not None:
        filters &= (pl.col("use_qk_norm") == use_qk_norm)
    if residual_depth is not None:
        filters &= (pl.col("residual_depth") == residual_depth)
    if model_scale is not None:
        filters &= (pl.col("model_scale") == model_scale)
    if model_scale_method is not None:
        filters &= (pl.col("model_scale_method") == model_scale_method)
    if depth is not None:
        filters &= (pl.col("depth") == depth)
    if width is not None:
        filters &= (pl.col("width") == width)
    if num_params is not None:
        filters &= (pl.col("num_params") == num_params)

    df = pl.scan_csv(file).filter(filters).collect()
    df.sort("run_num")
    arrays = [series_to_array(df[to_plot][run_num]) for run_num in df["run_num"].unique()]

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
    epochs_str = "epochs_train" if "train" in to_plot else "epochs_val"
    xs = [series_to_array(df[epochs_str][run_num]) for run_num in df["run_num"].unique()]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_str = "tokens_seen_train" if "train" in to_plot else "tokens_seen_val"
    xs = [series_to_array(df[tokens_str][run_num]) for run_num in df["run_num"].unique()]
    return interpolate_linearly(xs, arrays)


def interpolate_linearly(
        xs: list[np.ndarray], ys: list[np.ndarray], num_samples: int = 500,
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


def get_unique_settings(file: str, targets: list[str]) -> list[str | int | float | bool]:
    settings = []
    
    # Load the unique combinations of the targets
    combinations = (
        pl.scan_csv(file)
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
        settings.append(tuple(features))

    return settings


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


def plot_results_compare_norms(
        file: str,
        depth: int | None = None,
        width: int | None = None,
        model_scale: float | None = None,
        model_scale_method: Literal["depth", "width", "both"] | None = None,
        to_plot: str = "val_pplx",
        plot_over: Literal["step", "epoch", "token"] = "epoch",
        plot_all: bool = False,
) -> None:
    assert (
        (depth is not None and width is not None)
        or (model_scale is not None and model_scale_method is not None)
    ), "Must specify depth & width or model scale & method"

    settings = get_unique_settings(file, ["use_qk_norm", "linear"])
    colors = generate_distinct_colors(len(settings))

    for color, (use_qk_norm, linear) in zip(colors, settings, strict=True):
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file,
            linear=linear,
            use_x_norm=True,
            use_qk_norm=use_qk_norm,
            depth=depth,
            width=width,
            model_scale=model_scale,
            model_scale_method=model_scale_method,
            to_plot=to_plot,
            plot_over=plot_over,
        )

        if plot_all:
            for y in ys:
                plt.plot(xs, y, color=color, alpha=0.1)

        plt.plot(xs, avg_ys, color=color, label=f"{use_qk_norm=}, {linear=}")

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.tight_layout()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    plot_results_compare_norms(
        "../results/results_v040_1000_steps_10_tries_sqrt_dh.csv",
        depth=8,
        width=384,
    )
