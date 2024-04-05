"""Evaluate results."""

import ast
import itertools
from typing import Literal

import polars as pl
import matplotlib.pyplot as plt
import numpy as np 
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
        embedding_type: Literal["learned", "rotary"] | None = None,
        to_plot: str = "val_loss",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "step",
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
    if embedding_type is not None:
        filters &= (pl.col("embedding_type") == embedding_type)

    df = pl.scan_csv(file).filter(filters).collect()
    df.sort("run_num")
    arrays = [series_to_array(df[to_plot][i]) for i in range(len(df[to_plot]))]

    if plot_over == "step":
        return load_steps_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "epoch":
        return load_epochs_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "token":
        return load_tokens_ys_avg_ys(df, arrays, to_plot)
    elif plot_over == "time_sec":
        return load_time_ys_avg_ys(df, arrays, to_plot)
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
    xs = [series_to_array(df[epochs_str][i]) for i in range(len(df[epochs_str]))]
    return interpolate_linearly(xs, arrays)


def load_tokens_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tokens_str = "tokens_seen_train" if "train" in to_plot else "tokens_seen_val"
    xs = [series_to_array(df[tokens_str][i]) for i in range(len(df[tokens_str]))]
    return interpolate_linearly(xs, arrays)


def load_time_ys_avg_ys(
        df: pl.DataFrame,
        arrays: list[np.ndarray],
        to_plot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert "val" in to_plot, "Only validation data has time data"
    time_str = "cumulative_time"
    xs = [series_to_array(df[time_str][i]) for i in range(len(df[time_str]))]
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
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
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
    close_plt()


def plot_results_compare_depth_width(
        file: str,
        linear: bool = False,
        use_x_norm: bool = True,
        use_qk_norm: bool = False,
        to_plot: str = "val_pplx",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
        plot_all: bool = False,
) -> None:
    settings = get_unique_settings(file, ["depth", "width"])
    colors = generate_distinct_colors(len(settings))

    for color, (depth, width) in zip(colors, settings, strict=True):
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file,
            depth=depth,
            width=width,
            linear=linear,
            use_x_norm=use_x_norm,
            use_qk_norm=use_qk_norm,
            to_plot=to_plot,
            plot_over=plot_over,
        )

        if plot_all:
            for y in ys:
                plt.plot(xs, y, color=color, alpha=0.1)

        plt.plot(xs, avg_ys, color=color, label=f"{depth=}, {width=}")

    plt.title(f"Depth vs. Width ({linear=}, {use_x_norm=}, {use_qk_norm=})")
    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    close_plt()


def plot_results_compare_norms_and_parameters(
        file: str,
        num_param_counts: int = 3,
        use_qk_norm: bool | None = None,
        linear: bool | None = None,
        to_plot: str = "val_pplx",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
        plot_all: bool = False,
) -> None:
    assert num_param_counts > 1, (
        "You want to compare by parameter count or what??? "
        f"{num_param_counts} is not enough! Use at least 2."
    )

    settings_targets = []
    if use_qk_norm is None:
        settings_targets.append("use_qk_norm")
    if linear is None:
        settings_targets.append("linear")
    
    settings = get_unique_settings(file, settings_targets)
    if use_qk_norm is not None:
        settings = [(use_qk_norm, s[0]) for s in settings]
    if linear is not None:
        settings = [(s[0], linear) for s in settings]

    colors = generate_distinct_colors(len(settings))

    nums_params = pl.scan_csv(file).select("num_params").collect().unique().sort("num_params")["num_params"]
    # Pick num_param_counts evenly spaced numbers from the list of unique num_params; must be contained in num_params!
    nums_params_indices = np.linspace(0, len(nums_params)-1, num_param_counts).astype(int).tolist()
    nums_params = [nums_params[i] for i in nums_params_indices]

    linestyles = itertools.cycle(("-", "--", "-.", ":"))
    for num_params, linestyle in zip(nums_params, linestyles, strict=False):
        print(f"\n{num_params=}")
        for color, (use_qk_norm_, linear_) in zip(colors, settings, strict=True):
            print(f"{use_qk_norm_=}, {linear_=}")
            xs, ys, avg_ys = load_xs_ys_avg_y(
                file,
                use_qk_norm=use_qk_norm_,
                linear=linear_,
                to_plot=to_plot,
                num_params=num_params,
                plot_over=plot_over,
            )

            if plot_all:
                for y in ys:
                    plt.plot(xs, y, color=color, linestyle=linestyle, alpha=0.1)

            label = f"{num_params=}"
            if use_qk_norm is None and use_qk_norm_:
                label += ", qk_norm"
            if linear is None:
                label += ", linear value" if linear_ else ", nonlinear value"

            plt.plot(xs, avg_ys, color=color, linestyle=linestyle, label=label)

    title = f"Number of Parameters vs. {to_plot}"
    if linear is not None:
        title += " (linear value)" if linear else " (nonlinear value)"
    if use_qk_norm is not None:
        title += f" ({use_qk_norm=})"
    plt.title(title)

    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    close_plt()


def plot_results_compare_norms_and_embedding_type(
        file: str,
        depth: int | None = None,
        width: int | None = None,
        model_scale: float | None = None,
        model_scale_method: Literal["depth", "width", "both"] | None = None,
        linear: bool = False,
        to_plot: str = "val_pplx",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
        plot_all: bool = False,
        loglog: bool = False,
) -> None:
    assert (
        (depth is not None and width is not None)
        or (model_scale is not None and model_scale_method is not None)
    ), "Must specify depth & width or model scale & method"

    settings = get_unique_settings(file, ["use_qk_norm", "embedding_type"])
    colors = generate_distinct_colors(len(settings))

    for color, (use_qk_norm, embedding_type) in zip(colors, settings, strict=True):
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file,
            embedding_type=embedding_type,
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
                if loglog:
                    plt.loglog(xs, y, color=color, alpha=0.1)
                else:
                    plt.plot(xs, y, color=color, alpha=0.1)

        if loglog:
            plt.loglog(xs, avg_ys, color=color, label=f"{use_qk_norm=}, {embedding_type=}")
        else:
            plt.plot(xs, avg_ys, color=color, label=f"{use_qk_norm=}, {embedding_type=}")

    scales = pl.scan_csv(file).filter(
        (pl.col("linear") == linear)
        & (pl.col("use_x_norm") == True)
        & (pl.col("use_qk_norm") == use_qk_norm)
        & ((pl.col("depth") == depth) if depth is not None else (pl.col("model_scale") == model_scale))
        & ((pl.col("width") == width) if width is not None else (pl.col("model_scale_method") == model_scale_method))
    ).select("depth", "width").collect()
    num_blocks, width = scales["depth"][0], scales["width"][0]

    plt.title(f"{num_blocks=}, {width=}")
    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()
    close_plt(
)
    

def plot_results_compare_norms_scale(
        file: str,
        embedding_type: Literal["learned", "rotary"] = "rotary",
        linear: bool = False,
        use_x_norm: bool = True,
        model_scale_method: Literal["depth", "width", "both"] | None = None,
        to_plot: str = "val_pplx",
        plot_over: Literal["step", "epoch", "token", "time_sec"] = "epoch",
        plot_all: bool = False,
        loglog: bool = False,
) -> None:
    settings_targets = ["use_qk_norm", "model_scale"]
    if model_scale_method is None:
        settings_targets.append("model_scale_method")
    settings = get_unique_settings(file, settings_targets)
    if model_scale_method is not None:
        settings = [(use_qk_norm, model_scale, model_scale_method) for use_qk_norm, model_scale in settings]
    colors = generate_distinct_colors(len(settings))

    for color, (use_qk_norm, model_scale, model_scale_method) in zip(colors, settings, strict=True):
        xs, ys, avg_ys = load_xs_ys_avg_y(
            file,
            embedding_type=embedding_type,
            linear=linear,
            use_x_norm=use_x_norm,
            use_qk_norm=use_qk_norm,
            model_scale=model_scale,
            model_scale_method=model_scale_method,
            to_plot=to_plot,
            plot_over=plot_over,
        )

        if plot_all:
            for y in ys:
                if loglog:
                    plt.loglog(xs, y, color=color, alpha=0.1)
                else:
                    plt.plot(xs, y, color=color, alpha=0.1)

        scales = pl.scan_csv(file).filter(
            (pl.col("linear") == linear)
            & (pl.col("use_x_norm") == True)
            & (pl.col("use_qk_norm") == use_qk_norm)
            & (pl.col("embedding_type") == embedding_type)
            & (pl.col("model_scale") == model_scale)
            & (pl.col("model_scale_method") == model_scale_method)
        ).select("depth", "width").collect()
        num_blocks, width = scales["depth"][0], scales["width"][0]

        label = f"{num_blocks=}, {width=}, {use_qk_norm=}"
        if loglog:
            plt.loglog(xs, avg_ys, color=color, label=label)
        else:
            plt.plot(xs, avg_ys, color=color, label=label)

    plt.title(f"{embedding_type=}")
    plt.xlabel(plot_over)
    plt.ylabel(to_plot)
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()



if __name__ == "__main__":
    file1000steps = "../results/results_v040_1000_steps_10_tries_sqrt_dh.csv"
    # UNIQUE DEPTHS & WIDTHS
    # depth: 4, 5, 8, 13, 16
    # width: 256, 384, 512, 640

    # UNIQUE COMBINATIONS
    # ┌───────┬───────┐
    # │ depth ┆ width │
    # │ ---   ┆ ---   │
    # │ i64   ┆ i64   │
    # ╞═══════╪═══════╡
    # │ 4     ┆ 384   │
    # │ 5     ┆ 256   │
    # │ 8     ┆ 512   │
    # │ 8     ┆ 256   │
    # │ 8     ┆ 384   │
    # │ 13    ┆ 640   │
    # │ 16    ┆ 384   │
    # └───────┴───────┘

    # UNIQUE NUM_PARAMS
    # num_params: 27_805_189, 29_034_760, 42_320_260, 42_321_796, 46_009_736, 53_385_616, 64_623_112, 97_678_093

    # for depth in (4, 8, 16):
    #     plot_results_compare_norms(
    #         file=file1000steps,
    #         depth=depth,
    #         width=384,
    #         plot_over="token",
    #     )
    # plot_results_compare_norms(
    #     file=file1000steps,
    #     depth=13,
    #     width=384,
    #     plot_over="token",
    # )
    # plot_results_compare_depth_width(
    #     file=file1000steps,
    #     linear=True,
    #     use_x_norm=True,
    #     use_qk_norm=False,
    # )
    # plot_results_compare_norms_and_parameters(
    #     file=file1000steps,
    #     num_param_counts=3,
    #     linear=True,
    #     to_plot="val_loss",
    #     plot_over="token",
    #     plot_all=False,
    # )
    

    # file10epochs = "../results/results_v040_10_epochs_5_tries_sqrt_dh.csv"
    # plot_results_compare_norms(
    #     file=file10epochs,
    #     width=384,
    #     depth=8,
    #     plot_over="token",
    #     to_plot="val_pplx",
    # )

    file = "../results/results_v040_2_epochs_5_tries_all_norm_combos_depth_width_all_embs.csv"
    # plot_results_compare_norms_and_embedding_type(
    #     file=file,
    #     model_scale=1.0,
    #     model_scale_method="depth",
    #     plot_over="token",
    #     to_plot="val_loss",
    #     loglog=True,
    #     linear=True,
    # )

    plot_results_compare_norms_scale(
        file=file,
        model_scale_method="depth",
        to_plot="val_loss",
        plot_over="token",
        loglog=False,
    )

