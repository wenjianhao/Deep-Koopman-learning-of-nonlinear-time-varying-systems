"""
Create a tracking video for the nonlinear time-varying system example.

The animation compares the true trajectory with TVDMD and DKTV one-step
predictions saved by ``main.py`` / the paper result folders.

Examples
--------
python3 plot_tracking_video.py

python3 plot_tracking_video.py \
    --results-dir SavedResults/SavedResults08 \
    --basis-class LNNsl \
    --output SavedResults/tracking_gamma08.mp4

python3 plot_tracking_video.py --output SavedResults/tracking.gif
"""

import argparse
from pathlib import Path

try:
    import numpy as np
    from numpy import linalg as LA
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: numpy. Install the project dependencies first, "
        "for example: pip install numpy"
    ) from exc

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: matplotlib. Install the project dependencies first, "
        "for example: pip install matplotlib"
    ) from exc

try:
    from scipy.interpolate import CubicSpline
except ImportError:
    CubicSpline = None

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: torch. Install the project dependencies first, "
        "for example: pip install torch"
    ) from exc

try:
    import joblib
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: joblib. Install the project dependencies first, "
        "for example: pip install joblib matplotlib torch numpy"
    ) from exc

from LNN import LNN, LNNsl


def load_array(path):
    if not path.exists():
        raise FileNotFoundError(f"Could not find required result file: {path}")
    return joblib.load(path)


def make_basis_model(name, dim_input, dim_output):
    if name == "LNN":
        return LNN(dim_input=dim_input, dim_output=dim_output)
    if name == "LNNsl":
        return LNNsl(dim_input=dim_input, dim_output=dim_output)
    raise ValueError(f"Unknown basis class: {name}")


def run_basis(model, state):
    with torch.no_grad():
        tensor_state = torch.from_numpy(np.asarray(state).T).float()
        try:
            lifted = model.forward(tensor_state, Test=True)
        except TypeError:
            lifted = model.forward(tensor_state)
    return lifted.cpu().detach().numpy()


def build_dktv_predictions(results_dir, basis_class, prebatch_size, batch_size):
    xdata = load_array(results_dir / "xdata.pkl")
    tvdmd = load_array(results_dir / "tvdpred.pkl")
    astack = load_array(results_dir / "Astk.pkl")
    cstack = load_array(results_dir / "Cstk.pkl")

    dim_states = xdata.shape[0]
    dim_lifting = astack.shape[0]
    model = make_basis_model(basis_class, dim_states, dim_lifting)

    basis_dir = results_dir / "nnbasis"
    if not basis_dir.exists():
        raise FileNotFoundError(
            f"Could not find {basis_dir}. DKTV video output needs the saved "
            "lifting networks under nnbasis/."
        )

    predictions = []
    model_indices = []
    max_model_index = astack.shape[2] - 1
    last_state_index = xdata.shape[1] - 1

    for state_index in range(prebatch_size, last_state_index):
        model_index = int(np.floor((state_index - prebatch_size) / batch_size)) + 1
        if model_index > max_model_index:
            break

        checkpoint_path = basis_dir / f"{model_index}liftnetwork.pth"
        if not checkpoint_path.exists():
            break

        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        state_dict = checkpoint.get("model_lifting", checkpoint)
        model.load_state_dict(state_dict)
        model.eval()

        lifted = run_basis(model, xdata[:, state_index])
        pred = np.matrix(cstack[:, :, model_index]) @ np.matrix(
            astack[:, :, model_index]
        ) @ np.matrix(lifted).T
        predictions.append(np.asarray(pred.T).squeeze())
        model_indices.append(model_index)

    if not predictions:
        raise RuntimeError(
            "No DKTV predictions were reconstructed. Check --prebatch-size, "
            "--batch-size, --basis-class, and the nnbasis files."
        )

    dktv = np.asarray(predictions).T
    frames = min(dktv.shape[1], tvdmd.shape[1], xdata[:, prebatch_size + 1 :].shape[1])
    true = xdata[:, prebatch_size + 1 : prebatch_size + 1 + frames]

    return {
        "true": true,
        "tvdmd": tvdmd[:, :frames],
        "dktv": dktv[:, :frames],
        "model_indices": np.asarray(model_indices[:frames]),
        "start_step": prebatch_size + 1,
    }


def set_line(line, x, y):
    line.set_data(x, y)


def set_line_3d(line, x, y, z):
    line.set_data(x, y)
    line.set_3d_properties(z)


def axis_limits(*arrays, pad_ratio=0.08):
    values = np.concatenate([np.ravel(arr) for arr in arrays])
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    span = high - low
    pad = pad_ratio * span if span > 0 else 1.0
    return low - pad, high + pad


def infer_gamma(results_dir):
    name = results_dir.name.lower()
    if "08" in name or "0.8" in name:
        return 0.8
    return 6


def save_tracking_video(data, output, fps, dpi, stride, phase_tail, gamma):
    true = data["true"][:, ::stride]
    tvdmd = data["tvdmd"][:, ::stride]
    dktv = data["dktv"][:, ::stride]
    model_indices = data["model_indices"][::stride]
    steps = data["start_step"] + np.arange(data["true"].shape[1])[::stride]

    dktv_error = LA.norm(dktv - true, axis=0)
    tvdmd_error = LA.norm(tvdmd - true, axis=0)
    frames = true.shape[1]

    colors = {
        "true": "#d62728",
        "tvdmd": "#222222",
        "dktv": "#1f77b4",
        "muted": "#b7bec8",
        "cursor": "#7a8795",
    }
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#d4dae3",
            "axes.grid": True,
            "axes.labelcolor": "#20242a",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "font.size": 11,
            "grid.alpha": 0.22,
            "grid.color": "#8d99a6",
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d4dae3",
            "lines.solid_capstyle": "round",
            "xtick.color": "#424a55",
            "ytick.color": "#424a55",
        }
    )
    fig, (ax_phase, ax_err) = plt.subplots(
        2,
        1,
        figsize=(9.6, 10.2),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )

    xlim = axis_limits(true[0], tvdmd[0], dktv[0])
    ylim = axis_limits(true[1], tvdmd[1], dktv[1])
    true_phase, = ax_phase.plot(
        [],
        [],
        linestyle="-",
        marker="o",
        color=colors["true"],
        linewidth=1.5,
        markersize=4,
        alpha=0.55,
        label="True",
    )
    tvd_phase, = ax_phase.plot(
        [],
        [],
        linestyle=(0, (5, 3)),
        marker="s",
        color=colors["tvdmd"],
        linewidth=1.2,
        markersize=4,
        alpha=0.48,
        label="TVDMD",
    )
    dktv_phase, = ax_phase.plot(
        [],
        [],
        linestyle=(0, (7, 2)),
        marker="^",
        color=colors["dktv"],
        linewidth=1.3,
        markersize=4,
        alpha=0.55,
        label="DKTV",
    )
    tvd_gap, = ax_phase.plot(
        [], [], color=colors["tvdmd"], linestyle=":", linewidth=1.4, alpha=0.55
    )
    dktv_gap, = ax_phase.plot(
        [], [], color=colors["dktv"], linestyle=":", linewidth=1.4, alpha=0.70
    )
    true_dot, = ax_phase.plot([], [], "o", color=colors["true"], markersize=8)
    tvd_dot, = ax_phase.plot([], [], "s", color=colors["tvdmd"], markersize=6)
    dktv_dot, = ax_phase.plot([], [], "^", color=colors["dktv"], markersize=7)
    ax_phase.set_xlim(xlim)
    ax_phase.set_ylim(ylim)
    ax_phase.set_xlabel("$x_1$")
    ax_phase.set_ylabel("$x_2$")
    ax_phase.set_title("Phase Tracking")
    ax_phase.set_aspect("equal", adjustable="box")
    ax_phase.legend(loc="upper left", ncol=3, fontsize=10)
    ax_phase.text(
        0.985,
        0.955,
        (
            "True dynamics\n"
            rf"$\dot x_1=(1+{gamma:g}t)\cos(x_2)$" "\n"
            rf"$\dot x_2=-(1+{gamma:g}t)\cos(x_1)$"
        ),
        transform=ax_phase.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        color="#20242a",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d4dae3",
            "alpha": 0.92,
        },
    )

    step_xlim = (steps[0], steps[-1])
    err_lim = axis_limits(tvdmd_error, dktv_error)

    tvd_err_line, = ax_err.plot(
        [], [], color=colors["tvdmd"], linestyle=(0, (5, 3)), linewidth=2.0, label="TVDMD"
    )
    dktv_err_line, = ax_err.plot(
        [], [], color=colors["dktv"], linewidth=2.2, label="DKTV"
    )
    err_cursor = ax_err.axvline(
        steps[0], color=colors["cursor"], linewidth=1.0, alpha=0.65
    )
    ax_err.set_xlim(step_xlim)
    ax_err.set_ylim(err_lim)
    ax_err.set_xlabel("Iteration step")
    ax_err.set_ylabel("$\\ell_2$ error")
    ax_err.set_title("One-Step Prediction Error")
    ax_err.legend(loc="upper left", fontsize=10)

    status = ax_err.text(
        0.98,
        0.94,
        "",
        transform=ax_err.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        color="#20242a",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": "#d4dae3",
            "alpha": 0.94,
        },
    )

    def update(frame):
        end = frame + 1
        step_slice = steps[:end]
        start = max(0, end - phase_tail)

        set_line(true_phase, true[0, start:end], true[1, start:end])
        set_line(tvd_phase, tvdmd[0, start:end], tvdmd[1, start:end])
        set_line(dktv_phase, dktv[0, start:end], dktv[1, start:end])
        set_line(true_dot, true[0, frame : end], true[1, frame : end])
        set_line(tvd_dot, tvdmd[0, frame : end], tvdmd[1, frame : end])
        set_line(dktv_dot, dktv[0, frame : end], dktv[1, frame : end])
        set_line(
            tvd_gap,
            [true[0, frame], tvdmd[0, frame]],
            [true[1, frame], tvdmd[1, frame]],
        )
        set_line(
            dktv_gap,
            [true[0, frame], dktv[0, frame]],
            [true[1, frame], dktv[1, frame]],
        )

        set_line(tvd_err_line, step_slice, tvdmd_error[:end])
        set_line(dktv_err_line, step_slice, dktv_error[:end])
        err_cursor.set_xdata([steps[frame], steps[frame]])
        status.set_text(
            f"step {int(steps[frame])}   DKTV model {int(model_indices[frame])}   "
            f"DKTV error {dktv_error[frame]:.3f}   TVDMD error {tvdmd_error[frame]:.3f}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                "Matplotlib cannot find ffmpeg for MP4 output. Install ffmpeg "
                "or write a GIF instead with --output SavedResults/tracking.gif."
            )
        writer = FFMpegWriter(
            fps=fps,
            metadata={"title": "DKTV tracking"},
            codec="libx264",
            bitrate=4200,
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

    with writer.saving(fig, str(output), dpi=dpi):
        for frame in range(frames):
            update(frame)
            writer.grab_frame()

    plt.close(fig)


def prepare_animation_data(data, stride):
    true = data["true"][:, ::stride]
    tvdmd = data["tvdmd"][:, ::stride]
    dktv = data["dktv"][:, ::stride]
    model_indices = data["model_indices"][::stride]
    steps = data["start_step"] + np.arange(data["true"].shape[1])[::stride]

    return {
        "true": true,
        "tvdmd": tvdmd,
        "dktv": dktv,
        "model_indices": model_indices,
        "steps": steps,
        "dktv_error": LA.norm(dktv - true, axis=0),
        "tvdmd_error": LA.norm(tvdmd - true, axis=0),
    }


def phase_window_start(true, end, max_points, target_arc_length):
    start = max(0, end - max_points)
    if end - start <= 1:
        return start

    recent = true[:, start:end]
    segment_lengths = LA.norm(np.diff(recent, axis=1), axis=0)
    arc_length = 0.0
    for offset, segment_length in enumerate(segment_lengths[::-1], start=1):
        arc_length += float(segment_length)
        if arc_length >= target_arc_length:
            return end - offset - 1
    return start


def smooth_trajectory(x, y, z, samples_per_segment=8):
    if len(z) < 4 or CubicSpline is None:
        return x, y, z

    dense_count = max(len(z) * samples_per_segment, len(z))
    dense_z = np.linspace(float(z[0]), float(z[-1]), dense_count)
    return (
        CubicSpline(z, x)(dense_z),
        CubicSpline(z, y)(dense_z),
        dense_z,
    )


def smooth_xy_over_steps(x, y, steps, samples_per_segment=8):
    if len(steps) < 4 or CubicSpline is None:
        return x, y

    dense_count = max(len(steps) * samples_per_segment, len(steps))
    dense_steps = np.linspace(float(steps[0]), float(steps[-1]), dense_count)
    return (
        CubicSpline(steps, x)(dense_steps),
        CubicSpline(steps, y)(dense_steps),
    )


def make_writer(output, fps):
    suffix = output.suffix.lower()
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            "Matplotlib cannot find ffmpeg for MP4 output. Install ffmpeg "
            "or write a GIF instead with --output SavedResults/tracking.gif."
        )
    return FFMpegWriter(
        fps=fps,
        metadata={"title": "DKTV gamma comparison"},
        codec="libx264",
        bitrate=5200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )


def add_true_dynamics_text(ax, gamma):
    text_fn = ax.text2D if getattr(ax, "name", "") == "3d" else ax.text
    text_fn(
        0.985,
        0.955,
        (
            "True dynamics\n"
            rf"$\dot x_1=(1+{gamma:g}t)\cos(x_2)$" "\n"
            rf"$\dot x_2=-(1+{gamma:g}t)\cos(x_1)$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#20242a",
        bbox={
            "boxstyle": "round,pad=0.32",
            "facecolor": "white",
            "edgecolor": "#d4dae3",
            "alpha": 0.92,
        },
    )


def save_gamma_comparison_video(
    output, fps, dpi, stride, phase_tail, target_arc_length
):
    cases = [
        {
            "title": r"$\gamma=0.8$   slow time-varying dynamics",
            "gamma": 0.8,
            "results_dir": Path("SavedResults/SavedResults08"),
            "basis_class": "LNNsl",
        },
        {
            "title": r"$\gamma=6$   fast time-varying dynamics",
            "gamma": 6,
            "results_dir": Path("SavedResults/Paper_Plots_data"),
            "basis_class": "LNN",
        },
    ]
    for case in cases:
        case["data"] = prepare_animation_data(
            build_dktv_predictions(
                case["results_dir"],
                case["basis_class"],
                prebatch_size=100,
                batch_size=10,
            ),
            stride,
        )

    frames = max(case["data"]["true"].shape[1] for case in cases)

    colors = {
        "true": "#d62728",
        "tvdmd": "#222222",
        "dktv": "#1f77b4",
        "cursor": "#7a8795",
    }
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#d4dae3",
            "axes.grid": True,
            "axes.labelcolor": "#20242a",
            "axes.labelsize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.titlesize": 14,
            "font.size": 11,
            "grid.alpha": 0.22,
            "grid.color": "#8d99a6",
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d4dae3",
            "lines.solid_capstyle": "round",
            "xtick.color": "#424a55",
            "ytick.color": "#424a55",
        }
    )

    fig, axs = plt.subplots(
        2,
        2,
        figsize=(15.2, 9.0),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.45, 1.0]},
    )

    artists = []
    for col, case in enumerate(cases):
        data = case["data"]
        true = data["true"]
        tvdmd = data["tvdmd"]
        dktv = data["dktv"]
        steps = data["steps"]
        tvdmd_error = data["tvdmd_error"]
        dktv_error = data["dktv_error"]

        ax_phase = axs[0, col]
        ax_err = axs[1, col]
        xlim = axis_limits(true[0], tvdmd[0], dktv[0])
        ylim = axis_limits(true[1], tvdmd[1], dktv[1])
        err_lim = axis_limits(tvdmd_error, dktv_error)

        true_phase, = ax_phase.plot(
            [],
            [],
            "-",
            color=colors["true"],
            linewidth=2.1,
            alpha=0.68,
            label="True",
        )
        tvd_phase, = ax_phase.plot(
            [],
            [],
            color=colors["tvdmd"],
            linestyle=(0, (5, 3)),
            linewidth=1.7,
            alpha=0.58,
            label="TVDMD",
        )
        dktv_phase, = ax_phase.plot(
            [],
            [],
            color=colors["dktv"],
            linestyle=(0, (7, 2)),
            linewidth=1.9,
            alpha=0.66,
            label="DKTV",
        )
        tvd_gap, = ax_phase.plot(
            [], [], color=colors["tvdmd"], linestyle=":", linewidth=1.3, alpha=0.55
        )
        dktv_gap, = ax_phase.plot(
            [], [], color=colors["dktv"], linestyle=":", linewidth=1.3, alpha=0.70
        )
        true_dot, = ax_phase.plot([], [], "o", color=colors["true"], markersize=8)
        tvd_dot, = ax_phase.plot([], [], "s", color=colors["tvdmd"], markersize=6)
        dktv_dot, = ax_phase.plot([], [], "^", color=colors["dktv"], markersize=7)

        ax_phase.set_title(case["title"])
        ax_phase.set_xlim(xlim)
        ax_phase.set_ylim(ylim)
        ax_phase.set_xlabel("$x_1$")
        ax_phase.set_ylabel("$x_2$")
        ax_phase.set_aspect("equal", adjustable="box")
        ax_phase.legend(loc="upper left", ncol=3, fontsize=9)
        add_true_dynamics_text(ax_phase, case["gamma"])

        tvd_err_line, = ax_err.plot(
            [],
            [],
            color=colors["tvdmd"],
            linestyle=(0, (5, 3)),
            linewidth=1.8,
            label="TVDMD",
        )
        dktv_err_line, = ax_err.plot(
            [], [], color=colors["dktv"], linewidth=2.0, label="DKTV"
        )
        err_cursor = ax_err.axvline(
            steps[0], color=colors["cursor"], linewidth=1.0, alpha=0.65
        )
        ax_err.set_xlim((steps[0], steps[-1]))
        ax_err.set_ylim(err_lim)
        ax_err.set_xlabel("Iteration step")
        ax_err.set_ylabel("$\\ell_2$ error")
        ax_err.set_title("One-Step Prediction Error")
        ax_err.legend(loc="upper left", fontsize=9)
        status = ax_err.text(
            0.98,
            0.94,
            "",
            transform=ax_err.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#20242a",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d4dae3",
                "alpha": 0.94,
            },
        )

        artists.append(
            {
                "data": data,
                "true_phase": true_phase,
                "tvd_phase": tvd_phase,
                "dktv_phase": dktv_phase,
                "tvd_gap": tvd_gap,
                "dktv_gap": dktv_gap,
                "true_dot": true_dot,
                "tvd_dot": tvd_dot,
                "dktv_dot": dktv_dot,
                "tvd_err_line": tvd_err_line,
                "dktv_err_line": dktv_err_line,
                "err_cursor": err_cursor,
                "status": status,
            }
        )

    def update(frame):
        for artist in artists:
            data = artist["data"]
            n_frames = data["true"].shape[1]
            idx = min(frame, n_frames - 1)
            end = idx + 1
            true = data["true"]
            tvdmd = data["tvdmd"]
            dktv = data["dktv"]
            steps = data["steps"]
            tvdmd_error = data["tvdmd_error"]
            dktv_error = data["dktv_error"]
            start = phase_window_start(
                true,
                end,
                max_points=phase_tail,
                target_arc_length=target_arc_length,
            )
            window_steps = steps[start:end]
            true_x, true_y = smooth_xy_over_steps(
                true[0, start:end], true[1, start:end], window_steps
            )
            tvd_x, tvd_y = smooth_xy_over_steps(
                tvdmd[0, start:end], tvdmd[1, start:end], window_steps
            )
            dktv_x, dktv_y = smooth_xy_over_steps(
                dktv[0, start:end], dktv[1, start:end], window_steps
            )

            set_line(artist["true_phase"], true_x, true_y)
            set_line(artist["tvd_phase"], tvd_x, tvd_y)
            set_line(artist["dktv_phase"], dktv_x, dktv_y)
            set_line(artist["true_dot"], true[0, idx:end], true[1, idx:end])
            set_line(artist["tvd_dot"], tvdmd[0, idx:end], tvdmd[1, idx:end])
            set_line(artist["dktv_dot"], dktv[0, idx:end], dktv[1, idx:end])
            set_line(
                artist["tvd_gap"],
                [true[0, idx], tvdmd[0, idx]],
                [true[1, idx], tvdmd[1, idx]],
            )
            set_line(
                artist["dktv_gap"],
                [true[0, idx], dktv[0, idx]],
                [true[1, idx], dktv[1, idx]],
            )

            step_slice = steps[:end]
            set_line(artist["tvd_err_line"], step_slice, tvdmd_error[:end])
            set_line(artist["dktv_err_line"], step_slice, dktv_error[:end])
            artist["err_cursor"].set_xdata([steps[idx], steps[idx]])
            artist["status"].set_text(
                f"step {int(steps[idx])}   DKTV {dktv_error[idx]:.3f}   "
                f"TVDMD {tvdmd_error[idx]:.3f}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(output, fps)
    with writer.saving(fig, str(output), dpi=dpi):
        for frame in range(frames):
            update(frame)
            writer.grab_frame()

    plt.close(fig)


def save_gamma_comparison_3d_video(output, fps, dpi, stride, trajectory_window):
    cases = [
        {
            "title": r"$\gamma=0.8$   slow time-varying dynamics",
            "gamma": 0.8,
            "results_dir": Path("SavedResults/SavedResults08"),
            "basis_class": "LNNsl",
        },
        {
            "title": r"$\gamma=6$   fast time-varying dynamics",
            "gamma": 6,
            "results_dir": Path("SavedResults/Paper_Plots_data"),
            "basis_class": "LNN",
        },
    ]
    for case in cases:
        case["data"] = prepare_animation_data(
            build_dktv_predictions(
                case["results_dir"],
                case["basis_class"],
                prebatch_size=100,
                batch_size=10,
            ),
            stride,
        )

    frames = max(case["data"]["true"].shape[1] for case in cases)
    colors = {
        "true": "#d62728",
        "tvdmd": "#222222",
        "dktv": "#1f77b4",
        "cursor": "#7a8795",
    }
    plt.rcParams.update(
        {
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.edgecolor": "#d4dae3",
            "font.size": 11,
            "grid.alpha": 0.22,
            "grid.color": "#8d99a6",
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#d4dae3",
        }
    )

    fig = plt.figure(figsize=(15.2, 7.8), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, index + 1, projection="3d") for index in range(2)]
    fig.suptitle(
        r"3D Tracking Comparison: $(x_1, x_2, k)$",
        fontsize=17,
        fontweight="bold",
    )

    artists = []
    for ax, case in zip(axes, cases):
        data = case["data"]
        true = data["true"]
        tvdmd = data["tvdmd"]
        dktv = data["dktv"]
        steps = data["steps"]

        xlim = axis_limits(true[0], tvdmd[0], dktv[0])
        ylim = axis_limits(true[1], tvdmd[1], dktv[1])
        zlim = (float(steps[0]), float(steps[-1]))

        true_line, = ax.plot(
            [],
            [],
            [],
            color=colors["true"],
            linewidth=2.6,
            alpha=0.82,
            label="True",
        )
        tvd_line, = ax.plot(
            [],
            [],
            [],
            color=colors["tvdmd"],
            linestyle=(0, (5, 3)),
            linewidth=2.0,
            alpha=0.70,
            label="TVDMD",
        )
        dktv_line, = ax.plot(
            [],
            [],
            [],
            color=colors["dktv"],
            linestyle=(0, (7, 2)),
            linewidth=2.2,
            alpha=0.78,
            label="DKTV",
        )
        tvd_gap, = ax.plot(
            [], [], [], color=colors["tvdmd"], linestyle=":", linewidth=1.6, alpha=0.65
        )
        dktv_gap, = ax.plot(
            [], [], [], color=colors["dktv"], linestyle=":", linewidth=1.6, alpha=0.80
        )
        true_dot, = ax.plot([], [], [], "o", color=colors["true"], markersize=8)
        tvd_dot, = ax.plot([], [], [], "s", color=colors["tvdmd"], markersize=6)
        dktv_dot, = ax.plot([], [], [], "^", color=colors["dktv"], markersize=7)
        status = ax.text2D(
            0.03,
            0.03,
            "",
            transform=ax.transAxes,
            fontsize=10,
            color="#20242a",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "edgecolor": "#d4dae3",
                "alpha": 0.94,
            },
        )

        ax.set_title(case["title"], fontsize=14, fontweight="semibold")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_box_aspect(
            (
                max(xlim[1] - xlim[0], 1e-9),
                max(ylim[1] - ylim[0], 1e-9),
                max(zlim[1] - zlim[0], 1e-9),
            )
        )
        ax.set_xlabel("$x_1$", labelpad=8)
        ax.set_ylabel("$x_2$", labelpad=8)
        ax.set_zlabel("Iteration step", labelpad=8)
        ax.view_init(elev=12, azim=-112)
        ax.legend(loc="upper left", fontsize=9)
        add_true_dynamics_text(ax, case["gamma"])

        artists.append(
            {
                "data": data,
                "true_line": true_line,
                "tvd_line": tvd_line,
                "dktv_line": dktv_line,
                "tvd_gap": tvd_gap,
                "dktv_gap": dktv_gap,
                "true_dot": true_dot,
                "tvd_dot": tvd_dot,
                "dktv_dot": dktv_dot,
                "status": status,
            }
        )

    def update(frame):
        azim = -112 + 0.24 * frame
        for ax in axes:
            ax.view_init(elev=12, azim=azim)

        for artist in artists:
            data = artist["data"]
            n_frames = data["true"].shape[1]
            idx = min(frame, n_frames - 1)
            end = idx + 1
            start = max(0, end - trajectory_window)
            true = data["true"]
            tvdmd = data["tvdmd"]
            dktv = data["dktv"]
            steps = data["steps"]
            tvdmd_error = data["tvdmd_error"]
            dktv_error = data["dktv_error"]
            z = steps[start:end]
            true_x, true_y, true_z = smooth_trajectory(
                true[0, start:end], true[1, start:end], z
            )
            tvd_x, tvd_y, tvd_z = smooth_trajectory(
                tvdmd[0, start:end], tvdmd[1, start:end], z
            )
            dktv_x, dktv_y, dktv_z = smooth_trajectory(
                dktv[0, start:end], dktv[1, start:end], z
            )
            z_current = [steps[idx]]

            set_line_3d(artist["true_line"], true_x, true_y, true_z)
            set_line_3d(artist["tvd_line"], tvd_x, tvd_y, tvd_z)
            set_line_3d(artist["dktv_line"], dktv_x, dktv_y, dktv_z)
            set_line_3d(artist["true_dot"], true[0, idx:end], true[1, idx:end], z_current)
            set_line_3d(artist["tvd_dot"], tvdmd[0, idx:end], tvdmd[1, idx:end], z_current)
            set_line_3d(artist["dktv_dot"], dktv[0, idx:end], dktv[1, idx:end], z_current)
            set_line_3d(
                artist["tvd_gap"],
                [true[0, idx], tvdmd[0, idx]],
                [true[1, idx], tvdmd[1, idx]],
                [steps[idx], steps[idx]],
            )
            set_line_3d(
                artist["dktv_gap"],
                [true[0, idx], dktv[0, idx]],
                [true[1, idx], dktv[1, idx]],
                [steps[idx], steps[idx]],
            )
            artist["status"].set_text(
                f"step {int(steps[idx])}   DKTV {dktv_error[idx]:.3f}   "
                f"TVDMD {tvdmd_error[idx]:.3f}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = make_writer(output, fps)
    with writer.saving(fig, str(output), dpi=dpi):
        for frame in range(frames):
            update(frame)
            writer.grab_frame()

    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate true/TVDMD/DKTV tracking for saved DKTV results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("SavedResults/Paper_Plots_data"),
        help="Directory containing xdata.pkl, tvdpred.pkl, Astk.pkl, Cstk.pkl, and nnbasis/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("SavedResults/tracking_video.mp4"),
        help="Output video path. Use .mp4 or .gif.",
    )
    parser.add_argument(
        "--basis-class",
        choices=["LNN", "LNNsl"],
        default="LNN",
        help="Lifting network class used for the saved nnbasis checkpoints.",
    )
    parser.add_argument("--prebatch-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=140)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Animate every Nth saved prediction frame.",
    )
    parser.add_argument(
        "--phase-tail",
        type=int,
        default=10,
        help="Number of recent positions shown in the moving phase trajectory window.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Gamma value used in the true dynamics annotation. Inferred from --results-dir when omitted.",
    )
    parser.add_argument(
        "--compare-gammas",
        action="store_true",
        help="Create a 2-column gamma comparison: gamma=0.8 and gamma=6.",
    )
    parser.add_argument(
        "--compare-gammas-3d",
        action="store_true",
        help="Create a 3D gamma comparison using iteration step as the third axis.",
    )
    parser.add_argument(
        "--phase-arc",
        type=float,
        default=1.2,
        help="Target recent true-trajectory arc length for each phase window in --compare-gammas mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")
    if args.phase_tail < 1:
        raise SystemExit("--phase-tail must be >= 1")
    if args.compare_gammas_3d:
        save_gamma_comparison_3d_video(
            args.output,
            args.fps,
            args.dpi,
            args.stride,
            args.phase_tail,
        )
        print(f"Saved 3D gamma comparison video to {args.output}")
        raise SystemExit
    if args.compare_gammas:
        save_gamma_comparison_video(
            args.output,
            args.fps,
            args.dpi,
            args.stride,
            args.phase_tail,
            args.phase_arc,
        )
        print(f"Saved gamma comparison video to {args.output}")
        raise SystemExit
    gamma = infer_gamma(args.results_dir) if args.gamma is None else args.gamma
    tracking_data = build_dktv_predictions(
        args.results_dir,
        args.basis_class,
        args.prebatch_size,
        args.batch_size,
    )
    save_tracking_video(
        tracking_data,
        args.output,
        args.fps,
        args.dpi,
        args.stride,
        args.phase_tail,
        gamma,
    )
    print(f"Saved tracking video to {args.output}")
