"""Create paper-style curves and summary tables from saved URVFL results.

The attack result arrays use the row order:
    Emb MSE, Emb cosine distance, Recon MSE, PSNR, SSIM.

Examples:
    python cifar10/plot_results.py --seed 327
    python cifar10/plot_results.py --seed 327 --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METRIC_NAMES = ("Emb MSE", "Emb Cos", "Recon MSE", "PSNR", "SSIM")
METHODS = ("fsha", "urvfl", "sync")

COLORS = {
    "fsha": "#2878B5",
    "urvfl": "#D62728",
    "sync": "#2CA02C",
}

LABELS = {
    "fsha": "FSHA",
    "urvfl": "URVFL",
    "sync": "URVFL_SYNC",
}


def _load_numeric(path: Path) -> np.ndarray:
    values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
    return values


def _load_attack_results(results_dir: Path, seed: int) -> dict[str, np.ndarray]:
    loaded: dict[str, np.ndarray] = {}
    for method in METHODS:
        path = results_dir / f"{method}_{seed}.npy"
        if not path.exists():
            continue
        values = _load_numeric(path)
        if values.ndim != 2 or values.shape[0] != 5:
            print(f"Skip {path.name}: expected shape (5, epochs), got {values.shape}")
            continue
        loaded[method] = values
    return loaded


def generate_attack_outputs(
    results_dir: str | Path,
    seed: int,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> list[Path]:
    """Generate an embedding-distance curve and a final-metric table."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _load_attack_results(results_dir, seed)
    if not data:
        print(f"No attack result files found for seed {seed} in {results_dir}")
        return []

    generated: list[Path] = []

    # Similar to Fig. 7, but the public code stores one point per epoch rather
    # than one point per mini-batch iteration, so the x-axis is named Epoch.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for method in METHODS:
        if method not in data:
            continue
        values = data[method]
        epochs = np.arange(1, values.shape[1] + 1)
        label = LABELS[method]
        axes[0].plot(epochs, values[0], label=label, color=COLORS[method], linewidth=1.8)
        axes[1].plot(epochs, values[1], label=label, color=COLORS[method], linewidth=1.8)

    axes[0].set_title("Embedding MSE distance across epochs")
    axes[1].set_title("Embedding cosine distance across epochs")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Distance")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=True)
    fig.tight_layout()
    curve_path = output_dir / f"attack_curves_seed{seed}.png"
    fig.savefig(curve_path, dpi=300, bbox_inches="tight")
    generated.append(curve_path)
    if show:
        plt.show()
    plt.close(fig)

    rows: list[list[str]] = []
    for method in METHODS:
        if method not in data:
            continue
        final = data[method][:, -1]
        # Reorder the five stored rows to match the requested paper-style table.
        rows.append(
            [
                LABELS[method],
                f"{final[2]:.6f}",
                f"{final[0]:.6f}",
                f"{final[1]:.6f}",
                f"{final[3]:.4f}",
                f"{final[4]:.6f}",
            ]
        )

    headers = ["Method", "Recon MSE (down)", "Emb MSE (down)", "Emb Cos (down)", "PSNR (up)", "SSIM (up)"]
    csv_path = output_dir / f"attack_table_seed{seed}.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)
    generated.append(csv_path)

    fig_height = 1.35 + 0.42 * len(rows)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.45)
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("#C9CED6")
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#E9EEF5")
            cell.set_text_props(weight="bold")
    ax.set_title(f"Final-epoch attack metrics (seed={seed})", fontsize=12, pad=10)
    fig.tight_layout()
    table_path = output_dir / f"attack_table_seed{seed}.png"
    fig.savefig(table_path, dpi=300, bbox_inches="tight")
    generated.append(table_path)
    if show:
        plt.show()
    plt.close(fig)

    for path in generated:
        print(f"Generated: {path}")
    return generated


def generate_detection_outputs(
    results_dir: str | Path,
    seed: int,
    output_dir: str | Path | None = None,
    show: bool = False,
) -> list[Path]:
    """Plot SG/GS score files if the detection run saved them."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir.parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    defenses: list[tuple[str, float, dict[str, np.ndarray]]] = []
    for defense, threshold in (("sg", 0.9), ("gs", 0.7)):
        curves: dict[str, np.ndarray] = {}
        for method in METHODS:
            path = results_dir / f"{defense}_score_{method}_{seed}.npy"
            if path.exists():
                values = _load_numeric(path).reshape(-1)
                if values.size:
                    curves[method] = values
        if curves:
            defenses.append((defense, threshold, curves))

    if not defenses:
        print("No SG/GS score files found. Run detection_cifar.py after adding score saving.")
        return []

    fig, axes = plt.subplots(1, len(defenses), figsize=(5.2 * len(defenses), 4.0), squeeze=False)
    for ax, (defense, threshold, curves) in zip(axes[0], defenses):
        for method in METHODS:
            if method not in curves:
                continue
            values = curves[method]
            if defense == "gs":
                # defense.py starts producing GS scores at batch 100 and then
                # records one score every 50 training batches.
                steps = 100 + np.arange(values.size) * 50
            else:
                steps = np.arange(1, values.size + 1)
            ax.plot(steps, values, label=LABELS[method], color=COLORS[method], linewidth=1.6)
        ax.axhline(threshold, color="#FF8C00", linestyle="--", linewidth=1.5, label="Detection threshold")
        ax.set_title(f"{defense.upper()} detection scores")
        ax.set_xlabel("Training step (batch)" if defense == "gs" else "Detection record")
        ax.set_ylabel("Score")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(frameon=True)
    fig.tight_layout()
    path = output_dir / f"detection_scores_seed{seed}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Generated: {path}")
    return [path]


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=327)
    parser.add_argument("--results-dir", type=Path, default=script_dir / "attack_result")
    parser.add_argument("--output-dir", type=Path, default=script_dir / "figures")
    parser.add_argument("--show", action="store_true", help="also open interactive plot windows")
    args = parser.parse_args()

    generate_attack_outputs(args.results_dir, args.seed, args.output_dir, args.show)
    generate_detection_outputs(args.results_dir, args.seed, args.output_dir, args.show)


if __name__ == "__main__":
    main()
