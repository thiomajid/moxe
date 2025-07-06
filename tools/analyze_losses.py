#!/usr/bin/env python3
"""
Script to visualize MoxE auxiliary losses from TensorBoard logs.

This script helps analyze the auxiliary loss components during training
to understand how well the MoE routing is working.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_data(log_dir: Path, tags: list[str]):
    """Load data from TensorBoard logs for specified tags."""
    event_acc = EventAccumulator(str(log_dir))
    event_acc.Reload()

    data = {}
    for tag in tags:
        if tag in event_acc.Tags()["scalars"]:
            scalar_events = event_acc.Scalars(tag)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            data[tag] = {"steps": steps, "values": values}
        else:
            print(f"Warning: Tag '{tag}' not found in logs")
            data[tag] = {"steps": [], "values": []}

    return data


def plot_auxiliary_losses(log_dir: Path, output_dir: Path = None):
    """Plot auxiliary losses from training logs."""

    # Tags to monitor
    train_tags = [
        "train/loss",
        "train/ce_loss",
        "train/z_loss",
        "train/load_balancing_loss",
        "train/d_loss",
        "train/group_loss",
    ]

    eval_tags = [
        "eval/loss",
        "eval/ce_loss",
        "eval/z_loss",
        "eval/load_balancing_loss",
        "eval/d_loss",
        "eval/group_loss",
    ]

    coefficient_tags = [
        "coefficients/z_loss_coef",
        "coefficients/load_balancing_loss_coef",
        "coefficients/d_loss_coef",
        "coefficients/group_loss_coef",
    ]

    # Load data
    print("Loading training data...")
    train_data = load_tensorboard_data(log_dir, train_tags)

    print("Loading evaluation data...")
    eval_data = load_tensorboard_data(log_dir, eval_tags)

    print("Loading coefficient data...")
    coeff_data = load_tensorboard_data(log_dir, coefficient_tags)

    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("MoxE Training: Auxiliary Loss Analysis", fontsize=16)

    # Plot 1: Total vs CE Loss (Training)
    ax1 = axes[0, 0]
    if train_data["train/loss"]["steps"]:
        ax1.plot(
            train_data["train/loss"]["steps"],
            train_data["train/loss"]["values"],
            "b-",
            label="Total Loss",
            alpha=0.7,
        )
    if train_data["train/ce_loss"]["steps"]:
        ax1.plot(
            train_data["train/ce_loss"]["steps"],
            train_data["train/ce_loss"]["values"],
            "r-",
            label="CE Loss",
            alpha=0.7,
        )
    ax1.set_title("Training: Total vs Cross-Entropy Loss")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Auxiliary Losses (Training)
    ax2 = axes[0, 1]
    colors = ["green", "orange", "purple", "brown"]
    aux_losses = [
        "train/z_loss",
        "train/load_balancing_loss",
        "train/d_loss",
        "train/group_loss",
    ]
    loss_names = ["Z Loss", "Load Balancing", "Difficulty Loss", "Group Loss"]

    for i, (tag, name, color) in enumerate(zip(aux_losses, loss_names, colors)):
        if train_data[tag]["steps"]:
            ax2.plot(
                train_data[tag]["steps"],
                train_data[tag]["values"],
                color=color,
                label=name,
                alpha=0.7,
            )
    ax2.set_title("Training: Auxiliary Losses")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Loss Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Loss Coefficients
    ax3 = axes[0, 2]
    coeff_names = ["Z Loss Coef", "LB Loss Coef", "D Loss Coef", "Group Loss Coef"]
    coeff_tags_short = [
        "coefficients/z_loss_coef",
        "coefficients/load_balancing_loss_coef",
        "coefficients/d_loss_coef",
        "coefficients/group_loss_coef",
    ]

    for i, (tag, name, color) in enumerate(zip(coeff_tags_short, coeff_names, colors)):
        if coeff_data[tag]["steps"]:
            ax3.plot(
                coeff_data[tag]["steps"],
                coeff_data[tag]["values"],
                color=color,
                label=name,
                alpha=0.7,
                marker="o",
                markersize=2,
            )
    ax3.set_title("Loss Coefficients")
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Coefficient Value")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Evaluation Losses
    ax4 = axes[1, 0]
    if eval_data["eval/loss"]["steps"]:
        ax4.plot(
            eval_data["eval/loss"]["steps"],
            eval_data["eval/loss"]["values"],
            "b-",
            label="Eval Loss",
            alpha=0.7,
            marker="o",
            markersize=3,
        )
    if eval_data["eval/ce_loss"]["steps"]:
        ax4.plot(
            eval_data["eval/ce_loss"]["steps"],
            eval_data["eval/ce_loss"]["values"],
            "r-",
            label="Eval CE Loss",
            alpha=0.7,
            marker="s",
            markersize=3,
        )
    ax4.set_title("Evaluation: Loss Comparison")
    ax4.set_xlabel("Steps")
    ax4.set_ylabel("Loss")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Plot 5: Auxiliary Loss Ratios (relative to CE loss)
    ax5 = axes[1, 1]
    if train_data["train/ce_loss"]["steps"] and all(
        tag in train_data and train_data[tag]["steps"] for tag in aux_losses
    ):
        ce_values = np.array(train_data["train/ce_loss"]["values"])
        steps = train_data["train/ce_loss"]["steps"]

        for i, (tag, name, color) in enumerate(zip(aux_losses, loss_names, colors)):
            if len(train_data[tag]["values"]) == len(ce_values):
                aux_values = np.array(train_data[tag]["values"])
                # Avoid division by zero
                ratio = np.where(ce_values != 0, aux_values / ce_values, 0)
                ax5.plot(
                    steps, ratio, color=color, label=f"{name} / CE Loss", alpha=0.7
                )

        ax5.set_title("Auxiliary Loss Ratios (relative to CE Loss)")
        ax5.set_xlabel("Steps")
        ax5.set_ylabel("Ratio")
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(
            0.5,
            0.5,
            "Insufficient data for ratio plot",
            ha="center",
            va="center",
            transform=ax5.transAxes,
        )
        ax5.set_title("Auxiliary Loss Ratios (insufficient data)")

    # Plot 6: Training vs Evaluation Loss Comparison
    ax6 = axes[1, 2]
    if train_data["train/loss"]["steps"] and eval_data["eval/loss"]["steps"]:
        ax6.plot(
            train_data["train/loss"]["steps"],
            train_data["train/loss"]["values"],
            "b-",
            label="Train Loss",
            alpha=0.7,
        )
        ax6.plot(
            eval_data["eval/loss"]["steps"],
            eval_data["eval/loss"]["values"],
            "r-",
            label="Eval Loss",
            alpha=0.7,
            marker="o",
            markersize=3,
        )
    ax6.set_title("Training vs Evaluation Loss")
    ax6.set_xlabel("Steps")
    ax6.set_ylabel("Loss")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "moxe_auxiliary_losses.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("AUXILIARY LOSS SUMMARY")
    print("=" * 60)

    for tag in train_tags:
        if train_data[tag]["values"]:
            values = train_data[tag]["values"]
            print(
                f"{tag:30} | Final: {values[-1]:.6f} | Avg: {np.mean(values):.6f} | Min: {np.min(values):.6f} | Max: {np.max(values):.6f}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MoxE auxiliary losses from TensorBoard logs"
    )
    parser.add_argument("log_dir", type=str, help="Path to TensorBoard log directory")
    parser.add_argument("--output", "-o", type=str, help="Output directory for plots")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory {log_dir} does not exist")
        return 1

    output_dir = Path(args.output) if args.output else None

    try:
        plot_auxiliary_losses(log_dir, output_dir)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
