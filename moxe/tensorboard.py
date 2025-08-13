"""TensorBoard logging utility for OrthoConv training."""

import io
import logging
import typing as tp
from pathlib import Path

import jax
import jax.tree_util as jtu
import numpy as np
import optax
import plotly.graph_objects as go
from flax.metrics.tensorboard import SummaryWriter
from PIL import Image

from moxe.output import BaseMoELayerOutput, MoxELayerOutput


class TensorBoardLogger:
    """Utility class for logging metrics and images to TensorBoard."""

    def __init__(self, log_dir: tp.Union[str, Path], name: str = "train"):
        """Initialize TensorBoard logger.

        Args:
            log_dir: Directory to save TensorBoard logs
            name: Name for this logger instance
        """
        self.log_dir = Path(log_dir)

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        self.name = name

        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / name))

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"TensorBoard logging initialized at {self.log_dir / name}")

    def log_scalar(
        self,
        tag: str,
        value: tp.Union[float, jax.Array, np.ndarray],
        step: int,
    ):
        """Log a scalar value.

        Args:
            tag: Name of the scalar
            value: Scalar value to log
            step: Global step
        """
        if isinstance(value, (jax.Array, np.ndarray)):
            value = float(value.item())
        elif not isinstance(value, (int, float)):
            value = float(value)

        self.writer.scalar(tag, value, step)

    def log_scalars(
        self,
        tag_scalar_dict: tp.Dict[str, tp.Union[float, jax.Array, np.ndarray]],
        step: int,
    ):
        """Log multiple scalars at once.

        Args:
            tag_scalar_dict: Dictionary of tag -> scalar value
            step: Global step
        """
        for tag, value in tag_scalar_dict.items():
            self.log_scalar(tag, value, step)

    def log_histogram(
        self,
        tag: str,
        values: tp.Union[jax.Array, np.ndarray],
        step: int,
    ):
        """Log a histogram of values.

        Args:
            tag: Name for the histogram
            values: Values to create histogram from
            step: Global step
        """
        if isinstance(values, jax.Array):
            values = np.array(values)

        self.writer.histogram(tag, values, step)

    def log_learning_rate(self, lr: float, step: int):
        """Log learning rate.

        Args:
            lr: Learning rate value
            step: Global step
        """
        self.log_scalar("learning_rate", lr, step)

    def log_gradients(self, grads: tp.Dict[str, tp.Any], step: int):
        """Log gradient statistics.

        Args:
            grads: Dictionary of gradients
            step: Global step
        """

        # Compute gradient norm
        grad_norm = optax.global_norm(grads)
        self.log_scalar("gradient_norm", grad_norm, step)

        # Log histogram of gradient values for key layers
        def log_grad_hist(path, grad):
            if isinstance(grad, jax.Array) and grad.size > 0:
                tag = f"gradients/{'/'.join(map(str, path))}"
                self.log_histogram(tag, grad, step)

        # Traverse gradient tree and log histograms for some key components
        jtu.tree_map_with_path(log_grad_hist, grads)

    def _write_layer_scalar_metrics(
        self,
        index: int,
        output: MoxELayerOutput | BaseMoELayerOutput,
        step: int,
    ):
        metrics = {
            "z_loss": output.z_loss[index].item(),
            "load_balancing_loss": output.load_balancing_loss[index].item(),
        }

        if output.conditioned_output is not None:
            metrics.update(
                d_loss=output.conditioned_output.d_loss[index].item(),
                group_loss=output.conditioned_output.group_loss[index].item(),
                router_entropy=output.conditioned_output.router_entropy[index].item(),
                predicted_entropy=output.conditioned_output.predicted_entropy[
                    index
                ].item(),
            )

        data = {f"layer_{index}/{metric}": value for metric, value in metrics.items()}

        self.log_scalars(data, step)

    def _plot_expert_usage(
        self,
        avg_weighting: jax.Array,
        expert_load: jax.Array,
        token_distribution: jax.Array,
    ) -> tp.Tuple[Image.Image, Image.Image, Image.Image]:
        """Create bar plots of expert weighting, load, and token distribution."""

        num_experts = avg_weighting.shape[0]
        ideal_weighting = avg_weighting.sum() / num_experts
        ideal_weighting = ideal_weighting.mean().item()

        # --- Average Expert Weighting Plot ---
        fig_avg_weighting = go.Figure()
        fig_avg_weighting.add_trace(
            go.Bar(
                x=list(range(num_experts)),
                y=jax.device_get(avg_weighting),
                name="Average Expert Weighting",
            )
        )
        fig_avg_weighting.add_trace(
            go.Scatter(
                x=[-0.5, num_experts - 0.5],
                y=[ideal_weighting, ideal_weighting],
                mode="lines",
                name="Ideal Weighting",
                line=dict(color="red", dash="dash"),
            )
        )
        fig_avg_weighting.update_layout(
            title="Average Expert Weighting Distribution",
            xaxis_title="Expert ID",
            yaxis_title="Average Weighting Probability",
            showlegend=True,
            width=800,
            height=400,
        )
        img_bytes_avg_weighting = fig_avg_weighting.to_image(format="png", scale=2.0)
        avg_weighting_img = Image.open(io.BytesIO(img_bytes_avg_weighting))

        # --- Expert Load Plot ---
        fig_load = go.Figure()
        fig_load.add_trace(
            go.Bar(
                x=list(range(num_experts)),
                y=jax.device_get(expert_load),
                name="Expert Load",
            )
        )
        fig_load.update_layout(
            title="Expert Load Distribution",
            xaxis_title="Expert ID",
            yaxis_title="Load (Fraction of Tokens)",
            showlegend=True,
            width=800,
            height=400,
        )
        img_bytes_load = fig_load.to_image(format="png", scale=2.0)
        expert_load_img = Image.open(io.BytesIO(img_bytes_load))

        # --- Token Distribution Plot ---
        fig_token_dist = go.Figure()
        fig_token_dist.add_trace(
            go.Bar(
                x=list(range(num_experts)),
                y=jax.device_get(token_distribution),
                name="Token Distribution",
            )
        )
        fig_token_dist.update_layout(
            title="Token Distribution per Expert",
            xaxis_title="Expert ID",
            yaxis_title="Token Count",
            showlegend=True,
            width=800,
            height=400,
        )
        img_bytes_token_dist = fig_token_dist.to_image(format="png", scale=2.0)
        token_dist_img = Image.open(io.BytesIO(img_bytes_token_dist))

        return (
            avg_weighting_img,
            expert_load_img,
            token_dist_img,
        )

    def write_moe_layers_metrics(
        self,
        layers_output: MoxELayerOutput | BaseMoELayerOutput,
        step: int,
    ):
        num_layers = layers_output.z_loss.shape[0]
        for layer_idx in range(num_layers):
            self._write_layer_scalar_metrics(layer_idx, layers_output, step)

            # (B*S, E) -> (E,)
            avg_expert_weighting = layers_output.router_probs[layer_idx].mean(axis=0)

            # (E,) - Fraction of tokens processed by each expert relative to total tokens
            expert_load = layers_output.expert_load[layer_idx]

            # (E,) - Absolute count of tokens assigned to each expert
            token_distribution = layers_output.expert_token_counts[layer_idx]

            avg_weighting_img, expert_load_img, token_dist_img = (
                self._plot_expert_usage(
                    avg_weighting=avg_expert_weighting,
                    expert_load=expert_load,
                    token_distribution=token_distribution,
                )
            )

            self.writer.image(
                f"layer_{layer_idx}/average_expert_weighting",
                np.array(avg_weighting_img).transpose(2, 0, 1),  # Convert to CHW format
                step,
            )

            self.writer.image(
                f"layer_{layer_idx}/expert_load",
                np.array(expert_load_img).transpose(2, 0, 1),
                step,
            )

            self.writer.image(
                f"layer_{layer_idx}/token_distribution",
                np.array(token_dist_img).transpose(2, 0, 1),
                step,
            )

            avg_weighting_img.close()
            avg_weighting_img = None

            expert_load_img.close()
            expert_load_img = None

            token_dist_img.close()
            token_dist_img = None

    def close(self):
        """Close the TensorBoard writer."""
        if hasattr(self, "writer"):
            self.writer.close()
            self.logger.info(f"TensorBoard logger closed for {self.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
