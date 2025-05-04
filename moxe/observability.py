import io
import typing as tp

import jax
import numpy as np
import plotly.graph_objects as go
from flax.metrics.tensorboard import SummaryWriter
from PIL import Image

from moxe.ops import compute_entropy
from moxe.output import MoxEForwardPassOutput, MoxELayerOutput


class RouterMetricsWriter:
    def __init__(self, writer: SummaryWriter, num_devices: int):
        self._writer = writer
        self.num_devices = num_devices

    def add_text(self, text: str, tag: str, global_step: int) -> None:
        """Add text to the writer."""
        self._writer.text(text, tag, global_step)

    def add_scalars(self, scalars: dict[str, tp.Any], global_step: int) -> None:
        """Add scalars to the writer."""
        for key, value in scalars.items():
            if isinstance(value, jax.Array):
                value = jax.device_get(value).item()
            self._writer.scalar(key, value, global_step)

    def log_moe_metrics(self, global_step: int, output: MoxEForwardPassOutput):
        for idx, layer_output in enumerate(output.model.layers_outputs):
            self._log_moe_layer_metrics(
                layer_idx=idx, layer_output=layer_output, global_step=global_step
            )

    def __plot_expert_usage(
        self,
        avg_usage: jax.Array,
        expert_usage: jax.Array,
    ) -> tp.Tuple[Image.Image, ...]:
        """Create a bar plot of expert usage and convert it to PIL Image."""

        num_experts = expert_usage.shape[0]
        ideal_usage = avg_usage.sum() / num_experts
        ideal_usage = ideal_usage.mean().item()

        fig = go.Figure()

        # Add expert usage bars
        fig.add_trace(
            go.Bar(
                x=list(range(num_experts)),
                y=jax.device_get(avg_usage),
                name="Average Expert Usage",
            )
        )

        # Add ideal usage line
        fig.add_trace(
            go.Scatter(
                x=[-0.5, num_experts - 0.5],
                y=[ideal_usage, ideal_usage],
                mode="lines",
                name="Ideal Usage",
                line=dict(color="red", dash="dash"),
            )
        )

        fig.update_layout(
            title="Average Expert Usage Distribution",
            xaxis_title="Expert ID",
            yaxis_title="Usage Probability",
            showlegend=True,
            width=800,
            height=400,
        )

        # Convert plot to image
        img_bytes = fig.to_image(format="png", scale=2.0)
        avg_usage_img = Image.open(io.BytesIO(img_bytes))

        # figure for absolute usage
        abs_fig = go.Figure()
        abs_fig.add_trace(
            go.Bar(
                x=list(range(num_experts)),
                y=jax.device_get(expert_usage),
                name="Absolute Expert Usage",
            )
        )

        abs_fig.update_layout(
            title="Absolute Expert Usage Distribution",
            xaxis_title="Expert ID",
            yaxis_title="Usage Count",
            showlegend=True,
            width=800,
            height=400,
        )

        # Convert plot to image
        abs_img_bytes = abs_fig.to_image(format="png", scale=2.0)
        absolute_usage_img = Image.open(io.BytesIO(abs_img_bytes))

        return (
            avg_usage_img,
            absolute_usage_img,
        )

    def _log_moe_layer_metrics(
        self,
        layer_idx: int,
        global_step: int,
        layer_output: MoxELayerOutput,
        expert_usage_counts: jax.Array,
    ) -> None:
        """Log MoE-specific metrics to TensorBoard for a single layer."""

        # Expert usage monitoring

        # (B*S, E)
        routing_weights = jax.nn.softmax(layer_output.router_logits, axis=-1)
        average_expert_usage = routing_weights.mean(axis=0)
        absolute_expert_usage = expert_usage_counts

        individual_expert_activation = {
            f"expert_{idx}": absolute_expert_usage[idx].mean().item()
            for idx in range(absolute_expert_usage.shape[0])
        }

        for expert_key, activation_prob in individual_expert_activation.items():
            self._writer.scalar(
                f"layer_{layer_idx}/dist/absolute_expert_usage/{expert_key}",
                activation_prob,
                global_step,
            )

        self._writer.histogram(
            f"layer_{layer_idx}/dist/absolute_expert_usage",
            jax.device_get(absolute_expert_usage),
            global_step,
        )

        average_individual_expert_activation = {
            f"expert_{idx}": average_expert_usage[idx].mean().item()
            for idx in range(average_expert_usage.shape[0])
        }

        for (
            expert_key,
            activation_prob,
        ) in average_individual_expert_activation.items():
            self._writer.scalar(
                f"layer_{layer_idx}/dist/average_expert_usage/{expert_key}",
                activation_prob,
                global_step,
            )

        self._writer.histogram(
            f"layer_{layer_idx}/dist/average_expert_usage",
            jax.device_get(average_expert_usage),
            global_step,
        )

        # mLSTM to sLSTM group activations probabilities
        num_experts = layer_output.router_logits.shape[-1]
        expert_per_group = num_experts // 2

        # mLSTM/sLSTM group probabilities
        mLSTM_experts_probs = (
            routing_weights[:, :expert_per_group].sum(axis=-1).mean().item()
        )
        sLSTM_experts_probs = (
            routing_weights[:, expert_per_group:].sum(axis=-1).mean().item()
        )

        groups_probs = {
            "mLSTM_experts_probs": mLSTM_experts_probs,
            "sLSTM_experts_probs": sLSTM_experts_probs,
        }

        for group_key, activation_prob in groups_probs.items():
            self._writer.scalar(
                f"layer_{layer_idx}/dist/sLSTM-to-mLSTM group activation/{group_key}",
                activation_prob,
                global_step,
            )

        # Add expert usage plot
        avg_img, abs_img = self.__plot_expert_usage(
            avg_usage=average_expert_usage,
            expert_usage=absolute_expert_usage,
        )

        self._writer.image(
            f"layer_{layer_idx}/average_expert_usage_plot",
            np.array(avg_img).transpose(2, 0, 1),  # Convert to CHW format
            global_step,
        )
        self._writer.image(
            f"layer_{layer_idx}/absolute_expert_usage_plot",
            np.array(abs_img).transpose(2, 0, 1),
            global_step,
        )

        avg_img.close()
        avg_img = None

        abs_img.close()
        abs_img = None

        layer_metrics = {
            "expert_usage/std": absolute_expert_usage.std().mean().item(),
            "expert_usage/max": absolute_expert_usage.max().mean().item(),
            "expert_usage/min": absolute_expert_usage.min().mean().item(),
            "router/z_loss": layer_output.z_loss.item(),
            "router/load_balancing_loss": layer_output.load_balancing_loss.item(),
        }

        if layer_output.conditioned_output is not None:
            unbiased_router_entropy = compute_entropy(
                jax.nn.softmax(
                    layer_output.conditioned_output.unbiased_logits,
                    axis=-1,
                )
            )

            unbiased_router_entropy = unbiased_router_entropy.mean().item()
            biased_router_entropy = compute_entropy(routing_weights).mean().item()
            predicted_entropy = layer_output.conditioned_output.d_t.mean().item()

            entropies = {
                "unbiased_entropy": unbiased_router_entropy,
                "biased_entropy": biased_router_entropy,
                "predicted_entropy": predicted_entropy,
            }

            for entropy, value in entropies.items():
                self._writer.scalar(
                    f"layer_{layer_idx}/dist/entropies/{entropy}",
                    value,
                    global_step,
                )

            layer_metrics["router/d_loss"] = (
                layer_output.conditioned_output.d_loss.item()
            )

            layer_metrics["router/group_loss"] = (
                layer_output.conditioned_output.group_loss.item()
            )

        # Log all metrics under the layer prefix
        for name, value in layer_metrics.items():
            self._writer.scalar(f"layer_{layer_idx}/{name}", value, global_step)
