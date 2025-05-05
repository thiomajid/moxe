import io
import typing as tp

import jax
import numpy as np
import plotly.graph_objects as go
import tensorflow as tf
from flax.metrics.tensorboard import SummaryWriter
from PIL import Image

from moxe.ops import compute_entropy
from moxe.output import MoxEForwardPassOutput, MoxELayerOutput


class RouterMetricsWriter:
    def __init__(self, log_dir: str):
        self._writer = SummaryWriter(log_dir)
        self.log_dir = log_dir

    def log_grouped_scalars(self, tag: str, scalar_dict: dict[str, tp.Any], step: int):
        writer = tf.summary.create_file_writer(
            self.log_dir, filename_suffix=tag.replace("/", "_")
        )
        with writer.as_default():
            for key, value in scalar_dict.items():
                # Create hierarchical tags to group metrics
                full_tag = f"{tag}/{key}"
                tf.summary.scalar(full_tag, value, step=step)
            writer.flush()

    def add_text(self, text: str, tag: str, global_step: int) -> None:
        """Add text to the writer."""
        self._writer.text(text, tag, global_step)

    def log_individual_scalars(
        self,
        scalars: dict[str, tp.Any],
        global_step: int,
    ) -> None:
        """Add scalars to the writer."""
        for key, value in scalars.items():
            if isinstance(value, jax.Array):
                value = jax.device_get(value).item()
            self._writer.scalar(key, value, global_step)

    def log_moe_metrics(self, global_step: int, output: MoxEForwardPassOutput):
        self.log_individual_scalars(
            global_step=global_step,
            scalars={
                "train/ce_loss": output.ce_loss.item(),
                "train/z_loss": output.z_loss.item(),
                "train/load_balance_loss": output.load_balance_loss.item(),
                "train/d_loss": output.d_loss.item(),
                "train/group_loss": output.group_loss.item(),
            },
        )

        for idx, layer_output in enumerate(output.model.layers_outputs):
            self._log_moe_layer_metrics(
                layer_idx=idx, layer_output=layer_output, global_step=global_step
            )

    def __plot_expert_usage(
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

    def _log_moe_layer_metrics(
        self,
        layer_idx: int,
        global_step: int,
        layer_output: MoxELayerOutput,
    ) -> None:
        """Log MoE-specific metrics to TensorBoard for a single layer."""

        # Expert usage monitoring

        # (B*S, E)
        routing_weights = layer_output.router_probs
        # (E,) - Average probability assigned to each expert across all tokens
        avg_expert_weighting = layer_output.router_probs.mean(axis=0)
        # (E,) - Fraction of tokens processed by each expert relative to total tokens
        expert_load = layer_output.expert_load
        # (E,) - Absolute count of tokens assigned to each expert
        token_distribution = layer_output.expert_token_counts

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
            "mLSTM": mLSTM_experts_probs,
            "sLSTM": sLSTM_experts_probs,
        }

        self.log_individual_scalars(
            groups_probs,
            global_step,
        )

        # Add expert usage plots
        avg_weighting_img, expert_load_img, token_dist_img = self.__plot_expert_usage(
            avg_weighting=avg_expert_weighting,
            expert_load=expert_load,
            token_distribution=token_distribution,
        )

        self._writer.image(
            f"layer_{layer_idx}/average_expert_weighting_plot",
            np.array(avg_weighting_img).transpose(2, 0, 1),  # Convert to CHW format
            global_step,
        )
        self._writer.image(
            f"layer_{layer_idx}/expert_load_plot",
            np.array(expert_load_img).transpose(2, 0, 1),
            global_step,
        )
        self._writer.image(
            f"layer_{layer_idx}/token_distribution_plot",
            np.array(token_dist_img).transpose(2, 0, 1),
            global_step,
        )

        avg_weighting_img.close()
        avg_weighting_img = None

        expert_load_img.close()
        expert_load_img = None

        token_dist_img.close()
        token_dist_img = None

        # Calculate metrics based on expert load (fraction of tokens)
        expert_load_metrics = {
            f"layer_{layer_idx}/expert_load/std": expert_load.std().item(),
            f"layer_{layer_idx}/expert_load/max": expert_load.max().item(),
            f"layer_{layer_idx}/expert_load/min": expert_load.min().item(),
        }
        self.log_individual_scalars(
            scalars=expert_load_metrics, global_step=global_step
        )

        # Calculate metrics based on token distribution (absolute counts)
        token_dist_metrics = {
            f"layer_{layer_idx}/token_distribution/std": token_distribution.std().item(),
            f"layer_{layer_idx}/token_distribution/max": token_distribution.max().item(),
            f"layer_{layer_idx}/token_distribution/min": token_distribution.min().item(),
        }
        self.log_individual_scalars(scalars=token_dist_metrics, global_step=global_step)

        layer_metrics = {
            f"layer_{layer_idx}/router/z_loss": layer_output.z_loss.item(),
            f"layer_{layer_idx}/router/load_balancing_loss": layer_output.load_balancing_loss.item(),
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
                "unbiased": unbiased_router_entropy,
                "biased": biased_router_entropy,
                "predicted": predicted_entropy,
            }

            self.log_grouped_scalars(
                tag=f"layer_{layer_idx}/entropies",
                scalar_dict=entropies,
                step=global_step,
            )

            layer_metrics.update(
                {
                    f"layer_{layer_idx}/router/d_loss": layer_output.conditioned_output.group_loss.item(),
                    f"layer_{layer_idx}/router/group_loss": layer_output.conditioned_output.d_loss.item(),
                }
            )

        # Log remaining layer-specific metrics
        self.log_individual_scalars(scalars=layer_metrics, global_step=global_step)
