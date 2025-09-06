import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from huggingface_hub import create_repo, repo_exists, upload_folder
from transformers import AutoTokenizer

from training.arguments import TrainingArguments
from training.state import TrainerState
from training.tensorboard import TensorBoardLogger
from training.utils.module import checkpoint_post_eval
from xlstm_jax.inference import GenerationCarry, GenerationMixin


class Callback:
    def __init__(self):
        pass

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, state: TrainerState, metrics: dict):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass


class CheckpointCallback(Callback):
    def __init__(
        self,
        model: nnx.Module,
        options: ocp.CheckpointManagerOptions,
        checkpoint_dir: Path,
        logger: logging.Logger,
    ):
        super().__init__()

        self.model = model
        self.logger = logger
        self.options = options
        self.checkpoint_dir = checkpoint_dir

    def on_epoch_end(self, state, metrics):
        checkpoint_post_eval(
            logger=self.logger,
            model=self.model,
            checkpoint_dir=self.checkpoint_dir,
            options=self.options,
            metrics=metrics,
            global_step=state.current_step,
            epoch=state.epoch,
        )


class PushToHubCallback(Callback):
    def __init__(self, logger: logging.Logger, args: TrainingArguments):
        super().__init__()

        self.logger = logger
        self.args = args

    def on_train_end(self):
        folder_path = Path(self.args.output_dir)
        self.logger.info(
            f"Pushing artifacts from {str(folder_path)} to Hugging Face Hub repository: {self.args.hub_model_id}..."
        )
        if not repo_exists(self.args.hub_model_id, token=self.args.hub_token):
            self.logger.info(f"Creating repository {self.args.hub_model_id}...")
            create_repo(
                repo_id=self.args.hub_model_id,
                token=self.args.hub_token,
                private=self.args.hub_private_repo,
                exist_ok=True,
            )

        upload_folder(
            repo_id=self.args.hub_model_id,
            folder_path=folder_path,
            token=self.args.hub_token,
            commit_message=self.args.upload_message,
        )
        self.logger.info(
            f"Push to Hub completed. Results can be viewed at https://huggingface.co/{self.args.hub_model_id}"
        )


class GenerateTextCallback(Callback):
    def __init__(
        self,
        model: GenerationMixin,
        tokenizer: AutoTokenizer,
        sample_tokens: jax.Array,
        max_new_tokens: int,
        temperature: float,
        reporter: TensorBoardLogger,
        logger: logging.Logger,
        greedy: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sample_tokens = sample_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.greedy = greedy

        self.reporter = reporter
        self.logger = logger
        self.dtype = sample_tokens.dtype

        self.batch_size = sample_tokens.shape[0]
        self.initial_sequence_length = sample_tokens.shape[1]
        self.total_sequence_length = self.initial_sequence_length + max_new_tokens

        full_x_init = jnp.zeros(
            shape=(self.batch_size, self.total_sequence_length),
            dtype=jnp.int32,
        )

        self.full_x_init = full_x_init.at[:, : self.initial_sequence_length].set(
            self.sample_tokens
        )

    def on_epoch_end(self, state, metrics):
        try:
            key = jax.random.key(state.current_step + 1256)
            carry: GenerationCarry = (
                self.full_x_init,
                self.initial_sequence_length,
                key,
            )

            sequences = self.model.generate(
                carry,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                greedy=self.greedy,
            )

            sequences = np.array(jax.device_get(sequences))
            decoded: list[str] = []

            for i in range(self.batch_size):
                generated_text: str = self.tokenizer.decode(
                    sequences[i],
                    remove_special_tokens=True,
                )

                # add markdown bold delimiters to distinguish the original text
                # from the generated one
                initial_text = generated_text[: self.initial_sequence_length]
                generated_text = (
                    "**"
                    + initial_text
                    + "**"
                    + generated_text[self.initial_sequence_length + 1 :]
                )

                generated_text = generated_text.replace(
                    str(self.tokenizer.pad_token), " "
                )

                generated_text = generated_text.replace(
                    str(self.tokenizer.eos_token), ""
                ).strip()

                decoded.append(generated_text)

            self.reporter.log_text(decoded, state.current_step)

            self.logger.info(
                f"Generated and logged {self.batch_size} text samples for epoch {state.epoch + 1}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to generate new text: {e}")
