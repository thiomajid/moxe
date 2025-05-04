# Copyright (c) NXAI GmbH and its affiliates 2023
# Korbinian Poeppel
# Converted to JAX/Flax by Abdoul Majid O. Thiombiano
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import nnx

from .src.vanilla import (
    slstm_forward,
    slstm_forward_step,
    slstm_pointwise_function_registry,
)

LOGGER = logging.getLogger(__name__)

DTYPE_DICT = {
    "bfloat16": jnp.bfloat16,
    "float16": jnp.float16,
    "float32": jnp.float32,
}
DTYPES = Literal["bfloat16", "float16", "float32"]


@dataclass
class sLSTMCellConfig:
    hidden_size: int = -1
    num_heads: int = 4  # must divide the hidden size
    num_states: int = 4  # sLSTM has 4, standard LSTM has 2
    backend: Literal["vanilla"] = (
        "vanilla"  # Only support vanilla backend in JAX version
    )
    function: str = "slstm"
    bias_init: Literal["powerlaw_blockdependent", "small_init", "standard"] = (
        "powerlaw_blockdependent"
    )
    recurrent_weight_init: Literal["zeros", "standard"] = "zeros"

    _block_idx: int = 0  # index in block sequence, needed for forget gate init
    _num_blocks: int = 1  # total blocks in model, needed for forget gate init

    num_gates: int = 4
    gradient_recurrent_clipval: Optional[float] = None
    forward_clipval: Optional[float] = None

    batch_size: int = 8
    input_shape: Literal["BSGNH", "SBGNH"] = "BSGNH"
    internal_input_shape: Literal["SBNGH", "SBGNH", "SBNHG"] = "SBNGH"
    output_shape: Literal["BNSH", "SBH", "BSH", "SBNH"] = "BNSH"

    constants: Dict[str, Any] = field(default_factory=dict)
    dtype: DTYPES = "bfloat16"
    dtype_b: Optional[DTYPES] = "float32"  # biases
    dtype_r: Optional[DTYPES] = None  # recurrent matrix
    dtype_w: Optional[DTYPES] = None  # inputs / w matrix
    dtype_g: Optional[DTYPES] = None  # gates
    dtype_s: Optional[DTYPES] = None  # states
    dtype_a: Optional[DTYPES] = None  # internal accumulation

    initial_val: Union[float, Sequence[float]] = 0.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_heads

    @property
    def input_dim(self):
        return 4 * self.hidden_size

    @property
    def jax_dtype(self) -> jnp.dtype:
        return DTYPE_DICT[self.dtype]

    @property
    def jax_dtype_b(self) -> jnp.dtype:
        return DTYPE_DICT[self.dtype_b]

    @property
    def jax_dtype_r(self) -> jnp.dtype:
        return DTYPE_DICT[self.dtype_r]

    @property
    def jax_dtype_w(self) -> jnp.dtype:
        return DTYPE_DICT[self.dtype_w]

    @property
    def jax_dtype_s(self) -> jnp.dtype:
        return DTYPE_DICT[self.dtype_s]

    def __post_init__(self):
        if self.num_heads <= 0:
            self.num_heads = 1

        if self.dtype_b is None:
            self.dtype_b = self.dtype
        if self.dtype_a is None:
            self.dtype_a = self.dtype_b
        if self.dtype_r is None:
            self.dtype_r = self.dtype
        if self.dtype_w is None:
            self.dtype_w = self.dtype
        if self.dtype_s is None:
            self.dtype_s = self.dtype_w
        if self.dtype_g is None:
            self.dtype_g = self.dtype_r

        # In JAX implementation we only support slstm function
        self.num_states = 4  # sLSTM has 4 states


class sLSTMCellBase(nnx.Module):
    """Base class for sLSTM cell implementation in JAX/Flax."""

    config_class = sLSTMCellConfig

    def __init__(
        self,
        config: sLSTMCellConfig,
        *,
        rngs: nnx.Rngs,
        dtype=jnp.float32,
    ):
        self.config = config
        self.dtype = dtype

        LOGGER.debug("Init module")

        head_dim = self.config.hidden_size // self.config.num_heads

        # Initialize recurrent kernel
        self._recurrent_kernel_ = nnx.Param(
            jnp.zeros(
                (self.config.num_heads, head_dim, self.config.num_gates, head_dim),
                dtype=DTYPE_DICT[self.config.dtype_r],
            ),
            init_fn=self._initialize_recurrent_kernel,
        )

        # Initialize bias
        self._bias_ = nnx.Param(
            jnp.zeros(
                (self.config.num_heads, self.config.num_gates, head_dim),
                dtype=DTYPE_DICT[self.config.dtype_b],
            ),
            init_fn=self._initialize_bias,
        )

        # Check configuration validity
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError(
                f"Hidden Size {self.config.hidden_size} must be divisible by head num {self.config.num_heads}"
            )

    def _initialize_recurrent_kernel(self, key, shape):
        result = jnp.zeros(shape)
        head_dim = self.config.head_dim

        if self.config.recurrent_weight_init == "zeros":
            # Keep zeros
            pass
        elif self.config.recurrent_weight_init == "standard":
            subkey = jax.random.split(
                key, self.config.num_heads * self.config.num_gates
            )
            scale = 1.0 / jnp.sqrt(self.config.hidden_size)

            for h in range(self.config.num_heads):
                for i in range(self.config.num_gates):
                    idx = h * self.config.num_gates + i
                    result = result.at[h, :, i, :].set(
                        jax.random.uniform(
                            subkey[idx],
                            (head_dim, head_dim),
                            minval=-scale,
                            maxval=scale,
                        )
                    )

        return result

    def _initialize_bias(self, key, shape):
        result = jnp.zeros(shape)
        head_dim = self.config.head_dim

        for h in range(self.config.num_heads):
            for i, gate in enumerate(["i", "f", "z", "o"]):
                if self.config.bias_init == "powerlaw_blockdependent" and gate == "f":
                    # Special initialization for forget gates
                    ratio_0_to_1 = (
                        self.config._block_idx / (self.config._num_blocks - 1)
                        if self.config._num_blocks > 1
                        else 0.0
                    )

                    positions = jnp.arange(head_dim) / (head_dim - 1)
                    power = 0.3 + 1.3 * ratio_0_to_1
                    init_values = -(-5.0 + 12.0 * positions**power)
                    result = result.at[h, i, :].set(init_values)

                elif self.config.bias_init == "small_init" and gate == "f":
                    # Linear spacing for forget gate bias
                    init_values = jnp.linspace(3.0, 6.0, head_dim)
                    result = result.at[h, i, :].set(init_values)

                elif self.config.bias_init == "standard":
                    # Standard uniform initialization
                    scale = 1 / jnp.sqrt(self.config.hidden_size)
                    subkey = jax.random.fold_in(key, h * self.config.num_gates + i)
                    values = jax.random.uniform(
                        subkey, (head_dim,), minval=-scale, maxval=scale
                    )
                    result = result.at[h, i, :].set(values)

        return result

    @property
    def head_dim(self):
        return self.config.hidden_size // self.config.num_heads

    def _permute_input(self, x: jnp.ndarray) -> jnp.ndarray:
        """Transform input to the internal format expected by slstm implementation."""
        if self.config.input_shape == "SBGNH":
            y = x.reshape(
                x.shape[0], x.shape[1], self.config.num_gates, self.config.num_heads, -1
            )
        elif self.config.input_shape == "BSGNH":
            # if the input is in the batch-first format transform it to be in sequence-first format
            y = jnp.transpose(
                x.reshape(
                    x.shape[0],
                    x.shape[1],
                    self.config.num_gates,
                    self.config.num_heads,
                    -1,
                ),
                (1, 0, 2, 3, 4),
            )
        else:
            raise ValueError("Bad input_shape value")

        if self.config.internal_input_shape == "SBGNH":
            return y.reshape(y.shape[0], y.shape[1], -1)
        elif self.config.internal_input_shape == "SBNGH":
            return jnp.transpose(y, (0, 1, 3, 2, 4)).reshape(y.shape[0], y.shape[1], -1)
        elif self.config.internal_input_shape == "SBNHG":
            return jnp.transpose(y, (0, 1, 3, 4, 2)).reshape(y.shape[0], y.shape[1], -1)
        else:
            raise ValueError("Bad internal_input_shape value")

    def _permute_output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Transform output from internal format to the desired output format."""
        if self.config.output_shape == "SBH":
            return x
        elif self.config.output_shape == "BSH":
            return jnp.transpose(x, (1, 0, 2))
        elif self.config.output_shape == "BNSH":
            reshaped = x.reshape(
                x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim
            )
            return jnp.transpose(reshaped, (1, 2, 0, 3))
        elif self.config.output_shape == "SBNH":
            return x.reshape(
                x.shape[0], x.shape[1], self.config.num_heads, self.config.head_dim
            )
        else:
            raise ValueError(f"Unsupported output_shape: {self.config.output_shape}")

    def _check_input(self, input: jnp.ndarray) -> None:
        expected_size = self.config.hidden_size * self.config.num_gates
        chex.assert_equal(input.shape[-1], expected_size)
        # if input.shape[-1] != expected_size:
        #     raise ValueError(
        #         f"Input size mismatch: Expected input size {expected_size}, but got {input.shape[-1]}."
        #     )

    def _zero_state(self, input: jnp.ndarray) -> jnp.ndarray:
        """Returns a zeros state matching dtype and batch size of `input`."""
        batch_dim = input.shape[1]
        return jnp.zeros(
            (self.config.num_states, batch_dim, self.config.hidden_size),
            dtype=input.dtype,
        )

    def _get_state(
        self,
        input: jnp.ndarray,
        state: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        if state is None:
            state = self._zero_state(input)
        else:
            # Check state shape
            expected_shape = (
                self.config.num_states,
                input.shape[1],
                self.config.hidden_size,
            )
            chex.assert_shape(inputs=[state], expected_shapes=[expected_shape])
            # if state.shape != expected_shape:
            #     raise ValueError(
            #         f"Expected state shape {expected_shape}, got {state.shape}"
            #     )
        return state

    def _get_final_state(self, all_states: jnp.ndarray) -> jnp.ndarray:
        """
        All states has the structure
        [STATES, SEQUENCE, BATCH, HIDDEN]
        """
        return all_states[:, -1]

    def __call__(
        self,
        input: jnp.ndarray,
        state: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Process a sequence through the sLSTM cell."""
        self._check_input(input)
        input = self._permute_input(input)
        states = self._get_state(input, state)
        all_states = self._impl(input, states)
        state = self._get_final_state(all_states)
        output = self._permute_output(all_states[0][1:])

        return output, state

    def reset_parameters(self, rngs: nnx.Rngs):
        """Resets this layer's parameters to their initial values."""
        # Initialize recurrent kernel parameters
        key = rngs.params()
        head_dim = self.config.head_dim

        # Initialize recurrent kernel weights
        if self.config.recurrent_weight_init == "zeros":
            # Zero initialization for recurrent weights
            self._recurrent_kernel_ = self._recurrent_kernel_.at[:].set(
                jnp.zeros_like(self._recurrent_kernel_)
            )
        elif self.config.recurrent_weight_init == "standard":
            # Standard uniform initialization with scaling
            scale = 1.0 / jnp.sqrt(self.config.hidden_size)
            for h in range(self.config.num_heads):
                for i in range(self.config.num_gates):
                    key = jax.random.fold_in(key, h * self.config.num_gates + i)
                    values = jax.random.uniform(
                        key, shape=(head_dim, head_dim), minval=-scale, maxval=scale
                    )
                    self._recurrent_kernel_ = self._recurrent_kernel_.at[
                        h, :, i, :
                    ].set(values)

        # Initialize bias parameters
        for h in range(self.config.num_heads):
            for i, gate in enumerate(["i", "f", "z", "o"]):
                if self.config.bias_init == "powerlaw_blockdependent":
                    if gate == "f":
                        # Special initialization for forget gates
                        ratio_0_to_1 = (
                            self.config._block_idx / (self.config._num_blocks - 1)
                            if self.config._num_blocks > 1
                            else 0.0
                        )

                        positions = jnp.arange(head_dim) / (head_dim - 1)
                        power = 0.3 + 1.3 * ratio_0_to_1
                        init_values = -(-5.0 + 12.0 * positions**power)
                        self._bias_ = self._bias_.at[h, i, :].set(init_values)
                    else:
                        # Zero initialization for other gates
                        self._bias_ = self._bias_.at[h, i, :].set(jnp.zeros(head_dim))
                elif self.config.bias_init == "small_init":
                    if gate == "f":
                        # Linear spacing for forget gate bias
                        init_values = jnp.linspace(3.0, 6.0, head_dim)
                        self._bias_ = self._bias_.at[h, i, :].set(init_values)
                    else:
                        # Zero initialization for other gates
                        self._bias_ = self._bias_.at[h, i, :].set(jnp.zeros(head_dim))
                elif self.config.bias_init == "zeros":
                    # Zero initialization for all gates
                    self._bias_ = self._bias_.at[h, i, :].set(jnp.zeros(head_dim))
                elif self.config.bias_init == "standard":
                    # Standard uniform initialization with scaling
                    scale = 1.0 / jnp.sqrt(self.config.hidden_size)
                    key = jax.random.fold_in(
                        key,
                        h * self.config.num_gates
                        + i
                        + self.config.num_heads * self.config.num_gates,
                    )
                    values = jax.random.uniform(
                        key, shape=(head_dim,), minval=-scale, maxval=scale
                    )
                    self._bias_ = self._bias_.at[h, i, :].set(values)


class sLSTMCell_vanilla(sLSTMCellBase):
    """Vanilla implementation of sLSTM cell using JAX/Flax."""

    config_class = sLSTMCellConfig

    def __init__(self, config: sLSTMCellConfig, rngs: nnx.Rngs, dtype=jnp.float32):
        super().__init__(config, rngs=rngs, dtype=dtype)

        # Load pointwise function
        self.pointwise = slstm_pointwise_function_registry[self.config.function]
        self.config.internal_input_shape = "SBGNH"

    def _recurrent_kernel_ext2int(self, recurrent_kernel_ext: jnp.ndarray):
        """Convert external kernel format to internal format."""

        return (
            recurrent_kernel_ext.reshape(
                self.config.num_heads,
                self.config.head_dim,
                self.config.num_gates,
                self.config.head_dim,
            )
            .transpose(0, 2, 3, 1)
            .reshape(
                self.config.num_heads,
                self.config.num_gates * self.config.head_dim,
                self.config.head_dim,
            )
        )

    def _recurrent_kernel_int2ext(self, recurrent_kernel_int: jnp.ndarray):
        """Convert internal kernel format to external format."""
        return jnp.transpose(
            recurrent_kernel_int.reshape(
                self.config.num_heads,
                self.config.num_gates,
                self.config.head_dim,
                self.config.head_dim,
            ),
            (0, 3, 1, 2),
        )

    def _bias_ext2int(self, bias_ext: jnp.ndarray) -> jnp.ndarray:
        """Convert external bias format to internal format."""
        return jnp.reshape(
            jnp.transpose(
                bias_ext.reshape(
                    self.config.num_heads, self.config.num_gates, self.config.head_dim
                ),
                axes=(1, 0, 2),
            ),
            shape=-1,
        )

    def _bias_int2ext(self, bias_int: jnp.ndarray) -> jnp.ndarray:
        """Convert internal bias format to external format."""
        return jnp.transpose(
            bias_int.reshape(
                self.config.num_gates, self.config.num_heads, self.config.head_dim
            ),
            (1, 0, 2),
        )

    def _impl(self, input: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        """Implementation of forward pass for full sequence."""
        # Convert internal parameters to formats expected by slstm_forward
        rk_internal = self._recurrent_kernel_ext2int(self._recurrent_kernel_)
        bias_internal = self._bias_ext2int(self._bias_)

        return slstm_forward(
            input,
            state,
            rk_internal,
            bias_internal,
            self.pointwise,
        )[0]

    def _impl_step(self, input: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        """Implementation of forward pass for single step."""
        # Convert internal parameters to formats expected by slstm_forward_step
        rk_internal = self._recurrent_kernel_ext2int(self._recurrent_kernel_)
        bias_internal = self._bias_ext2int(self._bias_)

        # Call vanilla implementation (no CUDA)
        return slstm_forward_step(
            input,
            state,
            rk_internal,
            bias_internal,
            self.pointwise,
        )[0]


class sLSTMCell(nnx.Module):
    """Factory class for sLSTM cell that returns vanilla implementation."""

    config_class = sLSTMCellConfig

    def __new__(cls, config: sLSTMCellConfig, rngs: nnx.Rngs, dtype=jnp.float32):
        # Override config to ensure vanilla backend
        config.backend = "vanilla"
        return sLSTMCell_vanilla(config, rngs=rngs, dtype=dtype)
