"""
Linear node with pre-synaptic activation for predictive coding networks.

LinearPreAct applies the activation function to its inputs *before* the
weight matrix, matching the architecture used in the PC-ALM reference
implementation (Seely & Gould, arXiv:2605.31022):

    z_mu = W @ activation(x_in) + b + x_skip

This differs from the standard ``Linear`` node which applies activation
*post-synaptically* (after the weight matrix):

    z_mu = activation(W @ x_in + b)

Two input slots:
  - "in"   (is_variance_scalable=True):  receives the transform path input.
            Has a weight matrix and is scaled by muPC.
  - "skip" (is_variance_scalable=False): receives the identity skip path.
            No weight matrix; passes through at scale 1.0.

To reproduce the PC-ALM paper's residual MLP:
  - First layer: LinearPreAct with IdentityActivation (no activation on raw input)
  - Hidden layers: LinearPreAct with ReLUActivation and skip connections
  - Output layer: LinearPreAct with ReLUActivation, no skip

Example — a pc-alm-style residual chain::

    source = IdentityNode(shape=(784,), name="x")

    first = LinearPreAct(shape=(W,), activation=IdentityActivation(),
                         name="h0", use_bias=False)
    edges = [Edge(source=source, target=first.slot("in"))]

    prev = first
    for i in range(1, depth - 1):
        layer = LinearPreAct(shape=(W,), activation=ReLUActivation(),
                             name=f"h{i}", use_bias=False)
        edges += [
            Edge(source=prev, target=layer.slot("in")),    # transform path
            Edge(source=prev, target=layer.slot("skip")),   # identity skip
        ]
        prev = layer

    output = LinearPreAct(shape=(10,), activation=ReLUActivation(),
                          name="y", use_bias=False)
    edges.append(Edge(source=prev, target=output.slot("in")))
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import (
    NodeBase,
    SlotSpec,
    FlattenInputMixin,
)
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, KaimingInitializer, initialize

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class LinearPreAct(FlattenInputMixin, NodeBase):
    """
    Linear node with pre-synaptic activation.

    Forward computation::

        z_mu = W @ activation(x_in) + b + x_skip

    The activation is applied to the inputs *before* the weight matrix,
    matching the PC-ALM paper's ``block_pred`` function.

    Two input slots:
      - ``"in"``   (variance-scalable): transform path with weight matrix
      - ``"skip"`` (non-scalable, skip connection): identity bypass path

    The skip slot is optional: if no edge connects to it, the node acts
    as a pure pre-synaptic linear layer without a residual connection.

    Args:
        shape: Output shape tuple (excluding batch dimension)
        name: Node name
        activation: ActivationBase instance applied to inputs before matmul
            (default: IdentityActivation). Use IdentityActivation for the
            first layer of a pc-alm chain (no activation on raw input).
        energy: EnergyFunctional instance (default: GaussianEnergy)
        use_bias: Whether to use bias (default: True)
        flatten_input: If True, flatten all input dims for dense behavior
        weight_init: InitializerBase for weights (default: KaimingInitializer)
        latent_init: InitializerBase for latent states
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: ActivationBase = IdentityActivation(),
        energy: EnergyFunctional = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: InitializerBase = KaimingInitializer(),
        latent_init: InitializerBase = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            flatten_input=flatten_input,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Two slots: 'in' for the transform path, 'skip' for identity bypass."""
        return {
            "in": SlotSpec(name="in", is_multi_input=True, is_variance_scalable=True),
            "skip": SlotSpec(
                name="skip",
                is_multi_input=True,
                is_variance_scalable=False,
                is_skip_connection=True,
            ),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize weight matrices for "in" slot edges only.
        Skip slot edges are identity (no parameters).
        """
        if config is None:
            config = {}

        flatten_input = config.get("flatten_input", False)

        key_w, key_b = jax.random.split(key)

        # Weight matrices only for "in" slot edges
        in_slot_shapes = {k: v for k, v in input_shapes.items() if ":in" in k}

        weights_dict = {}
        rand_key_w = dict(
            zip(in_slot_shapes.keys(), jax.random.split(key_w, len(in_slot_shapes)))
        )

        for edge_key, in_shape in in_slot_shapes.items():
            if flatten_input:
                in_numel = int(np.prod(in_shape))
                out_numel = int(np.prod(node_shape))
                weight_shape = (in_numel, out_numel)
            else:
                in_features = in_shape[-1]
                out_features = node_shape[-1]
                weight_shape = (in_features, out_features)

            weights_dict[edge_key] = initialize(
                rand_key_w[edge_key], weight_shape, weight_init
            )

        # Bias
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            b = jnp.zeros(bias_shape)

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        """
        Pre-synaptic activation forward pass.

        Computes: z_mu = W @ activation(x_in) + b + x_skip

        The activation is applied to each "in" slot input *before* the
        weight matrix multiplication. Skip slot inputs are summed
        without transformation.
        """
        # Separate inputs by slot
        in_inputs = {k: v for k, v in inputs.items() if ":in" in k}
        skip_inputs = {k: v for k, v in inputs.items() if ":skip" in k}

        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        flatten_input = node_info.node_config.get("flatten_input", False)

        # Apply activation to "in" inputs BEFORE matmul (pre-synaptic)
        activation = node_info.activation
        activated_inputs = {
            k: type(activation).forward(v, activation.config)
            for k, v in in_inputs.items()
        }

        # Linear transform on activated inputs
        if flatten_input:
            pre_activation = FlattenInputMixin.compute_linear(
                activated_inputs, params.weights, batch_size, out_shape
            )
        else:
            pre_activation = jnp.zeros((batch_size,) + out_shape)
            for edge_key, x in activated_inputs.items():
                pre_activation = pre_activation + jnp.matmul(
                    x, params.weights[edge_key]
                )

        # Add bias
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Sum skip inputs (identity, no transform)
        skip_sum = None
        for x in skip_inputs.values():
            skip_sum = x if skip_sum is None else skip_sum + x

        # Residual sum: z_mu = W @ activation(x_in) + b + x_skip
        z_mu = pre_activation + skip_sum if skip_sum is not None else pre_activation

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        return state
