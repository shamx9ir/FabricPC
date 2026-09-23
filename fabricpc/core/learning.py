from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeParams, NodeState
from fabricpc.core.inference import gather_inputs
from fabricpc.core.scaling import scale_inputs, scale_weight_grads
from fabricpc.core.state_ops import update_node_in_state


def compute_local_weight_gradients(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """
    Compute local weight gradients for each node using its own error signal.

    This implements the local Hebbian learning rule for predictive coding.
    muPC scaling is applied here (pre-scale inputs, post-scale weight
    gradients), keeping node methods (forward_and_weight_grads) scaling-unaware.

    The returned gradients are batch-summed: each node's
    ``forward_and_weight_grads`` differentiates the sum of its per-sample
    energies over the batch. Sums are associative, so they survive data
    sharding and gradient accumulation unchanged. The trainer's
    ``fabricpc.training.pc_weight_gradients`` divides them once by the
    prediction count ``fabricpc.training.grad_denominator``, whose docstring
    carries the rule for microbatching and padded positions. Optimizer-facing
    callers use that function; this one serves inspection and composition.

    Args:
        params: Current model parameters
        final_state: Converged state after inference
        structure: Graph structure

    Returns:
        GraphParams containing batch-summed gradients for the parameters
    """
    gradients = {}

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        # Source nodes have no weights, but need empty gradient dict for consistency
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)

        node_class = node_info.node_class
        sc = node_info.scaling_config

        # Pre-scale inputs by muPC forward scaling factors
        scaled_inputs = scale_inputs(in_edges_data, sc)

        # Compute local gradients using node's method (pure autodiff)
        node_state, grad_params = node_class.forward_and_weight_grads(
            params.nodes[node_name],
            scaled_inputs,
            final_state.nodes[node_name],
            node_info,
        )

        # Post-scale weight gradients by muPC factors
        grad_params = scale_weight_grads(grad_params, sc)

        # Store gradients
        gradients[node_name] = grad_params

    # convert to GraphParams
    params_gradients = GraphParams(nodes=gradients)

    return params_gradients


def compute_local_weight_gradients_alm(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
    rho: float,
) -> GraphParams:
    """
    Compute local weight gradients under the augmented Lagrangian (PC-ALM).

    For each internal node, shifts only that node's z_latent by +dual/rho
    before computing the standard PC energy and its parameter gradient.
    By the completed-square identity (Eq. 9 of arXiv:2605.31022), this
    yields the augmented Lagrangian weight gradient:

        nabla_{W_i} L_rho = -(rho * r_i + lambda_i) * d sigma(W_i h_{i-1}) / dW_i

    Predecessors' z_latent values are NOT shifted, so the inputs to each
    node's forward pass remain the original converged latents.

    Args:
        params: Current model parameters.
        final_state: Converged state after PC-ALM inference (with duals).
        structure: Graph structure.
        rho: Augmented Lagrangian penalty parameter.

    Returns:
        GraphParams containing augmented Lagrangian weight gradients.
    """
    import jax.numpy as jnp

    gradients = {}

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        # Gather inputs from ORIGINAL (unshifted) predecessor z_latent values
        in_edges_data = gather_inputs(node_info, structure, final_state)

        node_class = node_info.node_class
        sc = node_info.scaling_config

        # Pre-scale inputs by muPC forward scaling factors
        scaled_inputs = scale_inputs(in_edges_data, sc)

        # Shift THIS node's z_latent by +dual/rho for the AL weight gradient.
        # This makes the energy E(z_latent + dual/rho, z_mu) = E_ALM, whose
        # gradient w.r.t. W gives the augmented Lagrangian weight gradient.
        ns = final_state.nodes[node_name]
        shifted_z = ns.z_latent + ns.dual / rho
        shifted_node_state = ns._replace(z_latent=shifted_z)

        # Compute local gradients with the shifted z_latent
        node_state, grad_params = node_class.forward_and_weight_grads(
            params.nodes[node_name],
            scaled_inputs,
            shifted_node_state,
            node_info,
        )

        # Post-scale weight gradients by muPC factors
        grad_params = scale_weight_grads(grad_params, sc)

        gradients[node_name] = grad_params

    return GraphParams(nodes=gradients)
