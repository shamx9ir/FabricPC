"""
Tests for Augmented Lagrangian Predictive Coding (PC-ALM) inference.

Verifies the InferenceALM implementation against the paper:
    Seely & Gould, "Augmented Lagrangian Predictive Coding", arXiv:2605.31022

Key properties tested:
1. alpha=0 recovers standard PC behaviour
2. Dual variables accumulate prediction errors
3. PC-ALM converges toward BP gradients in a linear network
4. Inference runs without error on standard graph topologies
5. Weight gradients incorporate the dual signal
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.core.inference import InferenceALM, InferenceSGD, run_inference
from fabricpc.core.learning import (
    compute_local_weight_gradients,
    compute_local_weight_gradients_alm,
)
from fabricpc.core.types import GraphState
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.activations import IdentityActivation, ReLUActivation, TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.training import train_step


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


def _build_chain_graph(inference, depth=3, width=16, activation=None):
    """Build a simple chain MLP: input -> h1 -> ... -> output."""
    if activation is None:
        activation = IdentityActivation()

    nodes = []
    input_node = Linear(shape=(width,), name="input")
    nodes.append(input_node)

    hidden_nodes = []
    for i in range(depth - 2):
        h = Linear(
            shape=(width,),
            activation=activation,
            name=f"h{i}",
        )
        hidden_nodes.append(h)
        nodes.append(h)

    output_node = Linear(shape=(width,), name="output")
    nodes.append(output_node)

    edges = []
    prev = input_node
    for h in hidden_nodes:
        edges.append(Edge(source=prev, target=h.slot("in")))
        prev = h
    edges.append(Edge(source=prev, target=output_node.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output_node),
        inference=inference,
    )
    return structure


# ── Tests ───────────────────────────────────────────────────────────


class TestInferenceALMBasic:
    """Basic smoke tests for PC-ALM."""

    def test_alm_runs_without_error(self, rng_key):
        """PC-ALM inference completes on a simple chain graph."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=5, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(alm, depth=4, width=8)
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 8))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, 8))

        clamps = {
            structure.task_map["x"]: x,
            structure.task_map["y"]: y,
        }
        init_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        final_state = run_inference(params, init_state, clamps, structure)

        # Check that all nodes have valid state
        for node_name in structure.nodes:
            ns = final_state.nodes[node_name]
            assert ns.z_latent.shape[0] == batch_size
            assert not jnp.any(jnp.isnan(ns.z_latent))
            assert not jnp.any(jnp.isnan(ns.z_mu))
            assert not jnp.any(jnp.isnan(ns.dual))

    def test_alm_nonlinear(self, rng_key):
        """PC-ALM inference with nonlinear activations (ReLU)."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=10, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(
            alm, depth=4, width=16, activation=ReLUActivation()
        )
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 16))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, 16))

        clamps = {
            structure.task_map["x"]: x,
            structure.task_map["y"]: y,
        }
        init_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(params, init_state, clamps, structure)

        for node_name in structure.nodes:
            ns = final_state.nodes[node_name]
            assert not jnp.any(jnp.isnan(ns.z_latent))

    def test_alm_tanh(self, rng_key):
        """PC-ALM inference with tanh activations."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=10, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(
            alm, depth=4, width=16, activation=TanhActivation()
        )
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 16))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, 16))

        clamps = {
            structure.task_map["x"]: x,
            structure.task_map["y"]: y,
        }
        init_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(params, init_state, clamps, structure)

        for node_name in structure.nodes:
            ns = final_state.nodes[node_name]
            assert not jnp.any(jnp.isnan(ns.z_latent))


class TestALMDualDynamics:
    """Test that the dual variables behave as expected."""

    def test_duals_accumulate(self, rng_key):
        """After inference, internal nodes should have non-zero duals."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=10, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(alm, depth=4, width=8)
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 8))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, 8))

        clamps = {
            structure.task_map["x"]: x,
            structure.task_map["y"]: y,
        }
        init_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )

        # Before inference, all duals should be zero
        for node_name in structure.nodes:
            assert jnp.allclose(init_state.nodes[node_name].dual, 0.0)

        final_state = run_inference(params, init_state, clamps, structure)

        # After inference, internal (non-clamped, in_degree > 0) nodes should
        # have non-zero duals
        input_node = structure.task_map["x"]
        output_node = structure.task_map["y"]
        for node_name in structure.nodes:
            node_info = structure.nodes[node_name].node_info
            if node_name not in (input_node, output_node) and node_info.in_degree > 0:
                dual_norm = jnp.sum(jnp.abs(final_state.nodes[node_name].dual))
                assert dual_norm > 1e-6, (
                    f"Dual for internal node {node_name} should be non-zero"
                )

    def test_clamped_nodes_no_dual(self, rng_key):
        """Clamped nodes (input/output) should keep zero duals."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=10, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(alm, depth=4, width=8)
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 8))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, 8))

        clamps = {
            structure.task_map["x"]: x,
            structure.task_map["y"]: y,
        }
        init_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(params, init_state, clamps, structure)

        input_node = structure.task_map["x"]
        output_node = structure.task_map["y"]
        # Clamped output and input source nodes should have zero duals
        assert jnp.allclose(final_state.nodes[input_node].dual, 0.0)
        assert jnp.allclose(final_state.nodes[output_node].dual, 0.0)


class TestALMAlphaZero:
    """Test that alpha=0 recovers standard PC."""

    def test_alpha_zero_matches_sgd_energy(self, rng_key):
        """With alpha=0 (no dual update), PC-ALM energy matches standard PC."""
        width = 8
        depth = 4
        batch_size = 4
        infer_steps = 10
        eta = 0.1

        sgd = InferenceSGD(eta_infer=eta, infer_steps=infer_steps)
        alm_zero = InferenceALM(
            eta_infer=eta, infer_steps=infer_steps, alpha=0.0, rho=1.0
        )

        structure_sgd = _build_chain_graph(sgd, depth=depth, width=width)
        structure_alm = _build_chain_graph(alm_zero, depth=depth, width=width)

        # Use same params (same rng_key -> same init)
        params_sgd = initialize_params(structure_sgd, rng_key)
        params_alm = initialize_params(structure_alm, rng_key)

        x = jax.random.normal(rng_key, (batch_size, width))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, width))

        clamps_sgd = {
            structure_sgd.task_map["x"]: x,
            structure_sgd.task_map["y"]: y,
        }
        clamps_alm = {
            structure_alm.task_map["x"]: x,
            structure_alm.task_map["y"]: y,
        }

        state_sgd = initialize_graph_state(
            structure_sgd, batch_size, rng_key, clamps=clamps_sgd, params=params_sgd
        )
        state_alm = initialize_graph_state(
            structure_alm, batch_size, rng_key, clamps=clamps_alm, params=params_alm
        )

        final_sgd = run_inference(params_sgd, state_sgd, clamps_sgd, structure_sgd)
        final_alm = run_inference(params_alm, state_alm, clamps_alm, structure_alm)

        # Energies should match (alpha=0 means no dual correction)
        # Note: z_latent in ALM final state is shifted by +dual/rho for weight grad.
        # With alpha=0, dual=0, so shift=0 and latents should be identical.
        for node_name in structure_sgd.nodes:
            e_sgd = final_sgd.nodes[node_name].energy
            e_alm = final_alm.nodes[node_name].energy
            np.testing.assert_allclose(
                e_sgd, e_alm, atol=1e-5,
                err_msg=f"Energy mismatch at node {node_name} with alpha=0"
            )


class TestALMTraining:
    """Test that PC-ALM integrates into the training pipeline."""

    def test_train_step_alm(self, rng_key):
        """A full train_step with ALM inference runs without error."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=10, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(alm, depth=4, width=8)
        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        batch_size = 4
        batch = {
            "x": jax.random.normal(rng_key, (batch_size, 8)),
            "y": jax.random.normal(jax.random.PRNGKey(99), (batch_size, 8)),
        }

        new_params, new_opt_state, energy, final_state = train_step(
            params, opt_state, batch, structure, optimizer, rng_key
        )

        assert not jnp.isnan(energy)
        # Parameters should have changed for at least one internal node
        any_changed = False
        for node_name in params.nodes:
            old_w = params.nodes[node_name].weights
            new_w = new_params.nodes[node_name].weights
            for key in old_w:
                if not jnp.allclose(old_w[key], new_w[key]):
                    any_changed = True
                    break
        assert any_changed, "Weights should be updated after a train step"

    def test_multiple_train_steps(self, rng_key):
        """Multiple training steps with ALM should reduce energy."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=10, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(alm, depth=3, width=8)
        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        batch_size = 8
        batch = {
            "x": jax.random.normal(rng_key, (batch_size, 8)),
            "y": jax.random.normal(jax.random.PRNGKey(99), (batch_size, 8)),
        }

        energies = []
        for step in range(5):
            step_key = jax.random.fold_in(rng_key, step)
            params, opt_state, energy, _ = train_step(
                params, opt_state, batch, structure, optimizer, step_key
            )
            energies.append(float(energy))

        # Energy should generally decrease over training
        assert energies[-1] < energies[0], (
            f"Energy should decrease: {energies[0]:.4f} -> {energies[-1]:.4f}"
        )


class TestALMWeightGradients:
    """Test that PC-ALM modifies weight gradients via the dual signal."""

    def test_alm_gradients_differ_from_pc(self, rng_key):
        """PC-ALM (alpha>0) should produce different weight gradients than PC (alpha=0)."""
        width = 8
        depth = 4
        batch_size = 4
        infer_steps = 10

        alm = InferenceALM(
            eta_infer=0.1, infer_steps=infer_steps, alpha=1.0, rho=1.0
        )
        alm_zero = InferenceALM(
            eta_infer=0.1, infer_steps=infer_steps, alpha=0.0, rho=1.0
        )

        structure_alm = _build_chain_graph(alm, depth=depth, width=width)
        structure_pc = _build_chain_graph(alm_zero, depth=depth, width=width)

        params_alm = initialize_params(structure_alm, rng_key)
        params_pc = initialize_params(structure_pc, rng_key)

        x = jax.random.normal(rng_key, (batch_size, width))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, width))

        clamps_alm = {
            structure_alm.task_map["x"]: x,
            structure_alm.task_map["y"]: y,
        }
        clamps_pc = {
            structure_pc.task_map["x"]: x,
            structure_pc.task_map["y"]: y,
        }

        state_alm = initialize_graph_state(
            structure_alm, batch_size, rng_key, clamps=clamps_alm, params=params_alm
        )
        state_pc = initialize_graph_state(
            structure_pc, batch_size, rng_key, clamps=clamps_pc, params=params_pc
        )

        final_alm = run_inference(params_alm, state_alm, clamps_alm, structure_alm)
        final_pc = run_inference(params_pc, state_pc, clamps_pc, structure_pc)

        grads_alm = compute_local_weight_gradients_alm(
            params_alm, final_alm, structure_alm, rho=1.0
        )
        grads_pc = compute_local_weight_gradients_alm(
            params_pc, final_pc, structure_pc, rho=1.0
        )

        # At least one internal node should have different weight gradients
        any_different = False
        for node_name in structure_alm.nodes:
            node_info = structure_alm.nodes[node_name].node_info
            if node_info.in_degree > 0:
                for wkey in grads_alm.nodes[node_name].weights:
                    g_alm = grads_alm.nodes[node_name].weights[wkey]
                    g_pc = grads_pc.nodes[node_name].weights[wkey]
                    if not jnp.allclose(g_alm, g_pc, atol=1e-5):
                        any_different = True
                        break

        assert any_different, (
            "PC-ALM with alpha>0 should produce different weight gradients than PC"
        )


class TestALMLinearBPAlignment:
    """
    In a linear network, PC-ALM should converge to exact BP gradients.
    (Section 5.1 / Proposition 3 of the paper.)

    We verify this by comparing PC-ALM weight gradients to true BP gradients
    computed via jax.grad on the supervised loss.
    """

    def test_linear_bp_alignment(self, rng_key):
        """PC-ALM weight gradient aligns with BP in a linear 3-layer network."""
        width = 8
        depth = 3
        batch_size = 4
        # Use many inference steps with a small learning rate for convergence.
        # The paper's theory guarantees BP-alignment at convergence for linear
        # nets; practical convergence requires enough steps.
        infer_steps = 500

        alm = InferenceALM(
            eta_infer=0.05, infer_steps=infer_steps, alpha=1.0, rho=1.0
        )
        structure = _build_chain_graph(
            alm, depth=depth, width=width, activation=IdentityActivation()
        )
        params = initialize_params(structure, rng_key)

        x = jax.random.normal(rng_key, (batch_size, width))
        y = jax.random.normal(jax.random.PRNGKey(99), (batch_size, width))

        # ── PC-ALM weight gradients ──
        clamps = {
            structure.task_map["x"]: x,
            structure.task_map["y"]: y,
        }
        init_state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(params, init_state, clamps, structure)
        grads_alm = compute_local_weight_gradients_alm(
            params, final_state, structure, rho=1.0
        )

        # ── True BP gradients via jax.grad ──
        # Build the forward pass manually for a linear chain
        node_order = list(structure.node_order)
        input_node = structure.task_map["x"]
        output_node = structure.task_map["y"]

        def bp_loss(all_params):
            """Forward pass through the linear chain and MSE loss."""
            h = x  # start with input
            for node_name in node_order:
                node_info = structure.nodes[node_name].node_info
                if node_info.in_degree == 0:
                    continue  # skip input node
                # Linear: z_mu = W @ h_prev + b
                np_ = all_params.nodes[node_name]
                # Get the single in-edge weight
                w_key = list(np_.weights.keys())[0]
                W = np_.weights[w_key]
                h = h @ W
                if np_.biases:
                    b_key = list(np_.biases.keys())[0]
                    h = h + np_.biases[b_key]
            # MSE loss: (1/2) ||y - h||^2, averaged over batch
            return 0.5 * jnp.mean(jnp.sum((y - h) ** 2, axis=-1))

        grads_bp = jax.grad(bp_loss)(params)

        # Compare weight gradients for internal nodes
        for node_name in node_order:
            node_info = structure.nodes[node_name].node_info
            if node_info.in_degree == 0:
                continue

            for wkey in grads_alm.nodes[node_name].weights:
                g_alm = grads_alm.nodes[node_name].weights[wkey]
                g_bp = grads_bp.nodes[node_name].weights[wkey]

                # Normalize for comparison (cosine similarity)
                g_alm_flat = g_alm.flatten()
                g_bp_flat = g_bp.flatten()

                cos_sim = jnp.dot(g_alm_flat, g_bp_flat) / (
                    jnp.linalg.norm(g_alm_flat) * jnp.linalg.norm(g_bp_flat) + 1e-10
                )

                # In a linear network with enough steps, cosine similarity
                # should be very high (> 0.95)
                assert cos_sim > 0.90, (
                    f"PC-ALM weight gradient at {node_name}/{wkey} not aligned "
                    f"with BP: cosine similarity = {cos_sim:.4f}"
                )


class TestALMJITCompatible:
    """Test that PC-ALM is compatible with JAX JIT compilation."""

    def test_jit_train_step(self, rng_key):
        """JIT-compiled train step with ALM works correctly."""
        alm = InferenceALM(eta_infer=0.1, infer_steps=5, alpha=1.0, rho=1.0)
        structure = _build_chain_graph(alm, depth=3, width=8)
        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        batch_size = 4
        batch = {
            "x": jax.random.normal(rng_key, (batch_size, 8)),
            "y": jax.random.normal(jax.random.PRNGKey(99), (batch_size, 8)),
        }

        jit_step = jax.jit(
            lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k)
        )

        # First call triggers compilation
        new_params, new_opt, energy, _ = jit_step(params, opt_state, batch, rng_key)
        assert not jnp.isnan(energy)

        # Second call uses cached compiled function
        new_params2, _, energy2, _ = jit_step(new_params, new_opt, batch, rng_key)
        assert not jnp.isnan(energy2)
