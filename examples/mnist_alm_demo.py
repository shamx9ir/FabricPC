"""
Augmented Lagrangian Predictive Coding — MNIST Demo
====================================================

Train a predictive coding network on MNIST using PC-ALM (Augmented Lagrangian
Method) from:

    Seely & Gould, "Augmented Lagrangian Predictive Coding",
    arXiv:2605.31022

PC-ALM augments standard PC inference with per-layer Lagrange multipliers
(dual variables) that accumulate prediction errors. At convergence, the
duals recover exact backpropagation adjoints, closing the PC–BP gap even
in deep narrow networks.

Three modes are compared:

1. Standard PC (InferenceSGD) — post-synaptic activation, no duals
2. PC-ALM (InferenceALM) — post-synaptic activation, with duals
3. PC-ALM Reference — pre-synaptic activation (LinearPreAct), paper's
   PCALMScaling, and weight_credit_timing="pre_dual_energy", matching
   the architecture and parameterization of the reference implementation.

Architectures::

    Modes 1 & 2 (post-synaptic, shallow MLP):
        pixels(784) --> hidden1(128) --> hidden2(64) --> class(10)
         Identity       ReLU             ReLU           Softmax+CE

    Mode 3 (pre-synaptic, deep residual MLP matching the paper):
        x(784) --> h0(WIDTH) --> h1(WIDTH) --> ... --> h_{D-2}(WIDTH) --> y(10)
         Identity   Identity     ReLU+skip   ...     ReLU+skip           ReLU
                    (no act)     (pre-act)           (pre-act)          (pre-act)

    WIDTH and DEPTH are configurable; defaults: WIDTH=32, DEPTH=8.
"""

import jax
import math
import optax
import time

from fabricpc.nodes import Linear, LinearPreAct, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD, InferenceALM
from fabricpc.core.initializers import NormalInitializer, XavierInitializer
from fabricpc.core.mupc import PCALMScaling
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Shared hyperparameters ---

NUM_EPOCHS = 10
BATCH_SIZE = 200
LR = 0.001
INFER_STEPS = 20
ETA_INFER = 0.05

# --- PC-ALM reference hyperparameters ---

REF_WIDTH = 32
REF_DEPTH = 8   # total layers including first and output


def build_structure(inference, scaling=None):
    """Build a shallow 4-layer MLP (modes 1 & 2)."""
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden1 = Linear(
        shape=(128,),
        activation=ReLUActivation(),
        name="hidden1",
        weight_init=XavierInitializer(),
    )
    hidden2 = Linear(
        shape=(64,),
        activation=ReLUActivation(),
        name="hidden2",
        weight_init=XavierInitializer(),
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )
    return graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=inference,
        scaling=scaling,
    )


def build_pcalm_reference_structure(
    width=REF_WIDTH,
    depth=REF_DEPTH,
    eta_infer=0.25,
    alpha=1.0,
    rho=1.0,
    weight_credit_timing="pre_dual_energy",
):
    """
    Build a deep residual MLP matching the PC-ALM reference implementation.

    Architecture (from pcalm/model.py):
      - Layer 0: LinearPreAct(IdentityActivation) -- no activation on raw input
      - Layers 1..depth-2: LinearPreAct(ReLUActivation) + skip connection
      - Layer depth-1: LinearPreAct(ReLUActivation) -- output, no skip

    Scaling: PCALMScaling reproduces the reference's model_scales():
      - Layer 0: 1/sqrt(input_dim)
      - Hidden:  1/sqrt(width * depth)
      - Output:  1/width

    Uses NormalInitializer(std=1.0) for weights (matching the reference's
    jax.random.normal initialization without baked-in scaling).
    """
    input_dim = 784
    output_dim = 10
    infer_steps = 2 * depth  # paper recommends T = 2*L

    weight_init = NormalInitializer(std=1.0)

    source = IdentityNode(shape=(input_dim,), name="x")

    # Layer 0: no activation on raw input, no skip, flatten input
    first = LinearPreAct(
        shape=(width,),
        activation=IdentityActivation(),
        use_bias=False,
        weight_init=weight_init,
        flatten_input=True,
        name="h0",
    )

    all_nodes = [source, first]
    all_edges = [Edge(source=source, target=first.slot("in"))]

    # Hidden layers: pre-synaptic ReLU + skip connection
    prev = first
    for i in range(1, depth - 1):
        layer = LinearPreAct(
            shape=(width,),
            activation=ReLUActivation(),
            use_bias=False,
            weight_init=weight_init,
            name=f"h{i}",
        )
        all_nodes.append(layer)
        all_edges.append(Edge(source=prev, target=layer.slot("in")))
        all_edges.append(Edge(source=prev, target=layer.slot("skip")))
        prev = layer

    # Output layer: pre-synaptic ReLU, no skip
    output = LinearPreAct(
        shape=(output_dim,),
        activation=ReLUActivation(),
        use_bias=False,
        weight_init=weight_init,
        name="y",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=prev, target=output.slot("in")))

    # Learning rate: eta0 * sqrt(width / depth) per the reference
    param_lr = LR * math.sqrt(width / depth)

    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=source, y=output),
        inference=InferenceALM(
            eta_infer=eta_infer,
            infer_steps=infer_steps,
            alpha=alpha,
            rho=rho,
            weight_credit_timing=weight_credit_timing,
        ),
        scaling=PCALMScaling(depth=depth),
    )

    return structure, param_lr


def run_experiment(name, structure, rng_key, lr=LR):
    """Train and evaluate a model."""
    graph_key, train_key, eval_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, graph_key)

    train_loader = MnistLoader(
        "train", batch_size=BATCH_SIZE, tensor_format="flat", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=BATCH_SIZE, tensor_format="flat", shuffle=False
    )

    optimizer = optax.adamw(lr, weight_decay=0.1)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  {len(structure.nodes)} nodes, {len(structure.edges)} edges, {n_params:,} params")
    print(f"{'='*60}")
    print(f"Training for {NUM_EPOCHS} epochs...")

    start = time.time()
    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config={"num_epochs": NUM_EPOCHS},
        rng_key=train_key,
        verbose=True,
    )
    elapsed = time.time() - start
    avg_time = elapsed / NUM_EPOCHS

    metrics = evaluate_pcn(trained_params, structure, test_loader, {}, eval_key)
    acc = metrics["accuracy"] * 100

    print(f"\n  Results: {acc:.2f}% accuracy, {avg_time:.2f}s/epoch")
    return acc, avg_time


if __name__ == "__main__":
    master_key = jax.random.PRNGKey(0)
    key_pc, key_alm, key_ref = jax.random.split(master_key, 3)

    # 1. Standard PC
    structure_pc = build_structure(
        InferenceSGD(eta_infer=ETA_INFER, infer_steps=INFER_STEPS),
    )
    acc_pc, time_pc = run_experiment(
        "Standard PC (InferenceSGD)",
        structure_pc,
        key_pc,
    )

    # 2. PC-ALM (post-synaptic, same shallow MLP)
    structure_alm = build_structure(
        InferenceALM(
            eta_infer=ETA_INFER,
            infer_steps=INFER_STEPS,
            alpha=1.0,
            rho=1.0,
        ),
    )
    acc_alm, time_alm = run_experiment(
        "PC-ALM (InferenceALM, alpha=1.0, rho=1.0)",
        structure_alm,
        key_alm,
    )

    # 3. PC-ALM Reference (pre-synaptic, deep residual, paper's scaling)
    structure_ref, ref_lr = build_pcalm_reference_structure(
        width=REF_WIDTH,
        depth=REF_DEPTH,
        eta_infer=0.25,
        alpha=1.0,
        rho=1.0,
        weight_credit_timing="pre_dual_energy",
    )
    acc_ref, time_ref = run_experiment(
        f"PC-ALM Reference (LinearPreAct, W={REF_WIDTH}, D={REF_DEPTH})",
        structure_ref,
        key_ref,
        lr=ref_lr,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Standard PC      : {acc_pc:.2f}% ({time_pc:.2f}s/epoch)")
    print(f"  PC-ALM           : {acc_alm:.2f}% ({time_alm:.2f}s/epoch)")
    print(f"  PC-ALM Reference : {acc_ref:.2f}% ({time_ref:.2f}s/epoch)")
    print(f"{'='*60}")
