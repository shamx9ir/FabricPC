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

Architecture::

    pixels(784) --> hidden1(128) --> hidden2(64) --> class(10)
     Identity       ReLU             ReLU           Softmax+CE

Comparison: runs standard PC (InferenceSGD) and PC-ALM (InferenceALM) side
by side for the same architecture and hyperparameters.
"""

import jax
import optax
import time

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD, InferenceALM
from fabricpc.core.initializers import XavierInitializer
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


def build_structure(inference):
    """Build the same 4-layer MLP architecture with the given inference algorithm."""
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
    )


def run_experiment(name, inference, rng_key):
    """Train and evaluate a model with the given inference algorithm."""
    structure = build_structure(inference)
    graph_key, train_key, eval_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, graph_key)

    train_loader = MnistLoader(
        "train", batch_size=BATCH_SIZE, tensor_format="flat", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=BATCH_SIZE, tensor_format="flat", shuffle=False
    )

    optimizer = optax.adamw(LR, weight_decay=0.1)

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
    key_pc, key_alm = jax.random.split(master_key)

    # Standard PC
    acc_pc, time_pc = run_experiment(
        "Standard PC (InferenceSGD)",
        InferenceSGD(eta_infer=ETA_INFER, infer_steps=INFER_STEPS),
        key_pc,
    )

    # PC-ALM
    acc_alm, time_alm = run_experiment(
        "PC-ALM (InferenceALM, alpha=1.0, rho=1.0)",
        InferenceALM(
            eta_infer=ETA_INFER,
            infer_steps=INFER_STEPS,
            alpha=1.0,
            rho=1.0,
        ),
        key_alm,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    print(f"  Standard PC : {acc_pc:.2f}% ({time_pc:.2f}s/epoch)")
    print(f"  PC-ALM      : {acc_alm:.2f}% ({time_alm:.2f}s/epoch)")
    print(f"{'='*60}")
