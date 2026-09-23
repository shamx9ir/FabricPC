"""
Statistical Comparison: PC-ALM vs Backpropagation on CIFAR-10
==============================================================

Runs multiple independent training trials for both PC-ALM and backprop on
CIFAR-10, then performs statistical analysis to compare test accuracies.

PC-ALM (Augmented Lagrangian Predictive Coding) augments standard PC inference
with per-layer Lagrange multipliers (dual variables) that accumulate prediction
errors across inference steps. At convergence in linear networks, the duals
recover exact backpropagation gradients. In nonlinear networks, PC-ALM closes
the PC-BP gap, especially in deep narrow regimes where standard PC underperforms.

Reference: Seely & Gould, "Augmented Lagrangian Predictive Coding", arXiv:2605.31022

Architecture (mini-ResNet, identical topology, different activations)::

    stem:   input(32,32,3) ──→ conv(32,32,32, 3x3, s=1, act)
    block1: (32,32,32, s=1) — same-dim skip (no projection)
    block2: (16,16,64, s=2) — 1x1 projection skip
    block3: (8,8,128, s=2)  — 1x1 projection skip
    head:   avgpool(128, global) ──→ linear(10, softmax+CE)

    ALM activations:      tanh  (bounded, smooth — best for iterative inference)
    Backprop activations: relu  (standard — best for end-to-end autodiff)
    Skip path always uses IdentityActivation in both arms.

Each residual block:  conv_a(3x3, act) -> conv_b(3x3, act) -> skip_sum
Approximate parameters: ~310K

Key differences from the MNIST comparison:
- Mini-ResNet with skip connections — enough capacity for CIFAR-10 (~65-75% expected)
- ALM uses tanh — bounded, smooth, non-zero gradients everywhere (no dead-neuron stalls)
- Backprop uses ReLU — standard choice, avoids vanishing gradients in end-to-end training
- Increased inference steps (50) and rate (0.1) for stable convergence with conv layers

Reports:
- Per-trial accuracy results table
- Mean +/- standard error for each method
- Paired t-test with p-value
- Cohen's d effect size
- Power analysis: estimated n_trials needed for significance

Usage:
    python examples/PC_backprop_compare_cifar10.py                       # 5 trials, 30 epochs
    python examples/PC_backprop_compare_cifar10.py --n_trials 3          # faster: 3 trials
    python examples/PC_backprop_compare_cifar10.py --num_epochs 50 --augment  # higher accuracy
    python examples/PC_backprop_compare_cifar10.py --verbose             # per-epoch output
"""

import jax
import argparse
import numpy as np
import optax

from fabricpc.nodes import ConvNode, Linear, IdentityNode, AvgPool, SkipConnection
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    TanhActivation,
    SoftmaxActivation,
    ReLUActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceALM
from fabricpc.core.initializers import XavierInitializer
from fabricpc.training import train_pcn, evaluate_pcn
from fabricpc.training.train_backprop import train_backprop, evaluate_backprop
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.utils.data.dataloader import Cifar10Loader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")


# =============================================================================
# Data Augmentation
# =============================================================================


class AugmentedCifar10Loader:
    """Wraps Cifar10Loader with random horizontal flip and random crop+pad."""

    def __init__(self, base_loader, seed=42, pad=4):
        self.base_loader = base_loader
        self.seed = seed
        self.pad = pad
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        pad = self.pad
        for images, labels in self.base_loader:
            flip_mask = rng.random(images.shape[0]) > 0.5
            images[flip_mask] = images[flip_mask, :, ::-1, :]

            padded = np.pad(
                images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect"
            )
            B, H, W, C = images.shape
            crop_y = rng.integers(0, 2 * pad + 1, size=B)
            crop_x = rng.integers(0, 2 * pad + 1, size=B)
            for i in range(B):
                images[i] = padded[
                    i, crop_y[i] : crop_y[i] + H, crop_x[i] : crop_x[i] + W, :
                ]

            yield images, labels

    def __len__(self):
        return len(self.base_loader)


# =============================================================================
# Model Builders
# =============================================================================


def _make_residual_block(prev_node, channels, stride, block_name, weight_init, activation):
    """Build one residual block: conv_a -> conv_b -> skip_sum.

    The skip path uses IdentityActivation (no saturation, clean gradient flow).
    A 1x1 projection conv is added to the skip path whenever spatial dims or
    channel count change (stride > 1 or in_channels != channels).

    Returns:
        (nodes_list, edges_list, skip_node)
    """
    in_h, in_w, in_channels = prev_node._shape
    out_h = in_h // stride if stride > 1 else in_h
    out_w = in_w // stride if stride > 1 else in_w

    conv_a = ConvNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name=f"{block_name}_conv_a",
    )
    conv_b = ConvNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name=f"{block_name}_conv_b",
    )
    skip_sum = SkipConnection(
        shape=(out_h, out_w, channels),
        name=f"{block_name}_skip_sum",
    )

    nodes = [conv_a, conv_b, skip_sum]
    edges = [
        Edge(source=prev_node, target=conv_a.slot("in")),
        Edge(source=conv_a, target=conv_b.slot("in")),
        Edge(source=conv_b, target=skip_sum.slot("in")),
    ]

    needs_proj = (stride != 1) or (in_channels != channels)
    if needs_proj:
        conv_skip = ConvNode(
            shape=(out_h, out_w, channels),
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=IdentityActivation(),
            weight_init=weight_init,
            name=f"{block_name}_skip",
        )
        nodes.append(conv_skip)
        edges.append(Edge(source=prev_node, target=conv_skip.slot("in")))
        edges.append(Edge(source=conv_skip, target=skip_sum.slot("skip")))
    else:
        edges.append(Edge(source=prev_node, target=skip_sum.slot("skip")))

    return nodes, edges, skip_sum


def build_mini_resnet(
    activation,
    *,
    infer_steps=50,
    eta_infer=0.1,
    alpha=1.0,
    rho=1.0,
    weight_credit_timing="pre_dual_energy",
):
    """Build a mini-ResNet for CIFAR-10 (stem + 3 residual blocks).

    Architecture::

        input(32,32,3)
            -> stem  conv(32,32,32, 3x3, s=1, act)
            -> block1 (32,32,32, s=1)  -- same-dim skip (no projection)
            -> block2 (16,16,64, s=2)  -- 1x1 projection skip
            -> block3 (8,8,128, s=2)   -- 1x1 projection skip
            -> avgpool(128, global)
            -> linear(10, softmax+CE)

    Approximate parameters: ~310K
    Expected accuracy: 65-75% with 30 epochs (no augment); 75-82% with augment.

    Args:
        activation: Activation for conv layers (tanh for ALM, relu for backprop).
        infer_steps: Number of inference steps (ignored by backprop).
        eta_infer: Inference step size (ignored by backprop).
        alpha: ALM dual step size (default: 1.0).
        rho: ALM penalty strength (default: 1.0).
        weight_credit_timing: When to snapshot duals for weight gradients.
            "pre_dual_energy" (default) — duals from the last primal step.
            "post_dual_energy" — one extra dual update after the final primal.

    Returns:
        GraphStructure ready for initialize_params().
    """
    weight_init = XavierInitializer()

    input_node = IdentityNode(shape=(32, 32, 3), name="input")

    stem = ConvNode(
        shape=(32, 32, 32),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    # 3 stages: (channels, stride)
    stage_configs = [(32, 1), (64, 2), (128, 2)]
    prev = stem
    for stage_idx, (channels, stride) in enumerate(stage_configs, 1):
        nodes, edges, prev = _make_residual_block(
            prev_node=prev,
            channels=channels,
            stride=stride,
            block_name=f"s{stage_idx}",
            weight_init=weight_init,
            activation=activation,
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)

    avg_pool = AvgPool(shape=(128,), name="avgpool", global_pool=True)
    all_nodes.append(avg_pool)
    all_edges.append(Edge(source=prev, target=avg_pool.slot("in")))

    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=XavierInitializer(),
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=avg_pool, target=output.slot("in")))

    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceALM(
            eta_infer=eta_infer, infer_steps=infer_steps,
            alpha=alpha, rho=rho,
            weight_credit_timing=weight_credit_timing,
        ),
    )
    return structure


def create_alm_model(
    rng_key,
    *,
    infer_steps=50,
    eta_infer=0.1,
    alpha=1.0,
    rho=1.0,
    weight_credit_timing="pre_dual_energy",
):
    """PC-ALM model — tanh activations for stable iterative inference.

    Tanh is bounded and has non-zero gradients everywhere, which prevents
    dead-neuron stalls during the iterative ALM inference phase.
    """
    structure = build_mini_resnet(
        TanhActivation(), infer_steps=infer_steps, eta_infer=eta_infer,
        alpha=alpha, rho=rho, weight_credit_timing=weight_credit_timing,
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def create_backprop_model(
    rng_key,
    *,
    infer_steps=50,
    eta_infer=0.1,
    alpha=1.0,
    rho=1.0,
    weight_credit_timing="pre_dual_energy",
):
    """Backprop model — ReLU activations (standard for end-to-end training).

    ReLU avoids the vanishing-gradient problem that tanh can introduce when
    training deep networks with end-to-end autodiff.
    """
    structure = build_mini_resnet(
        ReLUActivation(), infer_steps=infer_steps, eta_infer=eta_infer,
        alpha=alpha, rho=rho, weight_credit_timing=weight_credit_timing,
    )
    params = initialize_params(structure, rng_key)
    return params, structure


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of PC-ALM vs Backprop on CIFAR-10"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of independent training trials per method (default: 5)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=30,
        help="Training epochs per trial (default: 30)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for AdamW (default: 0.001)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW (default: 0.01)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=50,
        help="ALM inference steps per batch (default: 50)",
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=0.1,
        help="ALM inference step size (default: 0.1)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="ALM dual step size — controls how fast multipliers accumulate errors (default: 1.0)",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=1.0,
        help="ALM penalty strength — should match energy precision (default: 1.0)",
    )
    parser.add_argument(
        "--weight_credit_timing",
        type=str,
        default="pre_dual_energy",
        choices=["pre_dual_energy", "post_dual_energy"],
        help="When to snapshot duals for weight gradients: "
        "'pre_dual_energy' uses duals from the last primal step, "
        "'post_dual_energy' adds one extra dual update (default: pre_dual_energy)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (random crop + horizontal flip)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output for each trial",
    )
    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()

    train_config = {"num_epochs": args.num_epochs}
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)

    print("=" * 70)
    print("Statistical Comparison: PC-ALM vs Backpropagation")
    print("=" * 70)
    print("Dataset: CIFAR-10")
    print(
        "Architecture: mini-ResNet  stem(32) -> block1(32,s=1) -> block2(64,s=2)"
        " -> block3(128,s=2) -> avgpool -> 10  (~310K params)"
    )
    print("ALM activations: tanh | Backprop activations: relu")
    print(f"Inference steps: {args.infer_steps}  |  eta_infer: {args.eta_infer}")
    print(f"ALM alpha: {args.alpha}  |  ALM rho: {args.rho}")
    print(f"Weight credit timing: {args.weight_credit_timing}")
    print(
        f"Epochs: {args.num_epochs}  |  Batch size: {args.batch_size}"
        f"  |  LR: {args.lr}  |  Augment: {args.augment}"
    )
    print(f"Trials: {args.n_trials}")
    print()

    # Capture ALM hyperparams in closures for model factories
    infer_steps = args.infer_steps
    eta_infer = args.eta_infer
    alpha = args.alpha
    rho = args.rho
    weight_credit_timing = args.weight_credit_timing

    arm_alm = ExperimentArm(
        name="ALM",
        model_factory=lambda rng: create_alm_model(
            rng, infer_steps=infer_steps, eta_infer=eta_infer,
            alpha=alpha, rho=rho,
            weight_credit_timing=weight_credit_timing,
        ),
        train_fn=train_pcn,
        eval_fn=evaluate_pcn,
        optimizer=optimizer,
        train_config=train_config,
    )

    arm_bp = ExperimentArm(
        name="Backprop",
        model_factory=lambda rng: create_backprop_model(
            rng, infer_steps=infer_steps, eta_infer=eta_infer,
            alpha=alpha, rho=rho,
            weight_credit_timing=weight_credit_timing,
        ),
        train_fn=train_backprop,
        eval_fn=evaluate_backprop,
        optimizer=optimizer,
        train_config=train_config,
    )

    batch_size = args.batch_size
    augment = args.augment

    def data_loader_factory(seed):
        base_train = Cifar10Loader("train", batch_size=batch_size, shuffle=True, seed=seed)
        train_loader = (
            AugmentedCifar10Loader(base_train, seed=seed) if augment else base_train
        )
        test_loader = Cifar10Loader("test", batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    experiment = ABExperiment(
        arm_a=arm_alm,
        arm_b=arm_bp,
        metric="accuracy",
        data_loader_factory=data_loader_factory,
        n_trials=args.n_trials,
        verbose=args.verbose,
    )

    results = experiment.run()
    results.print_summary()


if __name__ == "__main__":
    main()
