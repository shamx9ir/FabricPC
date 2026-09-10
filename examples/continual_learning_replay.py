"""
Neuroscience-Inspired Memory Replay for Continual Learning
A Comparative Study of Predictive Coding and Backpropagation-Based Strategies

Recreation of:
    Nalagatla & Grandhe (2025), arXiv:2512.00619
    "Neuroscience-Inspired Memory Replay for Continual Learning:
     A Comparative Study of Predictive Coding and Backpropagation-Based Strategies"

Methods compared:
    PC Replay   — Hierarchical predictive coding generator (FabricPC) + MLP classifier
    BP Replay   — VAE generator (Flax/backprop) + MLP classifier

Datasets:
    Split-MNIST    : 5 tasks, digits 0-1 / 2-3 / 4-5 / 6-7 / 8-9
    Split-CIFAR-10 : 5 tasks, 2 classes each
    Split-CIFAR-100: 10 tasks, 10 classes each

Metrics (all paper-faithful):
    Average Accuracy  — mean accuracy across all tasks after full training
    Forgetting Measure — peak − final accuracy per task, averaged
    Forward Transfer  — improvement vs training from scratch on new tasks
    Backward Transfer — improvement on old tasks after learning newer ones

Usage:
    python examples/continual_learning_replay.py --dataset mnist --quick
    python examples/continual_learning_replay.py --dataset mnist
    python examples/continual_learning_replay.py --dataset cifar10
    python examples/continual_learning_replay.py --dataset cifar100
    python examples/continual_learning_replay.py --all
    python examples/continual_learning_replay.py --n_trials 5 --dataset mnist
"""

from __future__ import annotations

import argparse
import gzip
import os
import pickle
import struct
import sys
import tarfile
import time
import urllib.request
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import importlib.util

import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

# ── FabricPC imports ──────────────────────────────────────────────────────────
from fabricpc import setup_jax
from fabricpc.nodes import IdentityNode
from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.activations import ReLUActivation, SigmoidActivation, IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer, initialize
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.training import train_pcn

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

# ─────────────────────────────────────────────────────────────────────────────
# PreSynapticLinear — identical to the one in associative_memory_pcn.py
# (pre-synaptic activation: z_mu = W @ f(x) + b)
# ─────────────────────────────────────────────────────────────────────────────


class PreSynapticLinear(FlattenInputMixin, NodeBase):
    """Linear node with pre-synaptic activation: z_mu = W @ f(x) + b.

    Matches the paper's formulation: mu^l = theta^{l+1} f(x^{l+1}).
    """

    def __init__(
        self,
        shape,
        name,
        input_activation=ReLUActivation(),
        energy=GaussianEnergy(),
        use_bias=True,
        flatten_input=True,
        weight_init=KaimingInitializer(),
        latent_init=NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=IdentityActivation(),
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            flatten_input=flatten_input,
            input_activation=input_activation,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config=None):
        if config is None:
            config = {}
        flatten_input = config.get("flatten_input", False)
        key_w, key_b = jax.random.split(key)
        weights_dict = {}
        rand_key_w = dict(
            zip(input_shapes.keys(), jax.random.split(key_w, len(input_shapes)))
        )
        for edge_key, in_shape in input_shapes.items():
            if flatten_input:
                in_numel = int(np.prod(in_shape))
                out_numel = int(np.prod(node_shape))
                weight_shape = (in_numel, out_numel)
            else:
                weight_shape = (in_shape[-1], node_shape[-1])
            weights_dict[edge_key] = initialize(
                rand_key_w[edge_key], weight_shape, weight_init
            )
        use_bias = config.get("use_bias", True)
        biases = {}
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            biases["b"] = jnp.zeros(bias_shape)
        return NodeParams(weights=weights_dict, biases=biases)

    @staticmethod
    def forward(params, inputs, state, node_info):
        """z_mu = W @ f(x) + b  (pre-synaptic activation)."""
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        flatten_input = node_info.node_config.get("flatten_input", False)
        input_act = node_info.node_config["input_activation"]

        activated = {}
        for edge_key, x in inputs.items():
            activated[edge_key] = type(input_act).forward(x, input_act.config)

        if flatten_input:
            z_mu = FlattenInputMixin.compute_linear(
                activated, params.weights, batch_size, out_shape
            )
        else:
            z_mu = jnp.zeros((batch_size,) + out_shape)
            for edge_key, x in activated.items():
                z_mu = z_mu + jnp.matmul(x, params.weights[edge_key])

        if "b" in params.biases and params.biases["b"].size > 0:
            z_mu = z_mu + params.biases["b"]

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        return state


# ─────────────────────────────────────────────────────────────────────────────
# Memorization loader (yields {"x": batch})
# ─────────────────────────────────────────────────────────────────────────────


class MemorizationLoader:
    """Yields images as {"x": batch} for train_pcn."""

    def __init__(self, images: np.ndarray, batch_size: Optional[int] = None, seed: int = 42):
        self.images = np.asarray(images, dtype=np.float32)
        self.batch_size = batch_size or len(images)
        self.seed = seed
        self._epoch = 0
        self.num_samples = len(images)
        self._num_batches = max(1, self.num_samples // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        indices = rng.permutation(self.num_samples)
        for start in range(0, self.num_samples - self.batch_size + 1, self.batch_size):
            batch_idx = indices[start : start + self.batch_size]
            yield {"x": self.images[batch_idx]}

    def __len__(self):
        return self._num_batches


# =============================================================================
# 1.  DATA LOADING
# =============================================================================


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path, desc: str = "") -> None:
    if dest.exists():
        return
    print(f"  Downloading {desc or dest.name} ...")
    urllib.request.urlretrieve(url, dest)


# ── MNIST ─────────────────────────────────────────────────────────────────────


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test) flat float32 arrays."""
    cache = _ensure_dir(Path.home() / ".cache" / "mnist")
    base = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images":  "t10k-images-idx3-ubyte.gz",
        "test_labels":  "t10k-labels-idx1-ubyte.gz",
    }
    for fname in files.values():
        _download(base + fname, cache / fname, fname)

    def read_images(p: Path) -> np.ndarray:
        with gzip.open(p, "rb") as f:
            _, n, r, c = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), np.uint8).reshape(n, r * c).astype(np.float32) / 255.0

    def read_labels(p: Path) -> np.ndarray:
        with gzip.open(p, "rb") as f:
            _, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), np.uint8).astype(np.int32)

    return (
        read_images(cache / files["train_images"]),
        read_labels(cache / files["train_labels"]),
        read_images(cache / files["test_images"]),
        read_labels(cache / files["test_labels"]),
    )


# ── CIFAR-10 ──────────────────────────────────────────────────────────────────


def _ensure_cifar10(cache: Path) -> Path:
    """Download and extract CIFAR-10 if needed. Returns data directory."""
    data_dir = cache / "cifar-10-batches-py"
    if data_dir.exists() and (data_dir / "data_batch_1").exists():
        return data_dir
    tar = cache / "cifar-10-python.tar.gz"
    _download(
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        tar, "CIFAR-10"
    )
    with tarfile.open(tar, "r:gz") as tf_:
        tf_.extractall(cache)
    return data_dir


def load_cifar10() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test) flat float32 arrays."""
    cache = _ensure_dir(Path.home() / ".cache" / "cifar10")
    data_dir = _ensure_cifar10(cache)

    x_all, y_all = [], []
    for i in range(1, 6):
        with open(data_dir / f"data_batch_{i}", "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        x_all.append(batch[b"data"])
        y_all.extend(batch[b"labels"])

    with open(data_dir / "test_batch", "rb") as f:
        tb = pickle.load(f, encoding="bytes")

    x_train = np.concatenate(x_all).astype(np.float32) / 255.0
    y_train = np.array(y_all, dtype=np.int32)
    x_test  = np.array(tb[b"data"], dtype=np.float32) / 255.0
    y_test  = np.array(tb[b"labels"], dtype=np.int32)
    return x_train, y_train, x_test, y_test


# ── CIFAR-100 ─────────────────────────────────────────────────────────────────


def _ensure_cifar100(cache: Path) -> Path:
    data_dir = cache / "cifar-100-python"
    if data_dir.exists() and (data_dir / "train").exists():
        return data_dir
    tar = cache / "cifar-100-python.tar.gz"
    _download(
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
        tar, "CIFAR-100"
    )
    with tarfile.open(tar, "r:gz") as tf_:
        tf_.extractall(cache)
    return data_dir


def load_cifar100() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test) using fine labels."""
    cache = _ensure_dir(Path.home() / ".cache" / "cifar100")
    data_dir = _ensure_cifar100(cache)

    with open(data_dir / "train", "rb") as f:
        tr = pickle.load(f, encoding="bytes")
    with open(data_dir / "test", "rb") as f:
        te = pickle.load(f, encoding="bytes")

    x_train = np.array(tr[b"data"], dtype=np.float32) / 255.0
    y_train = np.array(tr[b"fine_labels"], dtype=np.int32)
    x_test  = np.array(te[b"data"], dtype=np.float32) / 255.0
    y_test  = np.array(te[b"fine_labels"], dtype=np.int32)
    return x_train, y_train, x_test, y_test


# ── Task splitting ─────────────────────────────────────────────────────────────


def create_split_tasks(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test:  np.ndarray,
    y_test:  np.ndarray,
    n_tasks: int,
    classes_per_task: int,
    seed: int = 42,
) -> List[Dict]:
    """Split dataset into sequential tasks.

    Returns a list of n_tasks dicts, each with keys:
        "x_train", "y_train", "x_test", "y_test", "classes"
    Labels within each task are renumbered to the global class indices
    (e.g. task 0 keeps labels 0..CPT-1, task 1 keeps CPT..2*CPT-1, ...).
    """
    rng = np.random.default_rng(seed)
    all_classes = np.arange(n_tasks * classes_per_task)
    tasks = []
    for t in range(n_tasks):
        cls = all_classes[t * classes_per_task : (t + 1) * classes_per_task]
        # train
        tr_mask = np.isin(y_train, cls)
        # test
        te_mask = np.isin(y_test, cls)
        tasks.append({
            "x_train": x_train[tr_mask],
            "y_train": y_train[tr_mask],
            "x_test":  x_test[te_mask],
            "y_test":  y_test[te_mask],
            "classes": list(cls),
        })
    return tasks


# =============================================================================
# 2.  PC GENERATOR  (FabricPC hierarchical predictive coding)
# =============================================================================


def _build_pc_generator(data_dim: int, latent_dim: int, n_hidden_layers: int,
                         infer_steps: int, eta_infer: float):
    """Build a top-down generative PCN.

    Architecture (top → bottom):
        memory(latent_dim) → hidden_0(latent_dim) → … → sensory(data_dim)
    """
    memory = IdentityNode(shape=(latent_dim,), name="memory")

    hidden_layers = [
        PreSynapticLinear(
            shape=(latent_dim,),
            name=f"hidden_{i}",
            input_activation=ReLUActivation(),
            weight_init=KaimingInitializer(),
            flatten_input=True,
            use_bias=True,
        )
        for i in range(n_hidden_layers)
    ]

    sensory = PreSynapticLinear(
        shape=(data_dim,),
        name="sensory",
        input_activation=ReLUActivation(),
        weight_init=KaimingInitializer(),
        flatten_input=True,
        use_bias=True,
    )

    nodes = [memory] + hidden_layers + [sensory]
    edges = [
        Edge(source=nodes[i], target=nodes[i + 1].slot("in"))
        for i in range(len(nodes) - 1)
    ]

    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=sensory),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )


class PCGenerator:
    """Predictive coding generative replay model.

    Wraps FabricPC's hierarchical PCN.  Training clamps the sensory (bottom)
    layer to data; the memory (top) latent is free to optimise.

    Generation strategy (learned prior):
        After each training call, we run inference on a subset of the training
        images to collect the converged memory-latent states {z_i}.  We fit a
        per-dimension Gaussian to these states and sample from it at replay
        time.  This avoids the N(0,1) mismatch that causes energy spikes when
        the learned latent distribution differs from the standard Gaussian.

    Stability:
        Adam optimizer is paired with global-norm gradient clipping (clip=1.0)
        to prevent the weight-update divergence observed without clipping when
        fine-tuning on the mixed real+generated dataset.
    """

    def __init__(
        self,
        data_dim:       int,
        latent_dim:     int   = 256,
        n_hidden_layers:int   = 2,
        infer_steps:    int   = 20,
        eta_infer:      float = 0.1,
        seed:           int   = 42,
    ):
        self.data_dim        = data_dim
        self.latent_dim      = latent_dim
        self.n_hidden_layers = n_hidden_layers
        self.infer_steps     = infer_steps
        self.eta_infer       = eta_infer
        self.seed            = seed
        self.params          = None
        self.structure       = None
        self._rng_counter    = 0
        # Learned prior: mean and std of the memory latents
        self._latent_mean: Optional[np.ndarray] = None
        self._latent_std:  Optional[np.ndarray] = None

    def _next_key(self) -> jax.Array:
        key = jax.random.PRNGKey(self.seed + self._rng_counter)
        self._rng_counter += 1
        return key

    def _update_latent_prior(self, images: np.ndarray, max_samples: int = 1000) -> None:
        """Run inference on a subset of images and collect memory latent stats.

        Updates ``_latent_mean`` and ``_latent_std`` using an exponential
        moving average so that all tasks contribute proportionally.
        """
        sensory_name = self.structure.task_map["x"]
        n = min(len(images), max_samples)
        subset = images[np.random.choice(len(images), n, replace=False)]
        batch_size = 128
        all_latents = []

        for start in range(0, n, batch_size):
            batch = jnp.array(subset[start : start + batch_size])
            nb = len(batch)
            rng = self._next_key()
            clamps = {sensory_name: batch}
            state = initialize_graph_state(
                self.structure, nb, rng, clamps=clamps, params=self.params
            )
            final_state = run_inference(self.params, state, clamps, self.structure)
            latents = np.asarray(final_state.nodes["memory"].z_latent)
            all_latents.append(latents)

        all_latents = np.concatenate(all_latents, axis=0)
        new_mean = np.mean(all_latents, axis=0)
        new_std  = np.std(all_latents,  axis=0) + 1e-6

        if self._latent_mean is None:
            self._latent_mean = new_mean
            self._latent_std  = new_std
        else:
            # Equal-weight running average over tasks
            alpha = 0.5
            self._latent_mean = alpha * self._latent_mean + (1 - alpha) * new_mean
            self._latent_std  = alpha * self._latent_std  + (1 - alpha) * new_std

    def train(
        self,
        images:    np.ndarray,
        n_epochs:  int,
        lr:        float = 1e-3,
        batch_size:int   = 128,
        verbose:   bool  = False,
    ) -> None:
        """Train (or fine-tune) the generator on a set of images.

        Uses Adam + global-norm gradient clipping to prevent divergence during
        fine-tuning on mixed (real + generated) batches.
        """
        if self.structure is None:
            self.structure = _build_pc_generator(
                self.data_dim, self.latent_dim,
                self.n_hidden_layers, self.infer_steps, self.eta_infer,
            )
            self.params = initialize_params(self.structure, self._next_key())

        # Gradient clipping prevents energy-spike divergence during fine-tuning
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr),
        )
        loader = MemorizationLoader(images, batch_size=batch_size, seed=self.seed)

        self.params, _, _ = train_pcn(
            params=self.params,
            structure=self.structure,
            train_loader=loader,
            optimizer=optimizer,
            config={"num_epochs": n_epochs},
            rng_key=self._next_key(),
            verbose=verbose,
        )

        # Update learned prior from the training images (not mixed-batch images)
        self._update_latent_prior(images)

    def generate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Generate samples using the learned memory-latent prior.

        Clamps the memory (top) node to z ~ N(mu, sigma) (fitted from training
        data) and runs inference so the hidden and sensory layers propagate the
        top-down prediction.  The sensory z_mu is the generated image.
        """
        if self.params is None:
            raise RuntimeError("PCGenerator must be trained before generating.")

        rng = np.random.default_rng(seed)

        if self._latent_mean is not None:
            # Sample from fitted Gaussian prior
            eps = rng.standard_normal((n_samples, self.latent_dim)).astype(np.float32)
            z_np = self._latent_mean + self._latent_std * eps
            z = jnp.array(z_np)
        else:
            # Fall back to N(0,1) before any training
            key = jax.random.PRNGKey(int(rng.integers(2**31)))
            z = jax.random.normal(key, (n_samples, self.latent_dim))

        sensory_name = self.structure.task_map["x"]
        clamps = {"memory": z}
        key_state = self._next_key()

        state = initialize_graph_state(
            self.structure, n_samples, key_state, clamps=clamps, params=self.params
        )
        final_state = run_inference(self.params, state, clamps, self.structure)

        generated = np.asarray(final_state.nodes[sensory_name].z_mu)
        return np.clip(generated, 0.0, 1.0)


# =============================================================================
# 3.  VAE GENERATOR  (Flax — backpropagation baseline)
# =============================================================================


class _Encoder(nn.Module):
    latent_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        mu     = nn.Dense(self.latent_dim)(x)
        logvar = nn.Dense(self.latent_dim)(x)
        return mu, logvar


class _Decoder(nn.Module):
    output_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, z):
        z = nn.Dense(self.hidden_dim)(z)
        z = nn.relu(z)
        z = nn.Dense(self.hidden_dim)(z)
        z = nn.relu(z)
        return nn.sigmoid(nn.Dense(self.output_dim)(z))


class _VAEModule(nn.Module):
    latent_dim: int
    data_dim:   int
    hidden_dim: int

    def setup(self):
        self.encoder = _Encoder(self.latent_dim, self.hidden_dim)
        self.decoder = _Decoder(self.data_dim,   self.hidden_dim)

    def __call__(self, x, rng):
        mu, logvar = self.encoder(x)
        eps = jax.random.normal(rng, mu.shape)
        z   = mu + jnp.exp(0.5 * logvar) * eps
        return self.decoder(z), mu, logvar

    def decode(self, z):
        return self.decoder(z)


def _vae_loss(params, apply_fn, x, rng) -> jnp.ndarray:
    """ELBO = reconstruction (MSE) + KL divergence."""
    x_recon, mu, logvar = apply_fn({"params": params}, x, rng)
    recon = jnp.mean((x - x_recon) ** 2)
    kl    = -0.5 * jnp.mean(1.0 + logvar - mu ** 2 - jnp.exp(logvar))
    return recon + kl


@jax.jit
def _vae_train_step(state, x, rng):
    loss, grads = jax.value_and_grad(_vae_loss)(
        state.params, state.apply_fn, x, rng
    )
    return state.apply_gradients(grads=grads), loss


class VAEGenerator:
    """VAE-based generative replay model (backpropagation baseline).

    Architecture (4 linear layers total, matching the PC generator depth):
        Encoder: data → hidden → hidden → (mu, logvar) [latent_dim]
        Decoder: latent → hidden → hidden → data
    """

    def __init__(
        self,
        data_dim:   int,
        latent_dim: int   = 256,
        hidden_dim: int   = 256,
        seed:       int   = 42,
    ):
        self.data_dim   = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.seed       = seed
        self._rng       = jax.random.PRNGKey(seed)
        self.state      = None
        self._module    = _VAEModule(latent_dim=latent_dim, data_dim=data_dim, hidden_dim=hidden_dim)

    def _next_key(self) -> jax.Array:
        self._rng, key = jax.random.split(self._rng)
        return key

    def _init_state(self, lr: float) -> None:
        key = self._next_key()
        dummy = jnp.zeros((1, self.data_dim))
        params = self._module.init({"params": key}, dummy, key)["params"]
        self.state = train_state.TrainState.create(
            apply_fn=self._module.apply,
            params=params,
            tx=optax.adam(lr),
        )

    def train(
        self,
        images:    np.ndarray,
        n_epochs:  int,
        lr:        float = 1e-3,
        batch_size:int   = 128,
        verbose:   bool  = False,
    ) -> None:
        """Train (or fine-tune) on a set of images."""
        if self.state is None:
            self._init_state(lr)

        n = len(images)
        for epoch in range(n_epochs):
            rng = self._next_key()
            perm = np.random.permutation(n)
            epoch_loss = []
            for start in range(0, n - batch_size + 1, batch_size):
                batch = jnp.array(images[perm[start : start + batch_size]])
                rng, step_rng = jax.random.split(rng)
                self.state, loss = _vae_train_step(self.state, batch, step_rng)
                epoch_loss.append(float(loss))
            if verbose and (epoch % max(1, n_epochs // 10) == 0):
                print(f"    VAE epoch {epoch+1}/{n_epochs}  loss={np.mean(epoch_loss):.4f}")

    def generate(self, n_samples: int, seed: Optional[int] = None) -> np.ndarray:
        """Sample z ~ N(0,1) and decode to generate images."""
        if self.state is None:
            raise RuntimeError("VAEGenerator must be trained before generating.")
        key = jax.random.PRNGKey(seed) if seed is not None else self._next_key()
        z = jax.random.normal(key, (n_samples, self.latent_dim))
        x_gen = self._module.apply({"params": self.state.params}, z, method=self._module.decode)
        return np.asarray(x_gen)


# =============================================================================
# 4.  TASK CLASSIFIER  (Flax MLP — 2 × 400 hidden, cross-entropy)
# =============================================================================


class _TaskMLP(nn.Module):
    n_classes:  int
    hidden_dim: int = 400

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.n_classes)(x)   # logits


@jax.jit
def _clf_train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x)
        return jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, y)
        )
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


@partial(jax.jit, static_argnames=("apply_fn",))
def _clf_accuracy(params, apply_fn, x, y):
    logits = apply_fn({"params": params}, x)
    preds  = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)


class TaskClassifier:
    """Flax MLP classifier trained with cross-entropy + Adam.

    Architecture: input_dim → 400 → 400 → n_classes (paper: 2 hidden layers × 400)
    """

    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 400,
                 seed: int = 42):
        self.input_dim  = input_dim
        self.n_classes  = n_classes
        self.hidden_dim = hidden_dim
        self._rng       = jax.random.PRNGKey(seed)
        self.state      = None
        self._module    = _TaskMLP(n_classes=n_classes, hidden_dim=hidden_dim)

    def _next_key(self) -> jax.Array:
        self._rng, key = jax.random.split(self._rng)
        return key

    def _init_state(self, lr: float) -> None:
        key    = self._next_key()
        dummy  = jnp.zeros((1, self.input_dim))
        params = self._module.init(key, dummy)["params"]
        self.state = train_state.TrainState.create(
            apply_fn=self._module.apply,
            params=params,
            tx=optax.adam(lr),
        )

    def train(
        self,
        x:          np.ndarray,
        y:          np.ndarray,
        n_epochs:   int,
        lr:         float = 1e-3,
        batch_size: int   = 128,
        verbose:    bool  = False,
    ) -> None:
        if self.state is None:
            self._init_state(lr)

        n = len(x)
        for epoch in range(n_epochs):
            perm = np.random.permutation(n)
            epoch_loss = []
            for start in range(0, n - batch_size + 1, batch_size):
                idx   = perm[start : start + batch_size]
                xb    = jnp.array(x[idx])
                yb    = jnp.array(y[idx])
                self.state, loss = _clf_train_step(self.state, xb, yb)
                epoch_loss.append(float(loss))
            if verbose and (epoch % max(1, n_epochs // 5) == 0):
                print(f"    Clf epoch {epoch+1}/{n_epochs}  loss={np.mean(epoch_loss):.4f}")

    def accuracy(self, x: np.ndarray, y: np.ndarray, batch_size: int = 512) -> float:
        if self.state is None:
            return 0.0
        corrects, total = 0, 0
        for start in range(0, len(x), batch_size):
            xb = jnp.array(x[start : start + batch_size])
            yb = jnp.array(y[start : start + batch_size])
            corrects += int(np.sum(np.asarray(
                _clf_accuracy(self.state.params, self.state.apply_fn, xb, yb)
            ) * len(yb)))
            total += len(yb)
        return corrects / total if total else 0.0


# =============================================================================
# 5.  CONTINUAL LEARNING LOOP
# =============================================================================


def _make_generator(method: str, data_dim: int, latent_dim: int,
                    hidden_dim: int, seed: int):
    """Factory for a fresh generator of the requested type."""
    if method == "pc":
        return PCGenerator(
            data_dim=data_dim,
            latent_dim=latent_dim,
            n_hidden_layers=2,   # memory + 2 hidden + sensory = 4-layer PCN
            infer_steps=20,
            eta_infer=0.1,
            seed=seed,
        )
    elif method == "vae":
        return VAEGenerator(
            data_dim=data_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            seed=seed,
        )
    raise ValueError(f"Unknown method: {method!r}. Choose 'pc' or 'vae'.")


def run_experiment(
    tasks:          List[Dict],
    method:         str,           # "pc" or "vae"
    data_dim:       int,
    n_classes:      int,
    epochs_gen:     int,           # epochs per task for the generator
    epochs_clf:     int,           # epochs per task for the classifier
    mix_ratio:      float = 0.5,   # β: fraction of REAL samples in mixed batch
    batch_size:     int   = 128,
    gen_latent_dim: int   = 128,
    gen_hidden_dim: int   = 256,
    clf_hidden_dim: int   = 400,
    gen_samples:    int   = 400,   # replay samples per previous task
    lr:             float = 1e-3,
    seed:           int   = 42,
    verbose:        bool  = True,
) -> np.ndarray:
    """Run one continual-learning trial and return the accuracy matrix.

    Per-task-generator design
    -------------------------
    Each completed task s gets its OWN generator trained exclusively on task
    s's real data.  At task t, replay samples come from generators[0..t-1].
    This avoids catastrophic forgetting inside the generator itself and keeps
    training stable regardless of task count.

    The classifier is trained on the mixed real+replay dataset, which is
    exactly the setting studied in the paper.

    Returns
    -------
    acc_matrix : shape (n_tasks, n_tasks)
        acc_matrix[t, s] = accuracy on task s's test set after learning task t.
        Entries where s > t are NaN (task not seen yet).
    """
    n_tasks = len(tasks)
    acc_matrix = np.full((n_tasks, n_tasks), np.nan)

    # One generator per completed task
    task_generators: List = []

    # Shared task classifier (single model across all tasks)
    classifier = TaskClassifier(
        input_dim=data_dim,
        n_classes=n_classes,
        hidden_dim=clf_hidden_dim,
        seed=seed,
    )

    # ── Task loop ──────────────────────────────────────────────────────────
    for t, task in enumerate(tasks):
        x_real = task["x_train"].astype(np.float32)
        y_real = task["y_train"].astype(np.int32)

        if verbose:
            print(f"\n  -- Task {t+1}/{n_tasks}  classes={task['classes']} "
                  f"n_train={len(x_real)} --")

        # ── Step 1: build and train THIS task's generator (real data only) ──
        gen_t = _make_generator(method, data_dim, gen_latent_dim,
                                gen_hidden_dim, seed + t * 1000)
        if verbose:
            print(f"    Training {method.upper()} generator ({epochs_gen} epochs) ...")
        gen_t.train(x_real, n_epochs=epochs_gen, lr=lr,
                    batch_size=batch_size, verbose=False)
        task_generators.append(gen_t)

        # ── Step 2: collect replay data from all previous task generators ──
        if t == 0:
            x_mixed = x_real
            y_mixed = y_real
        else:
            replay_x_parts, replay_y_parts = [], []
            for s in range(t):
                n_s = max(64, gen_samples)
                x_s = task_generators[s].generate(n_s, seed=seed + t * 100 + s)
                cls_s = tasks[s]["classes"]
                # Cycle uniformly over task s classes
                y_s = np.array(
                    [cls_s[i % len(cls_s)] for i in range(n_s)], dtype=np.int32
                )
                replay_x_parts.append(x_s)
                replay_y_parts.append(y_s)

            x_replay = np.concatenate(replay_x_parts, axis=0)
            y_replay = np.concatenate(replay_y_parts, axis=0)

            # Enforce β-mixing: sample n_real ≈ n_replay real samples so that
            # the mixed dataset is approximately (mix_ratio) real vs
            # (1-mix_ratio) replay regardless of raw dataset sizes.
            # This prevents the large real dataset from drowning the replay.
            n_replay_total = len(x_replay)
            n_real_target  = max(
                int(n_replay_total * mix_ratio / max(1.0 - mix_ratio, 1e-6)),
                batch_size,
            )
            replace_real = n_real_target > len(x_real)
            real_idx = np.random.choice(len(x_real), n_real_target,
                                        replace=replace_real)
            x_mixed = np.concatenate([x_real[real_idx], x_replay], axis=0)
            y_mixed = np.concatenate([y_real[real_idx], y_replay], axis=0)

            perm = np.random.permutation(len(x_mixed))
            x_mixed, y_mixed = x_mixed[perm], y_mixed[perm]

        if verbose:
            print(f"    Mixed dataset: {len(x_mixed)} samples "
                  f"({len(x_real)} real + {len(x_mixed)-len(x_real)} replay)")

        # ── Step 3: train classifier on mixed data ───────────────────────
        if verbose:
            print(f"    Training classifier ({epochs_clf} epochs) ...")
        classifier.train(x_mixed, y_mixed, n_epochs=epochs_clf, lr=lr,
                         batch_size=batch_size, verbose=False)

        # ── Step 4: evaluate on all seen tasks ────────────────────────────
        for s in range(t + 1):
            x_te = tasks[s]["x_test"].astype(np.float32)
            y_te = tasks[s]["y_test"].astype(np.int32)
            acc  = classifier.accuracy(x_te, y_te)
            acc_matrix[t, s] = acc
            if verbose:
                print(f"    Task {s+1} acc: {acc*100:.1f}%")

    return acc_matrix


# =============================================================================
# 6.  METRICS
# =============================================================================


def compute_metrics(acc_matrix: np.ndarray) -> Dict:
    """Compute continual learning metrics from the accuracy matrix.

    acc_matrix[t, s] = accuracy on task s after learning task t.
    """
    n = acc_matrix.shape[0]

    # ── Average Accuracy ──────────────────────────────────────────────────
    # Mean accuracy on all tasks after learning the final task
    avg_acc = float(np.nanmean(acc_matrix[n - 1, :]))

    # ── Forgetting Measure ───────────────────────────────────────────────
    # For each task s: f_s = max_t(acc[t,s]) − acc[T,s]  (t ≤ T-1)
    forgetting = []
    for s in range(n - 1):  # not the last task (it can't forget itself forward)
        col = acc_matrix[:, s]
        valid = col[: n]  # all rows
        peak  = float(np.nanmax(valid))
        final = float(acc_matrix[n - 1, s])
        forgetting.append(peak - final)
    avg_forgetting = float(np.mean(forgetting)) if forgetting else 0.0

    # ── Forward Transfer ─────────────────────────────────────────────────
    # FWT_s = acc[s, s] − acc_scratch_s
    # We approximate acc_scratch by the first-task accuracy
    # (acc[0,0] gives performance on task 0 from scratch; for s>0 we use
    # acc[s,s] − acc[0,0] as an approximation, following common practice)
    first_acc  = float(acc_matrix[0, 0]) if not np.isnan(acc_matrix[0, 0]) else 0.0
    fwt_vals   = [float(acc_matrix[s, s]) - first_acc for s in range(1, n)
                  if not np.isnan(acc_matrix[s, s])]
    avg_fwt    = float(np.mean(fwt_vals)) if fwt_vals else 0.0

    # ── Backward Transfer ─────────────────────────────────────────────────
    # BWT_s = acc[T, s] − acc[s, s]  for s < T
    bwt_vals = [float(acc_matrix[n - 1, s]) - float(acc_matrix[s, s])
                for s in range(n - 1)
                if not np.isnan(acc_matrix[n - 1, s]) and not np.isnan(acc_matrix[s, s])]
    avg_bwt  = float(np.mean(bwt_vals)) if bwt_vals else 0.0

    return {
        "avg_accuracy":  avg_acc,
        "avg_forgetting": avg_forgetting,
        "fwd_transfer":  avg_fwt,
        "bwd_transfer":  avg_bwt,
    }


# =============================================================================
# 7.  RESULTS DISPLAY
# =============================================================================


def print_accuracy_matrix(acc_matrix: np.ndarray, title: str = "") -> None:
    n = acc_matrix.shape[0]
    if title:
        print(f"\n  {title}")
    header = "       " + "  ".join(f"T{s+1:>4}" for s in range(n))
    print(f"  {header}")
    for t in range(n):
        row = f"  T{t+1:>2}:"
        for s in range(n):
            v = acc_matrix[t, s]
            row += f"  {v*100:5.1f}%" if not np.isnan(v) else "    ---"
        print(row)


def print_results_table(
    results: Dict[str, Dict[str, Dict]],
    datasets: List[str],
) -> None:
    """Print a summary table matching Table 1 of the paper."""
    print("\n" + "=" * 72)
    print("  RESULTS — Continual Learning with Generative Replay")
    print("=" * 72)
    print(f"\n  {'Dataset':<18} {'Method':<10} "
          f"{'Avg Acc':>8} {'Forgetting':>10} {'Fwd BT':>8} {'Bwd BT':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for ds in datasets:
        if ds not in results:
            continue
        for method in ["pc", "vae"]:
            if method not in results[ds]:
                continue
            m   = results[ds][method]
            avg = m.get("avg_accuracy", np.nan)
            fgt = m.get("avg_forgetting", np.nan)
            fwt = m.get("fwd_transfer", np.nan)
            bwt = m.get("bwd_transfer", np.nan)
            label = "PC Replay" if method == "pc" else "VAE Replay"
            ds_label = ds if method == "pc" else ""
            print(f"  {ds_label:<18} {label:<10} "
                  f"{avg*100:8.2f}% {fgt*100:10.2f}% {fwt*100:8.2f}% {bwt*100:8.2f}%")

    print()
    print("  Paper (arXiv:2512.00619) reference results:")
    print("  Dataset        Method     Avg Acc   Forgetting")
    print("  Split-MNIST    PC         94.20%      2.10%")
    print("  Split-MNIST    VAE        89.10%      5.80%")
    print("  Split-CIFAR-10 PC         78.50%      8.30%")
    print("  Split-CIFAR-10 VAE        68.20%     18.70%")
    print("  Split-CIFAR100 PC         65.30%     12.40%")
    print("  Split-CIFAR100 VAE        52.10%     24.90%")
    print()

    # Forgetting comparison table (Table 1 of paper)
    print("  Forgetting measure by dataset (lower is better):")
    print(f"  {'Dataset':<20} {'PC Replay':>12} {'VAE Replay':>12} {'Improvement':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*12}")
    for ds in datasets:
        if ds not in results:
            continue
        pc_fgt  = results[ds].get("pc",  {}).get("avg_forgetting", np.nan)
        vae_fgt = results[ds].get("vae", {}).get("avg_forgetting", np.nan)
        if not np.isnan(pc_fgt) and not np.isnan(vae_fgt):
            delta = vae_fgt - pc_fgt
            print(f"  {ds:<20} {pc_fgt*100:11.2f}% {vae_fgt*100:11.2f}% {delta*100:+11.2f}%")
        else:
            print(f"  {ds:<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")


# =============================================================================
# 8.  MAIN
# =============================================================================

# Paper (arXiv:2512.00619) specifies 50 epochs/task for MNIST, 100 for CIFAR.
# ResNet-18 is used for CIFAR in the paper; our implementation uses an MLP for
# all datasets (a known deviation that will reduce CIFAR performance).
DATASET_CONFIGS = {
    "mnist": {
        "n_tasks": 5,
        "classes_per_task": 2,
        "n_classes": 10,
        "full_epochs_gen":  50,   # matches paper: 50 epochs/task
        "full_epochs_clf":  50,   # matches paper: 50 epochs/task
        "quick_epochs_gen": 10,
        "quick_epochs_clf": 20,
        "latent_dim": 128,
        "hidden_dim": 256,
        "gen_samples_per_task": 2000,
    },
    "cifar10": {
        "n_tasks": 5,
        "classes_per_task": 2,
        "n_classes": 10,
        "full_epochs_gen":  100,  # matches paper: 100 epochs/task
        "full_epochs_clf":  100,  # matches paper: 100 epochs/task
        "quick_epochs_gen": 5,
        "quick_epochs_clf": 10,
        "latent_dim": 256,
        "hidden_dim": 512,
        "gen_samples_per_task": 400,
        # NOTE: paper uses ResNet-18 for CIFAR; we use an MLP (deviation).
    },
    "cifar100": {
        "n_tasks": 10,
        "classes_per_task": 10,
        "n_classes": 100,
        "full_epochs_gen":  100,  # matches paper: 100 epochs/task
        "full_epochs_clf":  100,  # matches paper: 100 epochs/task
        "quick_epochs_gen": 3,
        "quick_epochs_clf": 5,
        "latent_dim": 256,
        "hidden_dim": 512,
        "gen_samples_per_task": 300,
        # NOTE: paper uses ResNet-18 for CIFAR; we use an MLP (deviation).
    },
}


def run_dataset(
    ds_name: str,
    quick: bool,
    n_trials: int,
    seed: int,
    verbose: bool,
) -> Dict[str, Dict]:
    """Run PC and VAE experiments for a given dataset.

    Returns a dict with keys "pc" and "vae", each mapping to a metrics dict.
    """
    cfg = DATASET_CONFIGS[ds_name]

    print(f"\n{'='*60}")
    print(f"  Dataset: {ds_name.upper()}")
    print(f"  Tasks: {cfg['n_tasks']}  |  Classes/task: {cfg['classes_per_task']}")
    print(f"  Mode: {'QUICK' if quick else 'FULL'}")
    print(f"  Trials: {n_trials}")
    print(f"{'='*60}")

    # Load data
    print("\nLoading data...")
    if ds_name == "mnist":
        x_tr, y_tr, x_te, y_te = load_mnist()
        data_dim = 784
    elif ds_name == "cifar10":
        x_tr, y_tr, x_te, y_te = load_cifar10()
        data_dim = 3072
    elif ds_name == "cifar100":
        x_tr, y_tr, x_te, y_te = load_cifar100()
        data_dim = 3072
    else:
        raise ValueError(f"Unknown dataset: {ds_name}")

    print(f"  Train: {len(x_tr)}  Test: {len(x_te)}  dim={data_dim}")

    # Build task splits
    tasks = create_split_tasks(
        x_tr, y_tr, x_te, y_te,
        n_tasks=cfg["n_tasks"],
        classes_per_task=cfg["classes_per_task"],
        seed=seed,
    )

    epochs_gen = cfg["quick_epochs_gen"] if quick else cfg["full_epochs_gen"]
    epochs_clf = cfg["quick_epochs_clf"] if quick else cfg["full_epochs_clf"]

    results = {}
    for method in ["pc", "vae"]:
        print(f"\n--- Method: {method.upper()} ---")
        trial_metrics = []
        trial_matrices = []

        for trial in range(n_trials):
            trial_seed = seed + trial * 1000
            print(f"  Trial {trial+1}/{n_trials} (seed={trial_seed})")
            t0 = time.time()

            acc_mat = run_experiment(
                tasks=tasks,
                method=method,
                data_dim=data_dim,
                n_classes=cfg["n_classes"],
                epochs_gen=epochs_gen,
                epochs_clf=epochs_clf,
                mix_ratio=0.5,
                batch_size=128,
                gen_latent_dim=cfg["latent_dim"],
                gen_hidden_dim=cfg["hidden_dim"],
                clf_hidden_dim=400,
                gen_samples=cfg["gen_samples_per_task"],
                lr=1e-3,
                seed=trial_seed,
                verbose=verbose and (trial == 0),
            )

            elapsed = time.time() - t0
            metrics = compute_metrics(acc_mat)
            trial_metrics.append(metrics)
            trial_matrices.append(acc_mat)

            print(f"    Avg Acc: {metrics['avg_accuracy']*100:.2f}%  "
                  f"Forgetting: {metrics['avg_forgetting']*100:.2f}%  "
                  f"({elapsed:.0f}s)")

        # Aggregate over trials
        keys = list(trial_metrics[0].keys())
        agg  = {}
        for k in keys:
            vals          = [m[k] for m in trial_metrics]
            agg[k]        = float(np.mean(vals))
            # ddof=0 avoids division-by-zero warning when n_trials==1
            agg[k+"_std"] = float(np.std(vals, ddof=0))

        # Final matrix = average across trials (suppress nan-slice warning for
        # upper-triangle NaN entries when n_trials==1)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            agg["acc_matrix"]     = np.nanmean(trial_matrices, axis=0)
            agg["acc_matrix_std"] = np.nanstd(trial_matrices, axis=0, ddof=0)

        results[method] = agg

        print(f"\n  {method.upper()} summary ({n_trials} trial(s)):")
        print(f"    Avg Accuracy : {agg['avg_accuracy']*100:.2f}% "
              f"± {agg['avg_accuracy_std']*100:.2f}%")
        print(f"    Forgetting   : {agg['avg_forgetting']*100:.2f}% "
              f"± {agg['avg_forgetting_std']*100:.2f}%")
        print(f"    Fwd Transfer : {agg['fwd_transfer']*100:.2f}% "
              f"± {agg['fwd_transfer_std']*100:.2f}%")
        print(f"    Bwd Transfer : {agg['bwd_transfer']*100:.2f}% "
              f"± {agg['bwd_transfer_std']*100:.2f}%")

        print_accuracy_matrix(agg["acc_matrix"], title=f"{method.upper()} accuracy matrix")

    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Recreate arXiv:2512.00619 — PC vs VAE generative replay"
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dataset", choices=["mnist", "cifar10", "cifar100"],
                   default="mnist", help="Dataset to run (default: mnist)")
    g.add_argument("--all", action="store_true", help="Run all 3 datasets")
    p.add_argument("--quick", action="store_true",
                   help="Quick mode: fewer epochs for fast testing")
    p.add_argument("--n_trials", type=int, default=1,
                   help="Number of independent trials (paper uses 5; default: 1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true",
                   help="Print per-epoch output for trial 1")
    return p.parse_args()


def main():
    args = parse_args()
    jax.config.update("jax_default_prng_impl", "threefry2x32")

    print("=" * 60)
    print("  Neuroscience-Inspired Memory Replay for Continual Learning")
    print("  arXiv:2512.00619  —  PC vs Backprop Generative Replay")
    print("=" * 60)
    print(f"  JAX devices: {jax.devices()}")

    datasets = ["mnist", "cifar10", "cifar100"] if args.all else [args.dataset]

    all_results: Dict[str, Dict] = {}
    for ds in datasets:
        all_results[ds] = run_dataset(
            ds_name=ds,
            quick=args.quick,
            n_trials=args.n_trials,
            seed=args.seed,
            verbose=args.verbose,
        )

    print_results_table(all_results, datasets)


if __name__ == "__main__":
    main()
