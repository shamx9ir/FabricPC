"""
Associative Memories via Predictive Coding
============================================

Reproduces experiments from:
    Salvatori et al. (2021) "Associative Memories via Predictive Coding"
    arXiv:2109.08063

Uses FabricPC (JAX predictive coding framework) to demonstrate that
generative predictive coding networks store training data as attractors,
enabling retrieval from corrupted or partial observations.

Architecture (generative PCN, top-down)::

    memory(n) --> hidden_{L-1}(n) --> ... --> hidden_0(n) --> sensory(d)
    (IdentityNode)   (Linear+ReLU)           (Linear+ReLU)   (Linear, clamped)

During training the sensory layer is clamped to the target image.  The
memory node (in_degree=0, unclamped) and all hidden layers evolve freely
under the inference dynamics, then weights are updated to minimise the
prediction-error energy.  After training, each stored image becomes an
attractor of the network dynamics.

Note on activation placement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The paper applies activations *pre-synaptically*:
``mu^l = theta^{l+1} f(x^{l+1})``.  This script uses a custom
``PreSynapticLinear`` node that implements exactly this:
``z_mu = W @ f(x) + b``, matching the paper's formulation.

Experiments
-----------
1. **Denoising** (Section 3): Retrieve stored CIFAR-10 images from
   Gaussian-corrupted versions, sweeping hidden dimensions and dataset sizes.
2. **Partial Retrieval** (Section 4): Reconstruct images given only a
   fraction of known pixels, using custom inference with pixel masking.
3. **Deep PCNs** (Section 5.1): Increase network depth to improve
   retrieval capacity on larger datasets.

Usage::

    python examples/associative_memory_pcn.py                      # quick demo
    python examples/associative_memory_pcn.py --experiment denoising
    python examples/associative_memory_pcn.py --experiment partial
    python examples/associative_memory_pcn.py --experiment deep
    python examples/associative_memory_pcn.py --experiment all
"""

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import IdentityNode, Linear
from fabricpc.nodes.base import NodeBase, SlotSpec, FlattenInputMixin
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.activations import ReLUActivation, IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import KaimingInitializer, NormalInitializer, initialize
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.training import train_pcn
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")


# ============================================================================
# Pre-Synaptic Linear Node (paper-faithful activation placement)
# ============================================================================


class PreSynapticLinear(FlattenInputMixin, NodeBase):
    """Linear node with pre-synaptic activation: z_mu = W @ f(x) + b

    The paper (Salvatori et al., 2021) defines the prediction as:
        mu^l = theta^{l+1} @ f(x^{l+1})
    where f is applied to the *source* (upstream) node's latent BEFORE the
    weight multiplication.  FabricPC's standard ``Linear`` node applies the
    activation *post-synaptically*: ``z_mu = f(W @ x + b)``.

    This node stores an ``input_activation`` (e.g. ReLU) in its config and
    applies it to all incoming latent vectors before the matrix multiply.
    The node's own ``activation`` is ``IdentityActivation`` (no post-synaptic
    transform), so the full forward is::

        z_mu = W @ input_activation(x) + b
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
            activation=IdentityActivation(),  # no post-synaptic activation
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            flatten_input=flatten_input,
            input_activation=input_activation,  # stored in node_config
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config=None):
        """Same weight layout as Linear."""
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

        # Apply pre-synaptic activation to all inputs
        activated = {}
        for edge_key, x in inputs.items():
            activated[edge_key] = type(input_act).forward(x, input_act.config)

        # Linear transformation on activated inputs
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

        # No post-synaptic activation (node.activation is Identity)
        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)
        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        return state


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Associative Memory via Predictive Coding")
    p.add_argument(
        "--experiment",
        choices=["demo", "denoising", "partial", "deep", "all"],
        default="demo",
        help="Which experiment to run (default: demo — a quick single-config test)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--save-images",
        action="store_true",
        help="Save reconstruction images (requires matplotlib)",
    )
    return p.parse_args()


# ============================================================================
# Data Loading
# ============================================================================


def load_cifar10_subset(num_images, seed=42):
    """Load a random subset of CIFAR-10 training images.

    Returns
    -------
    images : ndarray, shape (num_images, 3072), dtype float32, values in [0, 1]
    """
    import tensorflow_datasets as tfds
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    ds = tfds.load("cifar10", split="train", as_supervised=True)

    all_images = []
    for img, _label in ds:
        all_images.append(img.numpy().astype(np.float32) / 255.0)
        if len(all_images) >= max(num_images * 3, 1000):
            break

    all_images = np.stack(all_images, axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_images), num_images, replace=False)
    images = all_images[indices]
    # Flatten (N, 32, 32, 3) → (N, 3072)
    return images.reshape(num_images, -1)


# ============================================================================
# Memorization Data Loader
# ============================================================================


class MemorizationLoader:
    """Yields the memorization dataset in batches.

    For small datasets the full set fits in one batch; for larger datasets
    it is split, with shuffling per epoch.
    """

    def __init__(self, images, batch_size=None, seed=42):
        self.images = np.asarray(images, dtype=np.float32)
        self.batch_size = batch_size or len(images)
        self.seed = seed
        self._epoch = 0
        self.num_samples = len(images)
        self._num_batches = max(1, self.num_samples // self.batch_size)

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self._epoch if self.seed is not None else None
        )
        self._epoch += 1
        indices = rng.permutation(self.num_samples)
        for start in range(
            0, self.num_samples - self.batch_size + 1, self.batch_size
        ):
            batch_idx = indices[start : start + self.batch_size]
            yield {"x": self.images[batch_idx]}

    def __len__(self):
        return self._num_batches


# ============================================================================
# Graph Construction
# ============================================================================


def build_generative_pcn(
    data_dim,
    hidden_dim,
    num_hidden_layers=1,
    infer_steps=32,
    eta_infer=0.1,
):
    """Build a generative PCN for associative memory.

    Parameters
    ----------
    data_dim : int
        Dimensionality of the data (e.g. 3072 for CIFAR-10).
    hidden_dim : int
        Width of every hidden layer.
    num_hidden_layers : int
        Number of hidden Linear layers (paper's "L-1" internal layers).
        ``num_hidden_layers=1`` gives a 2-layer PCN (1 weight matrix from
        memory to hidden, 1 from hidden to sensory).
    infer_steps : int
        Inference iterations per training step (paper's *T*).
    eta_infer : float
        Inference learning rate (paper's γ).

    Returns
    -------
    GraphStructure
    """
    # Memory node — top of the hierarchy, no incoming edges.
    # Its z_latent is updated during inference via top-down error gradients.
    memory = IdentityNode(shape=(hidden_dim,), name="memory")

    # Hidden layers — pre-synaptic ReLU: z_mu = W @ ReLU(x_parent) + b
    hidden_layers = []
    for i in range(num_hidden_layers):
        hidden_layers.append(
            PreSynapticLinear(
                shape=(hidden_dim,),
                name=f"hidden_{i}",
                input_activation=ReLUActivation(),
                weight_init=KaimingInitializer(),
                flatten_input=True,
                use_bias=True,
            )
        )

    # Sensory layer — clamped to data during training.
    # Pre-synaptic ReLU: μ⁰ = θ¹ · f(x¹) + b, matching the paper exactly.
    sensory = PreSynapticLinear(
        shape=(data_dim,),
        name="sensory",
        input_activation=ReLUActivation(),
        weight_init=KaimingInitializer(),
        flatten_input=True,
        use_bias=True,
    )

    # Top-down edges: memory → hidden_{L-1} → … → hidden_0 → sensory
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


# ============================================================================
# Training
# ============================================================================


def train_generative_pcn(
    images,
    hidden_dim,
    num_hidden_layers=1,
    num_epochs=2000,
    lr=0.001,
    infer_steps=24,
    eta_infer=0.1,
    batch_size=None,
    seed=42,
    verbose=True,
):
    """Train a generative PCN to memorize a set of images.

    Parameters
    ----------
    images : ndarray, shape (N, D)
        Training images normalised to [0, 1].
    hidden_dim : int
    num_hidden_layers : int
    num_epochs : int
    lr : float
        Weight learning rate (paper's α).
    infer_steps : int
        Inference steps per training iteration (paper's T).
    eta_infer : float
        Inference learning rate (paper's γ).
    batch_size : int or None
        Defaults to the full dataset size.
    seed : int
    verbose : bool

    Returns
    -------
    trained_params : GraphParams
    structure : GraphStructure
    energy_history : list
    """
    data_dim = images.shape[1]
    structure = build_generative_pcn(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        infer_steps=infer_steps,
        eta_infer=eta_infer,
    )

    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)
    params = initialize_params(structure, init_key)

    n_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    if verbose:
        print(
            f"  Architecture: memory({hidden_dim}) -> "
            + " -> ".join(
                f"hidden_{i}({hidden_dim})" for i in range(num_hidden_layers)
            )
            + f" -> sensory({data_dim})"
        )
        print(f"  Parameters: {n_params:,}")

    optimizer = optax.adam(lr)
    loader = MemorizationLoader(images, batch_size=batch_size, seed=seed)
    rng_key, train_key = jax.random.split(rng_key)

    trained_params, energy_history, _ = train_pcn(
        params=params,
        structure=structure,
        train_loader=loader,
        optimizer=optimizer,
        config={"num_epochs": num_epochs},
        rng_key=train_key,
        verbose=verbose,
    )
    return trained_params, structure, energy_history


# ============================================================================
# Retrieval — Denoising (Section 3, Figure 2)
# ============================================================================


def retrieve_from_corruption(
    params,
    structure,
    corrupted_images,
    num_iterations=30,
    retrieval_infer_steps=200,
    retrieval_eta_infer=0.01,
    seed=42,
):
    """Iterative retrieval function *F* from corrupted images.

    Algorithm (paper Fig. 2):
        1. Clamp sensory to current estimate.
        2. Run inference for *T* steps.
        3. Read μ⁰_T (prediction of sensory layer) as the new estimate.
        4. Repeat *num_iterations* times.

    Parameters
    ----------
    retrieval_eta_infer : float
        Inference learning rate for retrieval.  Defaults to 0.01 (lower
        than training) to prevent the dynamics from diverging when the
        sensory layer is clamped to a noisy image far from an attractor.

    Returns
    -------
    retrieved : ndarray, shape (N, D)
    """
    sensory_name = structure.task_map["x"]
    inference_obj = structure.config["inference"]
    inference_cls = type(inference_obj)
    base_config = dict(inference_obj.config)

    # Override inference parameters for retrieval — use a lower eta to
    # keep the dynamics stable when starting far from an attractor.
    base_config["infer_steps"] = retrieval_infer_steps
    base_config["eta_infer"] = retrieval_eta_infer

    batch_size = corrupted_images.shape[0]
    current = jnp.array(corrupted_images)
    rng_key = jax.random.PRNGKey(seed)

    for m in range(num_iterations):
        rng_key, step_key = jax.random.split(rng_key)
        clamps = {sensory_name: current}

        state = initialize_graph_state(
            structure, batch_size, step_key, clamps=clamps, params=params
        )

        # Run inference with custom step count
        def _body(t, s):
            return inference_cls.inference_step(
                params, s, clamps, structure, base_config
            )

        final_state = jax.lax.fori_loop(
            0, retrieval_infer_steps, _body, state
        )

        # μ⁰_T — the prediction of the sensory layer
        current = final_state.nodes[sensory_name].z_mu
        # Clip to prevent runaway values between F iterations
        current = jnp.clip(current, -1.0, 2.0)

    return np.asarray(current)


# ============================================================================
# Retrieval — Partial Images (Section 4, Algorithm 2)
# ============================================================================


def retrieve_from_partial(
    params,
    structure,
    images,
    mask,
    infer_steps=500,
    retrieval_eta_infer=0.01,
    seed=42,
):
    """Retrieve complete images from partial observations.

    Known pixels (where *mask* is True) are pinned after every inference
    step; unknown pixels evolve freely under the energy dynamics.

    Parameters
    ----------
    images : ndarray, shape (N, D)
        Full images (only the masked pixels are used).
    mask : ndarray, shape (N, D), dtype bool
        True where the pixel is known.
    infer_steps : int
        Number of inference iterations.

    Returns
    -------
    reconstructed : ndarray, shape (N, D)
    """
    sensory_name = structure.task_map["x"]
    inference_obj = structure.config["inference"]
    inference_cls = type(inference_obj)
    config = dict(inference_obj.config)
    config["infer_steps"] = infer_steps
    config["eta_infer"] = retrieval_eta_infer

    batch_size = images.shape[0]
    rng_key = jax.random.PRNGKey(seed)

    # Sensory is NOT clamped — it is free to update during inference,
    # but we re-pin the known pixels after every step.
    clamps = {}
    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )

    # Set initial sensory z_latent: known pixels from image, rest from init
    images_jnp = jnp.array(images)
    mask_jnp = jnp.array(mask)
    z_init = jnp.where(mask_jnp, images_jnp, state.nodes[sensory_name].z_latent)
    state = update_node_in_state(state, sensory_name, z_latent=z_init)

    def _body(t, s):
        s = inference_cls.inference_step(params, s, clamps, structure, config)
        # Re-pin known pixels
        z = s.nodes[sensory_name].z_latent
        z = jnp.where(mask_jnp, images_jnp, z)
        s = update_node_in_state(s, sensory_name, z_latent=z)
        return s

    final_state = jax.lax.fori_loop(0, infer_steps, _body, state)
    return np.asarray(final_state.nodes[sensory_name].z_latent)


# ============================================================================
# Noise / Mask Utilities
# ============================================================================


def add_gaussian_noise(images, variance=0.2, seed=42):
    """Corrupt images with additive Gaussian noise.

    Parameters
    ----------
    variance : float
        Variance of the noise (paper default: 0.2 → std ≈ 0.447).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, np.sqrt(variance), images.shape).astype(np.float32)
    return images + noise


def generate_pixel_mask(num_images, data_dim, fraction, seed=42):
    """Random pixel mask — each pixel is kept independently with prob *fraction*."""
    rng = np.random.default_rng(seed)
    return rng.random((num_images, data_dim)) < fraction


# ============================================================================
# Evaluation
# ============================================================================


def retrieval_metrics(original, retrieved, threshold=0.005):
    """Compute per-image MSE and fraction correctly retrieved.

    An image is "correctly retrieved" when its mean squared pixel error
    is below *threshold* (0.005 for denoising, 0.001 for partial retrieval).
    """
    per_img = np.mean((original - retrieved) ** 2, axis=1)
    return {
        "accuracy": float(np.mean(per_img < threshold)),
        "mean_mse": float(np.mean(per_img)),
        "max_mse": float(np.max(per_img)),
        "per_image_mse": per_img,
    }


# ============================================================================
# Visualisation (optional)
# ============================================================================


def save_reconstruction_grid(
    original, corrupted_or_partial, retrieved, path, ncols=10
):
    """Save a 3-row grid: original / corrupted / retrieved."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [matplotlib not available — skipping image save to {path}]")
        return

    n = min(len(original), ncols)
    fig, axes = plt.subplots(3, n, figsize=(2 * n, 6))
    if n == 1:
        axes = axes[:, None]
    titles = ["Original", "Input", "Retrieved"]
    for row_idx, imgs in enumerate([original, corrupted_or_partial, retrieved]):
        for col in range(n):
            img = np.nan_to_num(imgs[col], nan=0.5).reshape(32, 32, 3).clip(0, 1)
            axes[row_idx, col].imshow(img)
            axes[row_idx, col].axis("off")
            if col == 0:
                axes[row_idx, col].set_ylabel(titles[row_idx], fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved reconstruction grid → {path}")


# ============================================================================
# Experiment 1 — Denoising (Section 3)
# ============================================================================


def experiment_denoising(
    num_images_list=(100,),
    hidden_dims=(512,),
    noise_variance=0.2,
    num_epochs=2000,
    lr=0.001,
    infer_steps_train=24,
    eta_infer=0.1,
    retrieval_iters=20,
    retrieval_infer_steps=200,
    threshold=0.005,
    seed=42,
    save_images=False,
):
    """Section 3: Retrieve stored images from Gaussian-corrupted versions."""
    print("\n" + "=" * 70)
    print("  Experiment 1 — Denoising (Section 3)")
    print("=" * 70)

    for N in num_images_list:
        images = load_cifar10_subset(N, seed=seed)
        print(f"\nLoaded {N} CIFAR-10 images  (dim={images.shape[1]})")

        for n in hidden_dims:
            print(f"\n--- hidden_dim={n}, N={N} ---")
            t0 = time.time()
            params, structure, _ = train_generative_pcn(
                images,
                hidden_dim=n,
                num_hidden_layers=1,
                num_epochs=num_epochs,
                lr=lr,
                infer_steps=infer_steps_train,
                eta_infer=eta_infer,
                seed=seed,
            )
            print(f"  Training time: {time.time() - t0:.1f}s")

            # Corrupt with Gaussian noise
            corrupted = add_gaussian_noise(images, variance=noise_variance, seed=seed)

            t0 = time.time()
            retrieved = retrieve_from_corruption(
                params,
                structure,
                corrupted,
                num_iterations=retrieval_iters,
                retrieval_infer_steps=retrieval_infer_steps,
                seed=seed,
            )
            print(f"  Retrieval time: {time.time() - t0:.1f}s")

            m = retrieval_metrics(images, retrieved, threshold=threshold)
            print(
                f"  Retrieved: {m['accuracy']*100:.1f}%  "
                f"mean_MSE={m['mean_mse']:.6f}  max_MSE={m['max_mse']:.6f}"
            )

            if save_images:
                save_reconstruction_grid(
                    images,
                    corrupted,
                    retrieved,
                    f"denoising_N{N}_n{n}.png",
                )


# ============================================================================
# Experiment 2 — Partial Retrieval (Section 4)
# ============================================================================


def experiment_partial(
    num_images=50,
    hidden_dim=1024,
    fractions=(0.5, 0.25, 0.125),
    num_epochs=2000,
    lr=0.001,
    infer_steps_train=24,
    eta_infer=0.1,
    retrieval_infer_steps=500,
    threshold=0.001,
    seed=42,
    save_images=False,
):
    """Section 4: Reconstruct from a fraction of known pixels."""
    print("\n" + "=" * 70)
    print("  Experiment 2 — Partial Image Retrieval (Section 4)")
    print("=" * 70)

    images = load_cifar10_subset(num_images, seed=seed)
    data_dim = images.shape[1]
    print(f"Loaded {num_images} CIFAR-10 images  (dim={data_dim})")

    t0 = time.time()
    params, structure, _ = train_generative_pcn(
        images,
        hidden_dim=hidden_dim,
        num_hidden_layers=1,
        num_epochs=num_epochs,
        lr=lr,
        infer_steps=infer_steps_train,
        eta_infer=eta_infer,
        seed=seed,
    )
    print(f"Training time: {time.time() - t0:.1f}s")

    for frac in fractions:
        print(f"\n--- fraction={frac} ---")
        mask = generate_pixel_mask(num_images, data_dim, frac, seed=seed)
        print(f"  Known pixels per image: {mask.sum(axis=1).mean():.0f} / {data_dim}")

        t0 = time.time()
        retrieved = retrieve_from_partial(
            params,
            structure,
            images,
            mask,
            infer_steps=retrieval_infer_steps,
            seed=seed,
        )
        print(f"  Retrieval time: {time.time() - t0:.1f}s")

        m = retrieval_metrics(images, retrieved, threshold=threshold)
        print(
            f"  Retrieved: {m['accuracy']*100:.1f}%  "
            f"mean_MSE={m['mean_mse']:.6f}  max_MSE={m['max_mse']:.6f}"
        )

        if save_images:
            partial_vis = np.where(mask, images, 0.5)  # grey for missing
            save_reconstruction_grid(
                images,
                partial_vis,
                retrieved,
                f"partial_frac{frac}_n{hidden_dim}.png",
            )


# ============================================================================
# Experiment 3 — Deep PCNs (Section 5.1)
# ============================================================================


def experiment_deep(
    num_images=200,
    hidden_dim=1024,
    depths=(1, 3, 5, 7),
    fractions=(0.5, 0.25),
    num_epochs=2000,
    lr=0.0005,
    infer_steps_train=24,
    eta_infer=0.1,
    retrieval_infer_steps=500,
    threshold=0.001,
    seed=42,
    save_images=False,
):
    """Section 5.1: Deeper networks improve retrieval capacity."""
    print("\n" + "=" * 70)
    print("  Experiment 3 — Deep Generative PCNs (Section 5.1)")
    print("=" * 70)

    images = load_cifar10_subset(num_images, seed=seed)
    data_dim = images.shape[1]
    print(f"Loaded {num_images} CIFAR-10 images  (dim={data_dim})")

    for depth in depths:
        print(f"\n{'='*50}")
        print(f"  Depth = {depth} hidden layers")
        print(f"{'='*50}")

        t0 = time.time()
        params, structure, _ = train_generative_pcn(
            images,
            hidden_dim=hidden_dim,
            num_hidden_layers=depth,
            num_epochs=num_epochs,
            lr=lr,
            infer_steps=infer_steps_train,
            eta_infer=eta_infer,
            seed=seed,
        )
        print(f"  Training time: {time.time() - t0:.1f}s")

        for frac in fractions:
            mask = generate_pixel_mask(num_images, data_dim, frac, seed=seed)

            t0 = time.time()
            retrieved = retrieve_from_partial(
                params,
                structure,
                images,
                mask,
                infer_steps=retrieval_infer_steps,
                seed=seed,
            )
            print(f"  frac={frac}  retrieval_time={time.time() - t0:.1f}s")

            m = retrieval_metrics(images, retrieved, threshold=threshold)
            print(
                f"    Retrieved: {m['accuracy']*100:.1f}%  "
                f"mean_MSE={m['mean_mse']:.6f}"
            )

            if save_images:
                partial_vis = np.where(mask, images, 0.5)
                save_reconstruction_grid(
                    images,
                    partial_vis,
                    retrieved,
                    f"deep_d{depth}_frac{frac}.png",
                )


# ============================================================================
# Quick Demo
# ============================================================================


def experiment_demo(seed=42, save_images=False):
    """Quick demo: memorise 50 images, retrieve from noise + partial."""
    print("\n" + "=" * 70)
    print("  Quick Demo — Associative Memory via Predictive Coding")
    print("=" * 70)

    N = 50
    hidden_dim = 512
    images = load_cifar10_subset(N, seed=seed)
    data_dim = images.shape[1]
    print(f"Loaded {N} CIFAR-10 images  (dim={data_dim})")

    # --- Train ---
    print("\n--- Training generative PCN ---")
    t0 = time.time()
    params, structure, energy_history = train_generative_pcn(
        images,
        hidden_dim=hidden_dim,
        num_hidden_layers=1,
        num_epochs=1000,
        lr=0.001,
        infer_steps=24,
        eta_infer=0.1,
        seed=seed,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s")

    # Final energy
    final_energies = energy_history[-1] if energy_history else []
    if final_energies:
        print(f"Final epoch avg energy: {sum(final_energies)/len(final_energies):.6f}")

    # --- Denoising ---
    print("\n--- Denoising retrieval (noise variance=0.2) ---")
    corrupted = add_gaussian_noise(images, variance=0.2, seed=seed)

    t0 = time.time()
    denoised = retrieve_from_corruption(
        params,
        structure,
        corrupted,
        num_iterations=20,
        retrieval_infer_steps=200,
        seed=seed,
    )
    print(f"Retrieval time: {time.time() - t0:.1f}s")

    m = retrieval_metrics(images, denoised, threshold=0.005)
    print(
        f"Retrieved: {m['accuracy']*100:.1f}%  "
        f"mean_MSE={m['mean_mse']:.6f}  max_MSE={m['max_mse']:.6f}"
    )

    if save_images:
        save_reconstruction_grid(images, corrupted, denoised, "demo_denoising.png")

    # --- Partial retrieval (1/2 of pixels) ---
    print("\n--- Partial retrieval (fraction=0.5) ---")
    mask = generate_pixel_mask(N, data_dim, 0.5, seed=seed)

    t0 = time.time()
    partial_retrieved = retrieve_from_partial(
        params, structure, images, mask, infer_steps=500, seed=seed
    )
    print(f"Retrieval time: {time.time() - t0:.1f}s")

    m = retrieval_metrics(images, partial_retrieved, threshold=0.001)
    print(
        f"Retrieved: {m['accuracy']*100:.1f}%  "
        f"mean_MSE={m['mean_mse']:.6f}  max_MSE={m['max_mse']:.6f}"
    )

    if save_images:
        partial_vis = np.where(mask, images, 0.5)
        save_reconstruction_grid(
            images, partial_vis, partial_retrieved, "demo_partial.png"
        )

    # --- Partial retrieval (1/4 of pixels) ---
    print("\n--- Partial retrieval (fraction=0.25) ---")
    mask_quarter = generate_pixel_mask(N, data_dim, 0.25, seed=seed + 1)

    t0 = time.time()
    partial_quarter = retrieve_from_partial(
        params, structure, images, mask_quarter, infer_steps=500, seed=seed
    )
    print(f"Retrieval time: {time.time() - t0:.1f}s")

    m = retrieval_metrics(images, partial_quarter, threshold=0.001)
    print(
        f"Retrieved: {m['accuracy']*100:.1f}%  "
        f"mean_MSE={m['mean_mse']:.6f}  max_MSE={m['max_mse']:.6f}"
    )

    if save_images:
        partial_vis = np.where(mask_quarter, images, 0.5)
        save_reconstruction_grid(
            images, partial_vis, partial_quarter, "demo_partial_quarter.png"
        )


# ============================================================================
# Main
# ============================================================================


def main():
    args = parse_args()

    if args.experiment in ("demo",):
        experiment_demo(seed=args.seed, save_images=args.save_images)

    if args.experiment in ("denoising", "all"):
        experiment_denoising(
            num_images_list=(100, 250, 500),
            hidden_dims=(256, 512, 1024),
            seed=args.seed,
            save_images=args.save_images,
        )

    if args.experiment in ("partial", "all"):
        experiment_partial(seed=args.seed, save_images=args.save_images)

    if args.experiment in ("deep", "all"):
        experiment_deep(seed=args.seed, save_images=args.save_images)


if __name__ == "__main__":
    main()
