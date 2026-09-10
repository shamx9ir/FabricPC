"""
Diagnose training convergence and retrieval for CIFAR-10.

Tests:
1. Does training converge with different hyperparameters?
2. After training, can the model reconstruct its own training data?
3. Does retrieval work from corrupted inputs?
"""
import sys, os, time, importlib.util
sys.path.insert(0, ".")

import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc import setup_jax
setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

spec = importlib.util.spec_from_file_location(
    "assoc_mem", os.path.join("examples", "associative_memory_pcn.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import run_inference

load_cifar10_subset = mod.load_cifar10_subset
build_generative_pcn = mod.build_generative_pcn
train_generative_pcn = mod.train_generative_pcn
retrieve_from_corruption = mod.retrieve_from_corruption
add_gaussian_noise = mod.add_gaussian_noise
MemorizationLoader = mod.MemorizationLoader

from fabricpc.training import train_pcn

SEED = 42

print(f"JAX devices: {jax.devices()}")

# Load CIFAR-10
print("Loading CIFAR-10...")
images = load_cifar10_subset(50, seed=SEED)
print(f"Loaded {len(images)} images, dim={images.shape[1]}\n")

# ===================================================================
# TEST 1: Hyperparameter sweep for convergence
# ===================================================================
print("=" * 60)
print("  TEST 1: Training convergence sweep")
print("=" * 60)

configs = [
    # (lr, eta_infer, infer_steps, epochs, optimizer_name)
    (1e-4, 0.1,  24, 3000, "adam"),
    (5e-4, 0.1,  24, 3000, "adam"),
    (1e-3, 0.1,  24, 3000, "adam"),
    (1e-4, 0.5,  24, 3000, "adam"),
    (1e-4, 1.0,  24, 3000, "adam"),
    (1e-4, 0.1,  32, 3000, "adam"),
    (1e-4, 0.1,  24, 3000, "sgd"),   # paper uses SGD
    (1e-3, 0.1,  24, 3000, "sgd"),
]

best_config = None
best_energy = float('inf')

print(f"\n{'lr':>8} {'eta':>5} {'T':>4} {'opt':>5} {'E@500':>10} {'E@1000':>10} "
      f"{'E@2000':>10} {'E@3000':>10} {'Time':>7}")
print("-" * 75)

for lr, eta, T, epochs, opt_name in configs:
    data_dim = images.shape[1]
    structure = build_generative_pcn(
        data_dim=data_dim, hidden_dim=512, num_hidden_layers=1,
        infer_steps=T, eta_infer=eta,
    )
    rng_key = jax.random.PRNGKey(SEED)
    rng_key, init_key = jax.random.split(rng_key)
    params = initialize_params(structure, init_key)

    if opt_name == "adam":
        optimizer = optax.adam(lr)
    else:
        optimizer = optax.sgd(lr)

    loader = MemorizationLoader(images, batch_size=None, seed=SEED)
    rng_key, train_key = jax.random.split(rng_key)

    t0 = time.time()
    trained_params, energy_history, _ = train_pcn(
        params=params, structure=structure, train_loader=loader,
        optimizer=optimizer, config={"num_epochs": epochs},
        rng_key=train_key, verbose=False,
    )
    elapsed = time.time() - t0

    avg_e = [sum(e)/len(e) if e else 0 for e in energy_history]
    e500 = avg_e[499] if len(avg_e) > 499 else float('nan')
    e1000 = avg_e[999] if len(avg_e) > 999 else float('nan')
    e2000 = avg_e[1999] if len(avg_e) > 1999 else float('nan')
    e3000 = avg_e[2999] if len(avg_e) > 2999 else avg_e[-1]

    print(f"{lr:>8.0e} {eta:>5.1f} {T:>4} {opt_name:>5} {e500:>10.2f} {e1000:>10.2f} "
          f"{e2000:>10.2f} {e3000:>10.2f} {elapsed:>6.1f}s")

    if e3000 < best_energy:
        best_energy = e3000
        best_config = (lr, eta, T, epochs, opt_name)
        best_params = trained_params
        best_structure = structure
        best_history = avg_e

print(f"\nBest config: lr={best_config[0]}, eta={best_config[1]}, T={best_config[2]}, "
      f"opt={best_config[4]}, energy={best_energy:.4f}")


# ===================================================================
# TEST 2: Reconstruction quality after training
# ===================================================================
print("\n" + "=" * 60)
print("  TEST 2: Reconstruction quality (training data)")
print("=" * 60)

# Use the best model - clamp training images, run inference, read z_mu
sensory_name = best_structure.task_map["x"]
batch_size = len(images)
rng_key = jax.random.PRNGKey(SEED + 100)

clamps = {sensory_name: jnp.array(images)}
state = initialize_graph_state(
    best_structure, batch_size, rng_key, clamps=clamps, params=best_params
)
final_state = run_inference(best_params, state, clamps, best_structure)

# z_mu of sensory = prediction from hidden layer
z_mu_sensory = np.asarray(final_state.nodes[sensory_name].z_mu)
recon_mse = np.mean((images - z_mu_sensory) ** 2, axis=1)
print(f"Sensory z_mu MSE (per image): mean={recon_mse.mean():.6f}, "
      f"max={recon_mse.max():.6f}")

# Print per-node energies
for node_name in best_structure.nodes:
    node_e = float(jnp.sum(final_state.nodes[node_name].energy)) / batch_size
    print(f"  Node '{node_name}' energy: {node_e:.4f}")

# Check if model can reproduce its data with z_mu
acc_005 = float(np.mean(recon_mse < 0.005))
acc_001 = float(np.mean(recon_mse < 0.001))
print(f"Reconstruction accuracy (MSE<0.005): {acc_005*100:.1f}%")
print(f"Reconstruction accuracy (MSE<0.001): {acc_001*100:.1f}%")


# ===================================================================
# TEST 3: Retrieval with different parameters
# ===================================================================
print("\n" + "=" * 60)
print("  TEST 3: Retrieval parameter sweep")
print("=" * 60)

corrupted = add_gaussian_noise(images, variance=0.2, seed=SEED)
noise_mse = float(np.mean((images - corrupted) ** 2))
print(f"Noise MSE: {noise_mse:.6f}")

retrieval_configs = [
    # (F_iters, T_retrieval, eta_retrieval)
    (30, 100, 0.01),
    (30, 200, 0.01),
    (30, 500, 0.01),
    (30, 200, 0.05),
    (30, 200, 0.1),
    (50, 200, 0.01),
    (50, 500, 0.01),
]

print(f"\n{'F':>4} {'T':>5} {'eta':>6} {'Mean MSE':>10} {'Acc(0.005)':>10} {'Time':>7}")
print("-" * 50)

for F, T, eta in retrieval_configs:
    t0 = time.time()
    retrieved = retrieve_from_corruption(
        best_params, best_structure, corrupted,
        num_iterations=F, retrieval_infer_steps=T,
        retrieval_eta_infer=eta, seed=SEED,
    )
    elapsed = time.time() - t0

    per_img = np.mean((images - retrieved) ** 2, axis=1)
    acc = float(np.mean(per_img < 0.005))
    mean_mse = float(np.mean(per_img))

    print(f"{F:>4} {T:>5} {eta:>6.3f} {mean_mse:>10.6f} {acc*100:>9.1f}% {elapsed:>6.1f}s")

print("\nDone!")
