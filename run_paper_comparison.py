"""
Compare FabricPC associative memory results against the paper.

Paper: Salvatori et al. (2021) "Associative Memories via Predictive Coding"

Phase 1: Sweep training configs to find one that converges (energy -> 0).
Phase 2: With the converged model, sweep retrieval parameters.
Phase 3: Compare to paper claims.
"""
import sys, os, time, importlib.util
sys.path.insert(0, ".")

import numpy as np
import jax
import jax.numpy as jnp

from fabricpc import setup_jax
setup_jax(platform="cuda")

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
retrieve_from_partial = mod.retrieve_from_partial
add_gaussian_noise = mod.add_gaussian_noise
generate_pixel_mask = mod.generate_pixel_mask

SEED = 42
print(f"JAX devices: {jax.devices()}\n")

# ---- Load data ----
print("Loading CIFAR-10...")
images = load_cifar10_subset(50, seed=SEED)
print(f"Loaded {len(images)} images, dim={images.shape[1]}\n")


# =====================================================================
# PHASE 1: Train with best known config
# =====================================================================
# Best config from sweep: Adam lr=1e-4, T=32, eta=0.1, 30k epochs
# Energy: 10k->2.3, 30k->1.1, reconstruction: 98% at MSE<0.005
# =====================================================================
print("=" * 60)
print("  PHASE 1: Training (Adam lr=1e-4, T=32, eta=0.1, 30k epochs)")
print("=" * 60)

t0 = time.time()
params, structure, hist = train_generative_pcn(
    images, hidden_dim=512, num_hidden_layers=1,
    num_epochs=30000, lr=1e-4, infer_steps=32, eta_infer=0.1,
    seed=SEED, verbose=True,
)
train_time = time.time() - t0
avg_e = [sum(e)/len(e) if e else 0 for e in hist]
best_energy = avg_e[-1]
best_config = "Adam lr=1e-4, T=32, eta=0.1, 30k epochs"
print(f"\nTraining time: {train_time:.1f}s, final energy={best_energy:.4f}")
for ckpt in [5000, 10000, 15000, 20000, 25000, 30000]:
    if len(avg_e) >= ckpt:
        print(f"  E@{ckpt}: {avg_e[ckpt-1]:.4f}")


# =====================================================================
# PHASE 2: Check reconstruction quality
# =====================================================================
print("\n" + "=" * 60)
print("  PHASE 2: Reconstruction quality")
print("=" * 60)

sensory_name = structure.task_map["x"]
batch_size = len(images)
rng_key = jax.random.PRNGKey(SEED + 100)

clamps = {sensory_name: jnp.array(images)}
state = initialize_graph_state(
    structure, batch_size, rng_key, clamps=clamps, params=params
)
final_state = run_inference(params, state, clamps, structure)

z_mu = np.asarray(final_state.nodes[sensory_name].z_mu)
recon_mse = np.mean((images - z_mu) ** 2, axis=1)
print(f"Reconstruction MSE: mean={recon_mse.mean():.6f}, max={recon_mse.max():.6f}")
print(f"Reconstruction acc (MSE<0.005): {float(np.mean(recon_mse < 0.005))*100:.1f}%")
print(f"Reconstruction acc (MSE<0.001): {float(np.mean(recon_mse < 0.001))*100:.1f}%")


# =====================================================================
# PHASE 3: Denoising retrieval (Section 3)
# =====================================================================
print("\n" + "=" * 60)
print("  PHASE 3: Denoising retrieval (noise var=0.2, threshold=0.005)")
print("=" * 60)

corrupted = add_gaussian_noise(images, variance=0.2, seed=SEED)
noise_mse = float(np.mean((images - corrupted) ** 2))
print(f"Noise MSE: {noise_mse:.6f}")

# Paper retrieval params: F=30, T in {100,250,500}, eta in {1,0.5,0.1,0.05,0.01}
retrieval_configs = [
    (30, 100, 0.05), (30, 250, 0.05), (30, 500, 0.05),
    (30, 100, 0.1),  (30, 250, 0.1),  (30, 500, 0.1),
    (30, 100, 0.5),  (30, 250, 0.5),  (30, 500, 0.5),
    (30, 100, 1.0),  (30, 250, 1.0),  (30, 500, 1.0),
    (30, 100, 0.01), (30, 250, 0.01), (30, 500, 0.01),
]

print(f"\n{'F':>3} {'T':>4} {'eta':>5} {'Mean MSE':>10} {'Max MSE':>10} {'Acc(.005)':>10} {'Time':>7}")
print("-" * 55)

best_denoise_acc = 0
best_denoise_cfg = None

for F, T, eta in retrieval_configs:
    t0 = time.time()
    retrieved = retrieve_from_corruption(
        params, structure, corrupted,
        num_iterations=F, retrieval_infer_steps=T,
        retrieval_eta_infer=eta, seed=SEED,
    )
    elapsed = time.time() - t0

    if np.any(np.isnan(retrieved)):
        print(f"{F:>3} {T:>4} {eta:>5.2f}       NaN         NaN       NaN  {elapsed:>6.1f}s")
        continue

    per_img = np.mean((images - retrieved) ** 2, axis=1)
    acc = float(np.mean(per_img < 0.005))
    mean_mse = float(np.mean(per_img))
    max_mse = float(np.max(per_img))
    print(f"{F:>3} {T:>4} {eta:>5.2f} {mean_mse:>10.6f} {max_mse:>10.6f} {acc*100:>9.1f}% {elapsed:>6.1f}s")

    if acc > best_denoise_acc:
        best_denoise_acc = acc
        best_denoise_cfg = (F, T, eta)

if best_denoise_cfg:
    print(f"\nBest denoising: F={best_denoise_cfg[0]} T={best_denoise_cfg[1]} "
          f"eta={best_denoise_cfg[2]} -> {best_denoise_acc*100:.1f}% correct")


# =====================================================================
# PHASE 4: Partial retrieval (Section 4)
# =====================================================================
print("\n" + "=" * 60)
print("  PHASE 4: Partial retrieval (threshold=0.001)")
print("=" * 60)

# Try multiple eta values for partial retrieval too
for frac in [0.5, 0.25, 0.125, 0.0625]:
    mask = generate_pixel_mask(50, 3072, frac, seed=SEED)
    known = int(mask.sum(axis=1).mean())
    print(f"\n  Fraction={frac:.4f} ({known} known pixels):")

    for F, T, eta in [(30, 250, 0.01), (30, 500, 0.01), (30, 250, 0.05), (30, 500, 0.05)]:
        t0 = time.time()
        retrieved = retrieve_from_partial(
            params, structure, images, mask,
            num_iterations=F, retrieval_infer_steps=T,
            retrieval_eta_infer=eta, seed=SEED,
        )
        elapsed = time.time() - t0

        if np.any(np.isnan(retrieved)):
            print(f"    F={F} T={T} eta={eta:.2f}: NaN  t={elapsed:.1f}s")
            continue

        per_img = np.mean((images - retrieved) ** 2, axis=1)
        acc_001 = float(np.mean(per_img < 0.001))
        acc_005 = float(np.mean(per_img < 0.005))
        mean_mse = float(np.mean(per_img))
        print(f"    F={F} T={T} eta={eta:.2f}: MSE={mean_mse:.6f}  "
              f"Acc(.001)={acc_001*100:.1f}%  Acc(.005)={acc_005*100:.1f}%  t={elapsed:.1f}s")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("  COMPARISON: Paper vs Our Results")
print("=" * 60)
print(f"""
Training:
  Final energy: {best_energy:.4f}
  Config: {best_config}

Paper (Salvatori et al., 2021):
  Sec 3 (Denoising): noise=0.2, threshold=0.005
    PCN-256 retrieves ~250 CIFAR-10 images
    PCN-2048 retrieves ALL for N=100..1500
  Sec 4 (Partial): threshold=0.001
    N=50, h=1024 retrieves ALL from 1/4 pixels (TinyImageNet)
  Hyperparameters:
    Weight lr: 1e-4 or 5e-5 (SGD-like)
    Inference lr: {{1, 0.5, 0.1, 0.05, 0.01}}
    Training T: {{12, 16, 24, 32}}
    Retrieval T: {{100, 250, 500}}
    F iterations: 30

Our setup:
  PreSynapticLinear: z_mu = W @ ReLU(x) + b
  FabricPC framework, Adam optimizer
  Warm-start hidden states across F iterations
  N=50, hidden=512
""")
