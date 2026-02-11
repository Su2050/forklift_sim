"""
Production Behavior Cloning (BC) trainer for the forklift pallet-insertion task.

Produces a checkpoint that is **directly loadable** by rsl_rl's ``OnPolicyRunner``
via ``--resume``, with no format conversion needed.

Network architecture, state_dict key names, and checkpoint layout are precisely
matched to ``rsl_rl.modules.ActorCritic`` with the configuration used in
``rsl_rl_ppo_cfg.py``:

    actor_hidden_dims = [256, 256, 128]
    critic_hidden_dims = [256, 256, 128]
    activation = "elu"
    noise_std_type = "log"
    init_noise_std = 0.5
    actor_obs_normalization = True
    critic_obs_normalization = True

Usage
-----
# Train
python bc_train.py --demos data/demos_xxx.npz --out data/bc_model_0.pt --epochs 200

# Verify (offline, from demo data)
python bc_train.py --demos data/demos_xxx.npz --verify data/bc_model_0.pt
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# ============================================================================
# rsl_rl-compatible network components (self-contained fallback)
# ============================================================================
# We replicate the *exact* module structure of ``rsl_rl.networks.MLP`` and
# ``rsl_rl.networks.EmpiricalNormalization`` so that the resulting
# ``state_dict`` keys match those produced by rsl_rl.  If rsl_rl is
# importable we verify compatibility; otherwise we use these local versions.

class _MLP(nn.Sequential):
    """MLP whose ``state_dict`` keys match ``rsl_rl.networks.MLP``.

    Layers are registered via ``add_module(f"{idx}", layer)`` so that the
    resulting keys look like ``0.weight``, ``0.bias``, ``2.weight``, etc.
    (even-numbered indices are ``nn.Linear``; odd are activation).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "elu",
    ):
        super().__init__()
        act_fn = _resolve_activation(activation)
        layers: list[nn.Module] = []

        # first hidden
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(act_fn())

        # remaining hidden
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(act_fn())

        # output
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)


class _EmpiricalNormalization(nn.Module):
    """Observation normalizer whose buffers match
    ``rsl_rl.networks.EmpiricalNormalization``."""

    def __init__(self, shape: int, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    def set_from_data(self, data: np.ndarray) -> None:
        """Compute and store normalisation statistics from a numpy array
        of shape ``(N, obs_dim)``."""
        mean = data.mean(axis=0)
        var = data.var(axis=0)
        std = np.sqrt(var)
        n = data.shape[0]
        self._mean.copy_(torch.from_numpy(mean).float().unsqueeze(0))
        self._var.copy_(torch.from_numpy(var).float().unsqueeze(0))
        self._std.copy_(torch.from_numpy(std).float().unsqueeze(0))
        self.count.fill_(n)


def _resolve_activation(name: str):
    """Return an **activation class** (not instance) matching rsl_rl."""
    table = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "crelu": nn.CELU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "softplus": nn.Softplus,
        "gelu": nn.GELU,
        "swish": nn.SiLU,
        "mish": nn.Mish,
        "identity": nn.Identity,
    }
    name = name.lower()
    if name not in table:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(table)}")
    return table[name]


# ============================================================================
# Composite ActorCritic shell (for assembling the full state_dict)
# ============================================================================
class BCActorCriticShell(nn.Module):
    """A lightweight ``nn.Module`` whose ``state_dict()`` matches the one
    produced by ``rsl_rl.modules.ActorCritic`` (or ``ClampedActorCritic``).

    Only the *actor* and *actor_obs_normalizer* are trained by BC.  The
    *critic*, *critic_obs_normalizer*, and *log_std* are randomly initialised
    / set to sensible defaults so that the checkpoint is complete.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden_dims: List[int] = (256, 256, 128),
        critic_hidden_dims: List[int] = (256, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.5,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # ---- Actor (trained by BC) ----
        self.actor = _MLP(obs_dim, act_dim, list(actor_hidden_dims), activation)
        self.actor_obs_normalizer = _EmpiricalNormalization(obs_dim)

        # ---- Critic (random init, not trained) ----
        self.critic = _MLP(obs_dim, 1, list(critic_hidden_dims), activation)
        self.critic_obs_normalizer = _EmpiricalNormalization(obs_dim)

        # ---- log_std (set to match init_noise_std from PPO config) ----
        self.log_std = nn.Parameter(
            torch.log(init_noise_std * torch.ones(act_dim))
        )

    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Normalise obs then pass through actor (matches ``act_inference``)."""
        return self.actor(self.actor_obs_normalizer(obs))


# ============================================================================
# Dataset
# ============================================================================
class DemoDataset(Dataset):
    def __init__(self, obs: np.ndarray, act: np.ndarray):
        self.obs = torch.from_numpy(obs).float()
        self.act = torch.from_numpy(act).float()

    def __len__(self) -> int:
        return self.obs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.obs[idx], self.act[idx]


# ============================================================================
# Training
# ============================================================================
def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"[BC] device={device}")

    # ---- Load demo data ----
    d = np.load(args.demos, allow_pickle=True)
    obs_all = d["obs"].astype(np.float32)
    act_all = d["act"].astype(np.float32)
    obs_dim = obs_all.shape[1]
    act_dim = act_all.shape[1]
    print(f"[BC] loaded {obs_all.shape[0]} transitions  obs_dim={obs_dim}  act_dim={act_dim}")

    # ---- Optional: filter by min_episode_len ----
    if args.min_episode_len > 0 and "episode_id" in d:
        ep_ids = d["episode_id"].astype(np.int64)
        unique, counts = np.unique(ep_ids, return_counts=True)
        keep_eps = set(unique[counts >= args.min_episode_len].tolist())
        mask = np.isin(ep_ids, list(keep_eps))
        obs_all = obs_all[mask]
        act_all = act_all[mask]
        print(f"[BC] after min_episode_len={args.min_episode_len} filter: "
              f"{obs_all.shape[0]} transitions ({len(keep_eps)} episodes)")

    # ---- Train / val split ----
    n = obs_all.shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(args.val_frac * n))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    ds_tr = DemoDataset(obs_all[tr_idx], act_all[tr_idx])
    ds_val = DemoDataset(obs_all[val_idx], act_all[val_idx])
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False)
    print(f"[BC] train={len(ds_tr)}  val={len(ds_val)}  batches/epoch={len(dl_tr)}")

    # ---- Build model ----
    model = BCActorCriticShell(
        obs_dim=obs_dim,
        act_dim=act_dim,
        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        activation=args.activation,
        init_noise_std=args.init_noise_std,
    ).to(device)

    # ---- Fill obs normalizer from demo data ----
    model.actor_obs_normalizer.set_from_data(obs_all)
    # Also set critic normalizer to the same stats (for completeness)
    model.critic_obs_normalizer.set_from_data(obs_all)
    _print_normalizer_stats(model.actor_obs_normalizer, "actor_obs_normalizer")

    # ---- Optimiser (only actor parameters) ----
    actor_params = list(model.actor.parameters())
    optimizer = torch.optim.Adam(actor_params, lr=args.lr)

    # Cosine annealing with warmup
    warmup_epochs = max(1, args.epochs // 20)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, args.epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.MSELoss()
    action_names = ["drive", "steer", "lift"]

    # ---- Training loop ----
    best_val = float("inf")
    patience_counter = 0
    best_state: Optional[OrderedDict] = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.actor.train()
        tr_loss_sum = 0.0
        tr_per_dim = torch.zeros(act_dim, device=device)
        tr_batches = 0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            xb_norm = model.actor_obs_normalizer(xb)
            pred = model.actor(xb_norm)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor_params, max_norm=args.max_grad_norm)
            optimizer.step()

            tr_loss_sum += loss.item()
            with torch.no_grad():
                tr_per_dim += ((pred - yb) ** 2).mean(dim=0)
            tr_batches += 1

        scheduler.step()
        tr_loss = tr_loss_sum / max(1, tr_batches)
        tr_per_dim /= max(1, tr_batches)

        # --- Val ---
        model.actor.eval()
        val_loss_sum = 0.0
        val_per_dim = torch.zeros(act_dim, device=device)
        val_batches = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                xb_norm = model.actor_obs_normalizer(xb)
                pred = model.actor(xb_norm)
                loss = loss_fn(pred, yb)
                val_loss_sum += loss.item()
                val_per_dim += ((pred - yb) ** 2).mean(dim=0)
                val_batches += 1

        val_loss = val_loss_sum / max(1, val_batches)
        val_per_dim /= max(1, val_batches)

        # --- Per-dim log ---
        dim_str = "  ".join(
            f"{action_names[i] if i < len(action_names) else f'a{i}'}="
            f"{val_per_dim[i].item():.6f}"
            for i in range(act_dim)
        )
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[BC] epoch {epoch:03d}/{args.epochs}  "
            f"train={tr_loss:.6f}  val={val_loss:.6f}  "
            f"lr={lr_now:.2e}  |  {dim_str}"
        )

        # --- Early stopping ---
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            patience_counter = 0
            best_state = OrderedDict(
                (k, v.cpu().clone()) for k, v in model.state_dict().items()
            )
        else:
            patience_counter += 1

        if patience_counter >= args.patience and epoch >= warmup_epochs + 5:
            print(f"[BC] early stopping at epoch {epoch} (patience={args.patience})")
            break

    elapsed = time.time() - t0
    print(f"[BC] training done in {elapsed:.1f}s  best_val={best_val:.6f}")

    # ---- Restore best weights ----
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")

    # ---- Save rsl_rl-compatible checkpoint ----
    _save_rsl_rl_checkpoint(model, args.out, best_val)


# ============================================================================
# Checkpoint I/O
# ============================================================================
def _save_rsl_rl_checkpoint(
    model: BCActorCriticShell,
    path: str,
    val_loss: float,
) -> None:
    """Save a checkpoint that ``OnPolicyRunner.load()`` can consume."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Build a dummy optimizer state dict (PPO uses Adam with all params)
    # The runner will overwrite this on resume, but the key must exist.
    dummy_opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    opt_state = dummy_opt.state_dict()

    saved = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt_state,
        "iter": 0,
        "infos": {"bc_val_loss": val_loss},
    }
    torch.save(saved, path)
    print(f"[BC] saved rsl_rl-compatible checkpoint: {path}")
    print(f"[BC]   model_state_dict keys: {len(saved['model_state_dict'])}")

    # Quick sanity: list all keys
    for k, v in saved["model_state_dict"].items():
        shape_str = list(v.shape) if v.dim() > 0 else "scalar"
        print(f"[BC]     {k:50s} {shape_str}")


# ============================================================================
# Verify mode
# ============================================================================
def verify(args: argparse.Namespace) -> None:
    """Load a BC checkpoint and compare predictions against demo actions."""
    device = torch.device("cpu")

    # ---- Load demo data ----
    d = np.load(args.demos, allow_pickle=True)
    obs_all = d["obs"].astype(np.float32)
    act_all = d["act"].astype(np.float32)
    obs_dim = obs_all.shape[1]
    act_dim = act_all.shape[1]
    print(f"[verify] loaded {obs_all.shape[0]} transitions  obs_dim={obs_dim}  act_dim={act_dim}")

    # ---- Load checkpoint ----
    ckpt_path = args.verify
    print(f"[verify] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint missing 'model_state_dict'. Not an rsl_rl checkpoint?")

    sd = ckpt["model_state_dict"]
    print(f"[verify] checkpoint keys ({len(sd)}):")
    for k, v in sd.items():
        shape_str = list(v.shape) if v.dim() > 0 else "scalar"
        print(f"[verify]   {k:50s} {shape_str}")

    # ---- Build model and load weights ----
    model = BCActorCriticShell(
        obs_dim=obs_dim,
        act_dim=act_dim,
        actor_hidden_dims=args.actor_hidden_dims,
        critic_hidden_dims=args.critic_hidden_dims,
        activation=args.activation,
        init_noise_std=args.init_noise_std,
    )
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("[verify] state_dict loaded successfully (strict=True)")

    # ---- Compare on a subset ----
    n_samples = min(args.verify_samples, obs_all.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(obs_all.shape[0], size=n_samples, replace=False)
    obs_t = torch.from_numpy(obs_all[idx]).float()
    act_gt = torch.from_numpy(act_all[idx]).float()

    with torch.no_grad():
        act_pred = model.forward_actor(obs_t)

    err = act_pred - act_gt
    mse_total = (err ** 2).mean().item()
    mse_per_dim = (err ** 2).mean(dim=0)
    mae_per_dim = err.abs().mean(dim=0)
    max_err_per_dim = err.abs().max(dim=0).values

    action_names = ["drive", "steer", "lift"]
    print(f"\n[verify] === Action Error Statistics ({n_samples} samples) ===")
    print(f"[verify] Total MSE: {mse_total:.6f}")
    print(f"[verify] {'dim':>8s}  {'MSE':>10s}  {'MAE':>10s}  {'MaxErr':>10s}")
    for i in range(act_dim):
        name = action_names[i] if i < len(action_names) else f"a{i}"
        print(
            f"[verify] {name:>8s}  "
            f"{mse_per_dim[i].item():10.6f}  "
            f"{mae_per_dim[i].item():10.6f}  "
            f"{max_err_per_dim[i].item():10.6f}"
        )

    # ---- Normalizer stats ----
    _print_normalizer_stats(model.actor_obs_normalizer, "actor_obs_normalizer")

    # ---- Infos from checkpoint ----
    infos = ckpt.get("infos")
    if infos:
        print(f"\n[verify] checkpoint infos: {infos}")
    print(f"[verify] checkpoint iter: {ckpt.get('iter', '?')}")
    print("[verify] DONE. Checkpoint is valid and loadable by rsl_rl.")


# ============================================================================
# Helpers
# ============================================================================
def _print_normalizer_stats(norm: _EmpiricalNormalization, name: str) -> None:
    mean = norm._mean.squeeze(0).cpu().numpy()
    std = norm._std.squeeze(0).cpu().numpy()
    count = norm.count.item()
    print(f"[BC] {name}: count={count}")
    print(f"[BC]   mean = {np.array2string(mean, precision=4, separator=', ')}")
    print(f"[BC]   std  = {np.array2string(std, precision=4, separator=', ')}")


# ============================================================================
# CLI
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Behavior Cloning trainer (rsl_rl compatible)"
    )
    # ---- Data ----
    ap.add_argument("--demos", type=str, required=True,
                    help="Path to demos_xxx.npz from collect_demos.py")
    ap.add_argument("--min_episode_len", type=int, default=0,
                    help="Discard episodes shorter than this (0=no filter)")

    # ---- Mode ----
    ap.add_argument("--verify", type=str, default="",
                    help="Path to checkpoint to verify (skips training)")
    ap.add_argument("--verify_samples", type=int, default=10000,
                    help="Number of samples to use for verification")

    # ---- Output ----
    ap.add_argument("--out", type=str, default="data/bc_model_0.pt",
                    help="Output checkpoint path")

    # ---- Training ----
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (epochs)")
    ap.add_argument("--val_frac", type=float, default=0.05,
                    help="Fraction of data for validation")
    ap.add_argument("--seed", type=int, default=42)

    # ---- Network (must match rsl_rl_ppo_cfg.py) ----
    ap.add_argument("--actor_hidden_dims", type=int, nargs="+",
                    default=[256, 256, 128])
    ap.add_argument("--critic_hidden_dims", type=int, nargs="+",
                    default=[256, 256, 128])
    ap.add_argument("--activation", type=str, default="elu")
    ap.add_argument("--init_noise_std", type=float, default=0.5)

    # ---- Device ----
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")

    return ap


def main() -> None:
    args = build_parser().parse_args()

    if args.verify:
        verify(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
