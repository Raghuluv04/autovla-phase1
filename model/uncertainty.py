"""Milestone 8 (team contribution): Bayesian uncertainty over planned trajectories.

Two branches read the model's pre-decoder representation -- the vision encoder's
output tokens (when the scene has images) pooled together with the prompt token
embeddings and the ego state -- and a policy-level sampler:

  Branch A, aleatoric  ``AleatoricHead``: an MLP predicting the log-variance of
      the policy's trajectory error (heteroscedastic Gaussian NLL, Kendall & Gal
      2017). sigma^2_ale is the irreducible, input-dependent error scale in m^2.
  Branch B, epistemic  ``EpistemicHead``: variance of the decoded trajectory over
      T stochastic passes. ``mc_dropout`` keeps every ``nn.Dropout`` (the LoRA
      dropout, p=0.1) active at inference; ``ensemble`` cycles through members.
      sigma^2_epi is the mean over steps of the summed x/y position variance, m^2.

  confidence = 1 / (1 + sigma^2_ale + sigma^2_epi)          (fuse_confidence)

Calibration lives in :mod:`model.calibration`.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from autovla.config import COMMANDS
from autovla.schema import DrivingSample

from .backbone import Backbone
from .calibration import (TemperatureScaler, bootstrap_ci, compute_auroc, compute_ece, fit_calibrators, nll,
                          select_calibrator, temperature_scaling)
from .config import (CALIB_TAU_FAIL, ECE_TARGET, AUROC_TARGET, MC_DROPOUT_T, N_ENSEMBLE,
                     PHASE2_DIR, UNC_DROPOUT_P)
from .inference import PromptEncoder, generate_action_batch, generate_plan

LOGVAR_MIN, LOGVAR_MAX = -6.0, 8.0
EGO_DIM = 2 + len(COMMANDS)
SPEED_SCALE, ACCEL_SCALE = 10.0, 3.0


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
def ego_features(s: DrivingSample) -> np.ndarray:
    onehot = np.zeros(len(COMMANDS))
    if s.command in COMMANDS:
        onehot[COMMANDS.index(s.command)] = 1.0
    return np.concatenate([[s.velocity / SPEED_SCALE, s.acceleration / ACCEL_SCALE], onehot])


def _image_tokens(bb: Backbone, pixel_values, grid):
    base = bb.model.get_base_model() if hasattr(bb.model, "get_base_model") else bb.model
    inner = getattr(base, "model", base)
    fn = getattr(inner, "get_image_features", None) or base.get_image_features
    out = fn(pixel_values, grid)
    out = getattr(out, "pooler_output", out)
    return torch.cat(list(out)) if isinstance(out, (list, tuple)) else out


class FeatureExtractor:
    """Pooled pre-decoder features: ``[visual | prompt | ego]`` (dimension 2H + EGO_DIM)."""

    def __init__(self, bb: Backbone):
        self.bb, self.enc = bb, PromptEncoder(bb.tokenizer, bb.processor)
        self.hidden = bb.model.get_input_embeddings().weight.shape[1]
        self.image_id = bb.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    @property
    def dim(self) -> int:
        return 2 * self.hidden + EGO_DIM

    @torch.no_grad()
    def __call__(self, messages, sample: DrivingSample) -> torch.Tensor:
        inputs = {k: v.to(self.bb.device) for k, v in self.enc(messages, None).items()}
        ids = inputs["input_ids"][0]
        emb = self.bb.model.get_input_embeddings()(ids).float()
        text = emb[ids != self.image_id].mean(0)
        if "pixel_values" in inputs:
            vis = _image_tokens(self.bb, inputs["pixel_values"], inputs["image_grid_thw"]).float().mean(0)
        else:
            vis = torch.zeros(self.hidden, device=emb.device)
        ego = torch.as_tensor(ego_features(sample), dtype=torch.float32, device=emb.device)
        return torch.cat([vis, text, ego]).cpu()


# ---------------------------------------------------------------------------
# Branch A: aleatoric
# ---------------------------------------------------------------------------
class AleatoricHead(nn.Module):
    """Features -> log sigma^2 of the policy's trajectory error (scalar per scene)."""

    def __init__(self, in_dim: int, hidden: int = 64, p: float = UNC_DROPOUT_P):
        super().__init__()
        self.register_buffer("mu", torch.zeros(in_dim))
        self.register_buffer("sd", torch.ones(in_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(p), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd).squeeze(-1).clamp(LOGVAR_MIN, LOGVAR_MAX)

    @torch.no_grad()
    def sigma2(self, x) -> torch.Tensor:
        was = self.training
        self.eval()
        out = self(x).exp()
        self.train(was)
        return out


def gaussian_nll(log_var, sq_err):
    """Heteroscedastic regression loss: 0.5 * (exp(-s) * e + s)."""
    return (0.5 * (torch.exp(-log_var) * sq_err + log_var)).mean()


def train_aleatoric_head(X: torch.Tensor, sq_err: torch.Tensor, epochs: int = 300, lr: float = 3e-3,
                         weight_decay: float = 1e-4, batch_size: int = 256, val_frac: float = 0.2,
                         patience: int = 30, seed: int = 0):
    torch.manual_seed(seed)                  # head init and dropout masks, not just the shuffles
    g = torch.Generator().manual_seed(seed)
    n = len(X)
    perm = torch.randperm(n, generator=g)
    n_val = max(int(n * val_frac), 1)
    va, tr = perm[:n_val], perm[n_val:]
    head = AleatoricHead(X.shape[1])
    head.mu.copy_(X[tr].mean(0)); head.sd.copy_(X[tr].std(0).clamp(min=1e-6))
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, bad, hist = float("inf"), None, 0, []
    for ep in range(epochs):
        head.train()
        order = tr[torch.randperm(len(tr), generator=g)]
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            loss = gaussian_nll(head(X[b]), sq_err[b])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            v = float(gaussian_nll(head(X[va]), sq_err[va]))
        hist.append(v)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    head.eval()
    return head, dict(val_nll=best, epochs=len(hist), history=hist)


# ---------------------------------------------------------------------------
# Branch B: epistemic
# ---------------------------------------------------------------------------
@contextmanager
def mc_dropout(model: nn.Module):
    """Switch every ``nn.Dropout`` to train mode (stochastic) inside the block."""
    drops = [m for m in model.modules() if isinstance(m, nn.Dropout) and m.p > 0]
    was = [m.training for m in drops]
    for m in drops:
        m.train()
    try:
        yield len(drops)
    finally:
        for m, w in zip(drops, was):
            m.train(w)


@dataclass
class EpistemicResult:
    sigma2: float
    mean_traj: np.ndarray
    trajs: np.ndarray               # (T, steps+1, 3)

    @property
    def step_variance(self) -> np.ndarray:
        return self.trajs[:, 1:, :2].var(axis=0, ddof=1).sum(-1)


class EpistemicHead:
    """Variance of the plan over T stochastic samples.

    ``sampler(i)`` returns one decoded trajectory; for MC-dropout it is the same
    model with dropout on, for an ensemble it dispatches to member ``i mod K``.
    """

    def __init__(self, T: int = MC_DROPOUT_T, mode: str = "mc_dropout"):
        if mode not in ("mc_dropout", "ensemble"):
            raise ValueError(mode)
        self.T, self.mode = T, mode

    def estimate(self, sampler: Callable[[int], np.ndarray]) -> EpistemicResult:
        return self.from_trajs(np.stack([np.asarray(sampler(i), float) for i in range(self.T)]))

    @staticmethod
    def from_trajs(trajs: np.ndarray) -> EpistemicResult:
        """(T, steps+1, 3) stochastic plans -> sigma^2_epi (m^2)."""
        res = EpistemicResult(0.0, trajs.mean(0), trajs)
        res.sigma2 = float(res.step_variance.mean())
        return res


def mc_dropout_sampler(bb: Backbone, messages, codec, seed: int = 0):
    """Closure drawing one dropout-perturbed greedy plan per call."""
    def sample(i: int) -> np.ndarray:
        torch.manual_seed(seed * 7919 + i)
        plan = generate_plan(bb, messages, mode="fast")
        return codec.decode(plan.action_ids)
    return sample


def mc_dropout_batch(bb: Backbone, messages, codec, T: int, seed: int = 0, encoder=None) -> np.ndarray:
    """(T, steps+1, 3): T dropout-perturbed plans from one batched generate call."""
    torch.manual_seed(seed)
    with mc_dropout(bb.model):
        ids = generate_action_batch(bb, messages, T, encoder=encoder)
    return np.stack([codec.decode(i) for i in ids])


def ensemble_sampler(members: list[Backbone], messages, codec):
    def sample(i: int) -> np.ndarray:
        return codec.decode(generate_plan(members[i % len(members)], messages, mode="fast").action_ids)
    return sample


def fuse_confidence(sigma2_ale, sigma2_epi):
    """confidence = 1 / (1 + sigma^2_ale + sigma^2_epi), in (0, 1]."""
    return 1.0 / (1.0 + np.asarray(sigma2_ale, float) + np.asarray(sigma2_epi, float))


# ---------------------------------------------------------------------------
# scene-level outputs
# ---------------------------------------------------------------------------
@dataclass
class SceneOutputs:
    ids: list
    features: np.ndarray            # (N, D)
    sq_err: np.ndarray              # (N,) mean squared waypoint error of the greedy plan, m^2
    ade: np.ndarray                 # (N,)
    epi: np.ndarray                 # (N,) sigma^2_epi
    hard: np.ndarray                # (N,) bool
    pred: np.ndarray                # (N, steps+1, 3)
    pconf: np.ndarray = None        # (N,) policy-intrinsic confidence of the greedy fast plan (token probs)
    feat_s: np.ndarray = None       # (N,) seconds: feature extraction (aleatoric-only gate overhead)
    est_s: np.ndarray = None        # (N,) seconds: features + T MC passes (full-confidence overhead)

    def __len__(self):
        return len(self.ids)

    def subset(self, idx):
        idx = np.asarray(idx)
        pick = lambda a: None if a is None else a[idx]
        return SceneOutputs([self.ids[i] for i in idx], self.features[idx], self.sq_err[idx],
                            self.ade[idx], self.epi[idx], self.hard[idx], self.pred[idx],
                            pick(self.pconf), pick(self.feat_s), pick(self.est_s))

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, ids=np.array(self.ids), features=self.features, sq_err=self.sq_err,
                            ade=self.ade, epi=self.epi, hard=self.hard, pred=self.pred,
                            pconf=self.pconf, feat_s=self.feat_s, est_s=self.est_s)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        get = lambda k: z[k] if k in z.files else None
        return cls(list(z["ids"]), z["features"], z["sq_err"], z["ade"], z["epi"], z["hard"], z["pred"],
                   get("pconf"), get("feat_s"), get("est_s"))


def collect_scene_outputs(bb: Backbone, items, codec, T: int = MC_DROPOUT_T, seed: int = 0,
                          extractor: FeatureExtractor | None = None, progress: int = 0) -> SceneOutputs:
    from .grpo import GRPOConfig, policy_confidence, sample_rollout, token_logprobs

    extractor = extractor or FeatureExtractor(bb)
    epi_head = EpistemicHead(T) if T > 0 else None
    enc, gcfg = PromptEncoder(bb.tokenizer, bb.processor), GRPOConfig()
    ids, feats, sq, ade_, epi, hard, preds, t_feat, t_est, pconf = [], [], [], [], [], [], [], [], [], []
    t_start = time.perf_counter()
    for n, it in enumerate(items):
        gt = np.asarray(it.sample.future_traj, float)
        pred = codec.decode(generate_plan(bb, it.messages, mode="fast").action_ids)   # dropout off
        err = np.linalg.norm(pred[1:, :2] - gt[1:, :2], axis=1)
        t0 = time.perf_counter()
        feat = extractor(it.messages, it.sample).numpy()
        t1 = time.perf_counter()
        if epi_head is not None:                        # T=0 skips the (10x costlier) epistemic passes
            res_sigma2 = epi_head.from_trajs(
                mc_dropout_batch(bb, it.messages, codec, T, seed * 7919 + n, enc)).sigma2
        else:
            res_sigma2 = 0.0
        t_feat.append(t1 - t0); t_est.append(time.perf_counter() - t0)
        ro = sample_rollout(bb, enc, it.messages, codec, gt, gcfg, greedy=True)
        with torch.no_grad():
            pconf.append(policy_confidence(token_logprobs(bb, dict(enc(it.messages, None)), ro), ro.roles))
        ids.append(it.sample.sample_id); feats.append(feat)
        sq.append(float((err ** 2).mean())); ade_.append(float(err.mean()))
        epi.append(res_sigma2); hard.append(it.hard); preds.append(pred)
        if progress and (n + 1) % progress == 0:
            el = time.perf_counter() - t_start
            print(f"  collected {n + 1}/{len(items)}  ({el / (n + 1):.1f} s/scene, "
                  f"~{el / (n + 1) * (len(items) - n - 1) / 60:.0f} min left)", flush=True)
    return SceneOutputs(ids, np.stack(feats), np.array(sq), np.array(ade_), np.array(epi),
                        np.array(hard), np.stack(preds), np.array(pconf), np.array(t_feat), np.array(t_est))


# ---------------------------------------------------------------------------
# fit + evaluate
# ---------------------------------------------------------------------------
@dataclass
class UncertaintyModel:
    head: AleatoricHead
    scaler: TemperatureScaler
    tau: float
    T: int = MC_DROPOUT_T

    def sigma2_ale(self, feat: torch.Tensor) -> float:
        return float(self.head.sigma2(feat.unsqueeze(0))[0])

    def confidence(self, sigma2_ale, sigma2_epi=0.0, calibrated: bool = True):
        c = fuse_confidence(sigma2_ale, sigma2_epi)
        return self.scaler(c) if calibrated else c

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(head=self.head.state_dict(), in_dim=self.head.mu.numel(),
                        temperature=self.scaler.temperature, bias=self.scaler.bias, kind=self.scaler.kind,
                        tau=self.tau, T=self.T), path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, map_location="cpu")
        head = AleatoricHead(d["in_dim"]); head.load_state_dict(d["head"]); head.eval()
        return cls(head, TemperatureScaler(d["temperature"], d["bias"], d.get("kind", "logit")), d["tau"], d["T"])


def fit_and_evaluate(train: SceneOutputs, calib: SceneOutputs, test: SceneOutputs,
                     tau: float | str = CALIB_TAU_FAIL, calibrator: str = "auto", seed: int = 0,
                     epochs: int = 300):
    """Train Branch A on ``train``, calibrate on ``calib``, report on ``test``.

    ``calibrator``: ``logit_T`` (classic temperature scaling), ``variance_T``,
    ``logit_T_bias``, or ``auto`` = whichever of the three has the lowest NLL on
    ``calib``. Every candidate's test ECE/NLL is reported under ``calibrators``.
    """
    if tau == "auto":
        tau = float(np.median(calib.ade))
    head, info = train_aleatoric_head(torch.as_tensor(train.features, dtype=torch.float32),
                                      torch.as_tensor(train.sq_err, dtype=torch.float32),
                                      epochs=epochs, seed=seed)
    sig_a = lambda o: head.sigma2(torch.as_tensor(o.features, dtype=torch.float32)).numpy()
    ca, ct = sig_a(calib), sig_a(test)
    success = lambda o: (o.ade <= tau).astype(float)
    conf_c, conf_t = fuse_confidence(ca, calib.epi), fuse_confidence(ct, test.epi)
    cands = fit_calibrators(conf_c, success(calib))
    scaler = select_calibrator(conf_c, success(calib)) if calibrator == "auto" else cands[calibrator]
    cal_t = scaler(conf_t)
    y_t, fail_t = success(test), 1.0 - success(test)
    unc_t = ct + test.epi

    def block(conf):
        return dict(ece=compute_ece(conf, y_t), nll=nll(conf, y_t), auroc=compute_auroc(1 - conf, fail_t))

    hard, easy = test.hard, ~test.hard
    sep = lambda u, h: compute_auroc(u, h.astype(float))
    res = dict(
        tau=tau, n=dict(train=len(train), calib=len(calib), test=len(test)),
        failure_rate=dict(test=float(fail_t.mean()), calib=float(1 - success(calib).mean())),
        temperature=scaler.temperature, bias=scaler.bias, calibrator=scaler.name, head_val_nll=info["val_nll"],
        calibrators={k: dict(temperature=c.temperature, bias=c.bias, ece=compute_ece(c(conf_t), y_t),
                             nll=nll(c(conf_t), y_t), calib_nll=nll(c(conf_c), success(calib)))
                     for k, c in cands.items()},
        uncalibrated=block(conf_t), calibrated=block(cal_t),
        auroc_failure=dict(
            fused=compute_auroc(unc_t, fail_t), aleatoric_only=compute_auroc(ct, fail_t),
            epistemic_only=compute_auroc(test.epi, fail_t),
            ci_fused=bootstrap_ci(compute_auroc, unc_t, fail_t, n=300, seed=seed)),
        hard_vs_easy=dict(
            n_hard=int(hard.sum()), n_easy=int(easy.sum()),
            mean_unc_hard=float(unc_t[hard].mean()) if hard.any() else float("nan"),
            mean_unc_easy=float(unc_t[easy].mean()) if easy.any() else float("nan"),
            mean_conf_hard=float(cal_t[hard].mean()) if hard.any() else float("nan"),
            mean_conf_easy=float(cal_t[easy].mean()) if easy.any() else float("nan"),
            ale_hard=float(ct[hard].mean()) if hard.any() else float("nan"),
            ale_easy=float(ct[easy].mean()) if easy.any() else float("nan"),
            epi_hard=float(test.epi[hard].mean()) if hard.any() else float("nan"),
            epi_easy=float(test.epi[easy].mean()) if easy.any() else float("nan"),
            auroc_hard_from_unc=sep(unc_t, hard)),
    )
    res["targets"] = dict(
        ece_below_target=res["calibrated"]["ece"] < ECE_TARGET,
        auroc_above_target=res["auroc_failure"]["fused"] > AUROC_TARGET,
        hard_more_uncertain=res["hard_vs_easy"]["mean_unc_hard"] > res["hard_vs_easy"]["mean_unc_easy"])
    return UncertaintyModel(head, scaler, tau), res


def split_items(items, n_train: int, n_calib: int, n_test: int, seed: int = 0):
    perm = np.random.default_rng(seed).permutation(len(items))
    pick = lambda a, b: [items[i] for i in perm[a:b]]
    return (pick(0, n_train), pick(n_train, n_train + n_calib),
            pick(n_train + n_calib, n_train + n_calib + n_test))


def main():
    from .evaluate import build_eval_items
    from .config import BackboneConfig
    from .runtime import load_action_codec, make_backbone

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--standin", action="store_true")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--holdout", type=int, default=3000, help="must equal the SFT hold-out size")
    ap.add_argument("--n-train", type=int, default=1200)
    ap.add_argument("--n-calib", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=800)
    ap.add_argument("--T", type=int, default=MC_DROPOUT_T)
    ap.add_argument("--tau", default="auto", help="failure threshold in metres, or 'auto' (median calib ADE)")
    ap.add_argument("--calibrator", default="auto", choices=["auto", "logit_T", "variance_T", "logit_T_bias"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--compute-dtype", default=None, help="float16 on a T4 (no bf16)")
    ap.add_argument("--unseen-sft-config", default=None,
                    help="SFT yaml; draw scenes from records that run never saw (needs --sft-steps). "
                         "Use when the hold-out is smaller than n-train+n-calib+n-test")
    ap.add_argument("--sft-steps", type=int, default=None, help="optimizer steps of the (un-resumed) SFT run")
    ap.add_argument("--progress", type=int, default=25)
    args = ap.parse_args()

    tag = args.tag or ("standin" if args.standin else "model")
    bcfg = BackboneConfig()
    if args.compute_dtype:
        bcfg.compute_dtype = args.compute_dtype
    bb = make_backbone(args.standin, bcfg, args.checkpoint)
    codec = load_action_codec()
    cache = Path(args.cache or PHASE2_DIR / f"unc_scenes_{tag}.npz")
    if cache.exists():
        allo = SceneOutputs.load(cache)
        print(f"loaded {len(allo)} cached scene outputs from {cache}")
    else:
        n_all = args.n_train + args.n_calib + args.n_test
        if args.unseen_sft_config:
            from .trainer import SFTConfig, unseen_ids

            if not args.sft_steps:
                raise SystemExit("--unseen-sft-config needs --sft-steps")
            pool = unseen_ids(SFTConfig.from_yaml(args.unseen_sft_config), args.sft_steps)
            print(f"{len(pool)} records were never drawn by the {args.sft_steps}-step SFT run", flush=True)
            items = build_eval_items(n_all, ids=pool, obstacles=None, shuffle_seed=0)
        else:
            items = build_eval_items(n_all, holdout=args.holdout, obstacles=None)
        if len(items) < n_all:
            raise SystemExit(f"only {len(items)} usable scenes, need {n_all}; lower --n-* or retrain with a larger hold-out")
        import torch

        with torch.autocast("cuda", dtype=getattr(torch, bcfg.compute_dtype), enabled=bb.device.type == "cuda"):
            allo = collect_scene_outputs(bb, items, codec, args.T, progress=args.progress)
        allo.save(cache)
    idx = np.random.default_rng(0).permutation(len(allo))
    a, b = args.n_train, args.n_train + args.n_calib
    tr, ca, te = allo.subset(idx[:a]), allo.subset(idx[a:b]), allo.subset(idx[b:b + args.n_test])
    tau = args.tau if args.tau == "auto" else float(args.tau)
    model, res = fit_and_evaluate(tr, ca, te, tau, args.calibrator)
    out = Path(PHASE2_DIR); out.mkdir(parents=True, exist_ok=True)
    model.save(out / f"unc_head_{tag}.pt")
    (out / f"unc_report_{tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
