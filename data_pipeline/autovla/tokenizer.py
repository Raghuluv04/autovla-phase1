"""Action tokenizer (AutoVLA Appendix A.2).

Encoding: split a trajectory into 0.5 s segments, express each in the frame of
its own first pose, and map it to the nearest codebook entry under contour
distance -> [a_1, ..., a_T].

Decoding: look each token up in the codebook and compose the motion segments
sequentially from the initial ego pose, reconstructing a continuous trajectory
in the ego frame.

The tokens are surfaced to the language model as literal text tokens
<action_0> ... <action_2047>, added to the tokenizer vocabulary, so trajectory
prediction becomes next-token prediction in the same stream as the reasoning.
"""

from __future__ import annotations

import numpy as np

from .codebook import Codebook, _contour_points
from .config import N_ACTION_TOKENS
from .geometry import canonical_contour, compose, from_deltas, relative, to_deltas

ACTION_TOKEN_FMT = "<action_{}>"


class ActionTokenizer:
    def __init__(self, codebook: Codebook, chunk=4096):
        self.codebook = codebook
        self.contour = canonical_contour(
            codebook.meta.get("ego_length"), codebook.meta.get("ego_width"),
            codebook.meta.get("contour_points"))
        self._flat_cb, _ = _contour_points(codebook.actions, self.contour)
        self.chunk = chunk

    # -- vocabulary ------------------------------------------------------
    @property
    def vocab(self):
        return [ACTION_TOKEN_FMT.format(i) for i in range(len(self.codebook))]

    def ids_to_text(self, ids):
        return "".join(ACTION_TOKEN_FMT.format(int(i)) for i in ids)

    def text_to_ids(self, text):
        import re
        return [int(m) for m in re.findall(r"<action_(\d+)>", text)]

    # -- encode / decode -------------------------------------------------
    def nearest(self, segments):
        """(N, 3) motion segments -> (N,) token ids, (N,) distances."""
        segments = np.atleast_2d(np.asarray(segments, dtype=float))
        flat, _ = _contour_points(segments, self.contour)
        ids = np.empty(len(flat), dtype=np.int64)
        dists = np.empty(len(flat), dtype=float)
        for s in range(0, len(flat), self.chunk):
            block = flat[s:s + self.chunk]
            d = np.linalg.norm(block[:, None, :] - self._flat_cb[None], axis=-1)
            ids[s:s + len(block)] = d.argmin(1)
            dists[s:s + len(block)] = d.min(1)
        return ids, dists

    def encode(self, traj, mode="anchored"):
        """Absolute trajectory (T+1, 3) starting at the current pose -> ids.

        mode="open_loop" maps each segment relative to the *ground-truth* pose
        that precedes it. Because decoding composes the chosen tokens from the
        start pose, any heading error at step t rotates every later waypoint,
        so quantization error compounds and the 5 s FDE ends up an order of
        magnitude above the per-segment error.

        mode="anchored" (default) instead measures each segment relative to the
        pose actually reached so far, so each token corrects the drift left by
        its predecessors. It is the same nearest-neighbour lookup in the same
        codebook, just anchored correctly, and it is what makes the decoded
        trajectory track the ground truth over the full horizon.
        """
        traj = np.asarray(traj, dtype=float)
        if mode == "open_loop":
            return self.nearest(to_deltas(traj))[0]
        if mode != "anchored":
            raise ValueError(f"unknown encode mode {mode!r}")

        ids = np.empty(len(traj) - 1, dtype=np.int64)
        cur = traj[0].copy()
        for t in range(len(traj) - 1):
            delta = relative(cur, traj[t + 1])
            i = self.nearest(delta[None])[0][0]
            ids[t] = i
            cur = compose(cur, self.codebook.actions[i])
        return ids

    def decode(self, ids, start=(0.0, 0.0, 0.0)):
        """Token ids -> reconstructed absolute trajectory (T+1, 3)."""
        deltas = self.codebook.actions[np.asarray(ids, dtype=int)]
        return from_deltas(deltas, start)

    # -- evaluation ------------------------------------------------------
    def reconstruction_error(self, traj, mode="anchored"):
        """ADE / FDE (metres) and heading error (deg) of encode->decode."""
        traj = np.asarray(traj, dtype=float)
        rec = self.decode(self.encode(traj, mode=mode), start=traj[0])
        pos_err = np.linalg.norm(rec[1:, :2] - traj[1:, :2], axis=-1)
        head_err = np.abs(np.rad2deg(
            (rec[1:, 2] - traj[1:, 2] + np.pi) % (2 * np.pi) - np.pi))
        return dict(ade=float(pos_err.mean()), fde=float(pos_err[-1]),
                    heading_mae_deg=float(head_err.mean()))


def evaluate_tokenizer(tok: ActionTokenizer, trajectories, n_tokens=N_ACTION_TOKENS,
                       mode="anchored"):
    """Aggregate reconstruction accuracy over many trajectories.

    Mirrors the paper's Table 4 (tokenization accuracy vs codebook size).
    """
    ade, fde, hd, used = [], [], [], []
    for traj in trajectories:
        traj = np.asarray(traj, dtype=float)[: n_tokens + 1]
        if len(traj) < 2:
            continue
        ids = tok.encode(traj, mode=mode)
        used.extend(ids.tolist())
        e = tok.reconstruction_error(traj, mode=mode)
        ade.append(e["ade"]); fde.append(e["fde"]); hd.append(e["heading_mae_deg"])
    usage = len(set(used)) / max(len(tok.codebook), 1)
    return dict(ade=float(np.mean(ade)), fde=float(np.mean(fde)),
                heading_mae_deg=float(np.mean(hd)),
                codebook_usage=float(usage), n_traj=len(ade),
                token_ids=np.asarray(used))
