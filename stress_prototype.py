"""
PPG stress prototype: contrastive encoder + kNN head, streaming inference.

  python stress_prototype.py build  --ckpt models/model_1400.pt --subjects 2 3 4 ...
  python stress_prototype.py replay --ckpt models/model_1400.pt --subject 5
"""
import argparse
import collections
import os
import pickle

import numpy as np
import torch
from scipy.signal import resample_poly
from sklearn.neighbors import KNeighborsClassifier

FS, WIN_S, HOP_S = 64, 60, 15
WIN, HOP = FS * WIN_S, FS * HOP_S
SEG_DIR = f"simCLR_study/saved_data/window_{WIN_S}_hop_{HOP_S}/"
WESAD_DIR = "old_code/paper_codes/dataset/WESAD/"


# ----------------------------------------------------------------- model
class PPGModel(torch.nn.Module):
    def __init__(self, in_channels=1, embedding_dim=256, projection_dim=128):
        super().__init__()
        cfg = [(in_channels, 32, 15), (32, 64, 9), (64, 128, 7), (128, 256, 5)]
        for i, (ci, co, k) in enumerate(cfg, 1):
            setattr(self, f"conv_{i}", torch.nn.Conv1d(ci, co, k, stride=2, padding=k // 2))
            setattr(self, f"bn_{i}", torch.nn.BatchNorm1d(co))
        self.act_func = torch.nn.ReLU()
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.embedding_enc = torch.nn.Linear(256, embedding_dim)
        self.projection_enc = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, embedding_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(embedding_dim, projection_dim),
        )

    def encode(self, x):
        for i in range(1, 5):
            x = self.act_func(getattr(self, f"bn_{i}")(getattr(self, f"conv_{i}")(x)))
        return self.embedding_enc(self.global_pool(x).squeeze(-1))


def load_model(ckpt, device):
    m = PPGModel()
    m.load_state_dict(torch.load(ckpt, map_location=device))
    return m.to(device).eval()


# ----------------------------------------------------------- preprocessing
def zscore(w):
    # Same per-window normalisation as in training (the notebook's inference
    # cell skips this; the training loop does it).
    return (w - w.mean(-1, keepdims=True)) / (w.std(-1, keepdims=True) + 1e-8)


def to_64hz(x, fs_in, axis=-1):
    if fs_in == FS:
        return x
    from math import gcd
    g = gcd(int(fs_in), FS)
    return resample_poly(x, FS // g, int(fs_in) // g, axis=axis)


def window_ok(w, max_nan_frac=0.05):
    """Cheap quality gate: reject flat / NaN-heavy windows instead of predicting."""
    return np.isnan(w).mean() <= max_nan_frac and np.nanstd(w) > 1e-6


# ------------------------------------------------------------- predictor
class StressPredictor:
    def __init__(self, ckpt, bank_path, k=5, smooth=5, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(ckpt, self.dev)
        bank = np.load(bank_path)
        self.knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(bank["emb"], bank["y"])
        self.buf = collections.deque(maxlen=WIN)
        self.since_last = 0
        self.history = collections.deque(maxlen=smooth)

    @torch.no_grad()
    def embed(self, windows):  # (N, WIN) float32 -> (N, 256)
        x = torch.from_numpy(zscore(np.asarray(windows, np.float32))).unsqueeze(1).to(self.dev)
        return self.model.encode(x).cpu().numpy()

    def predict_window(self, w):
        """P(stress) for one 60 s window, or None if the window is unusable."""
        if len(w) != WIN or not window_ok(w):
            return None
        w = np.nan_to_num(w - np.nanmean(w))
        return float(self.knn.predict_proba(self.embed(w[None]))[0, 1])

    def push(self, chunk):
        """Feed raw 64 Hz samples (any chunk size). Returns a result dict every HOP_S seconds."""
        self.buf.extend(np.asarray(chunk, np.float32))
        self.since_last += len(chunk)
        if len(self.buf) < WIN or self.since_last < HOP:
            return None
        self.since_last = 0
        return self.update_window(np.array(self.buf))

    def update_window(self, w, reset=False):
        """Predict one full 60 s window and update the smoothing history."""
        if reset:
            self.history.clear()
        p = self.predict_window(w)
        if p is None:
            return {"p_raw": None, "p": None, "label": "no-signal"}
        self.history.append(p)
        p_s = float(np.mean(self.history))  # smoothing over the last few windows
        return {"p_raw": p, "p": p_s, "label": "stress" if p_s >= 0.5 else "non-stress"}


# ------------------------------------------------------------ build / replay
def build_bank(ckpt, subjects, out="bank.npz"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, E, Y = load_model(ckpt, dev), [], []
    for s in subjects:
        X = np.load(os.path.join(SEG_DIR, f"subject_{s}_data.npy")).astype(np.float32)
        y = np.load(os.path.join(SEG_DIR, f"subject_{s}_labels.npy"))
        for i in range(0, len(X), 256):
            x = torch.from_numpy(zscore(X[i:i + 256])).unsqueeze(1).to(dev)
            with torch.no_grad():
                E.append(model.encode(x).cpu().numpy())
        Y.append(y)
    np.savez(out, emb=np.concatenate(E), y=np.concatenate(Y).astype(int))
    print("saved", out, np.concatenate(E).shape)


def replay(ckpt, subject, bank="bank.npz", chunk_s=1):
    """Stream a WESAD subject's wrist BVP as if it were live, compare with ground truth."""
    with open(os.path.join(WESAD_DIR, f"S{subject}", f"S{subject}.pkl"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    bvp, lab = d["signal"]["wrist"]["BVP"][:, 0], d["label"]  # 64 Hz, 700 Hz
    pred, n = StressPredictor(ckpt, bank), FS * chunk_s
    for i in range(0, len(bvp) - n, n):
        r = pred.push(bvp[i:i + n])
        if r:
            truth = {1: "non-stress", 2: "stress"}.get(int(lab[int((i - WIN // 2) / FS * 700)]), "other")
            print(f"t={i / FS / 60:6.1f} min  p={r['p'] if r['p'] is None else round(r['p'], 2)}  "
                  f"pred={r['label']:<10} truth={truth}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "replay"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", type=int, nargs="+")
    ap.add_argument("--subject", type=int)
    a = ap.parse_args()
    build_bank(a.ckpt, a.subjects) if a.cmd == "build" else replay(a.ckpt, a.subject)
