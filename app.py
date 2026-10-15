"""
streamlit run app.py
Needs stress_prototype.py next to it, plus a checkpoint and bank.npz (see `build`).

Inputs:
  * WESAD subject (raw BVP + 700 Hz labels), or
  * uploaded signal: 1-D continuous stream, OR 2-D (n_windows, 60*fs) segmented windows
    (e.g. subject_16_data.npy). Labels: per-sample (1-D stream) or per-window (2-D).
"""
import io
import os
import pickle
import time

import numpy as np
import pandas as pd
import streamlit as st

import stress_prototype as sp

st.set_page_config(page_title="PPG stress prototype", layout="wide")
st.title("PPG stress detection – live replay")


def load_arr(f):
    return np.load(io.BytesIO(f.read()), allow_pickle=True) if f.name.endswith(".npy") else pd.read_csv(f).to_numpy()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    ckpt = st.text_input("Checkpoint", "simCLR_study/models/model_1400.pt")
    bank = st.text_input("Bank (.npz)", "bank.npz")
    k = st.slider("kNN k", 1, 25, 5)
    smooth = st.slider("Smoothing (windows)", 1, 10, 5)
    thr = st.slider("Stress threshold", 0.1, 0.9, 0.5, 0.05)
    speed = st.slider("Replay speed (x real time)", 1, 300, 60)
    src = st.radio("Source", ["WESAD subject", "Upload file"])

    signal, windows, fs_in, labels = None, None, sp.FS, None
    if src == "WESAD subject":
        subj = st.number_input("Subject id", 2, 17, 5)
        path = os.path.join(sp.WESAD_DIR, f"S{subj}", f"S{subj}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                d = pickle.load(f, encoding="latin1")
            signal = d["signal"]["wrist"]["BVP"][:, 0].astype(np.float32)
            labels = d["label"]  # 700 Hz
        else:
            st.error(f"Not found: {path}")
    else:
        up = st.file_uploader("PPG signal (.npy / .csv)", type=["csv", "npy"])
        fs_in = st.number_input("Sampling rate of file (Hz)", 16, 512, 64)
        if up is not None:
            arr = np.asarray(load_arr(up), np.float32)
            if arr.ndim == 2 and arr.shape[1] > 1 and arr.shape[0] > 1 and arr.shape[1] >= sp.WIN_S * 16:
                windows = sp.to_64hz(arr, fs_in, axis=1).astype(np.float32)  # (N, WIN)
                st.caption(f"Segmented input: {windows.shape[0]} windows × {windows.shape[1]} samples")
            else:
                signal = sp.to_64hz(arr.ravel(), fs_in).astype(np.float32)

    st.markdown("**Ground truth (optional)**")
    lab_up = st.file_uploader("Labels file (.npy / .csv)", type=["csv", "npy"], key="lab")
    seg_mode = windows is not None
    lab_fs = st.number_input("Labels sampling rate (Hz)", 0.01, 1000.0,
                             700.0 if src == "WESAD subject" else float(fs_in), disabled=seg_mode)
    v_stress = st.number_input("Label value = stress", value=2 if src == "WESAD subject" else 1)
    v_non = st.number_input("Label value = non-stress", value=1 if src == "WESAD subject" else 0)
    if lab_up is not None:
        labels = np.asarray(load_arr(lab_up)).ravel()
        if seg_mode and len(labels) != len(windows):
            st.error(f"{len(labels)} labels but {len(windows)} windows – must match (one label per window).")
            labels = None

    start = st.button("Start replay", type="primary", disabled=(signal is None and windows is None))

if signal is None and windows is None:
    st.info("Pick a source in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ layout
c1, c2, c3, c4 = st.columns(4)
m_label, m_p, m_t, m_acc = c1.empty(), c2.empty(), c3.empty(), c4.empty()
st.subheader("P(stress) over time")
ph_p = st.empty()
st.subheader("Last 10 s of PPG" if not seg_mode else "Current 60 s window")
ph_sig = st.empty()
progress = st.progress(0.0)


def to_truth(v):
    return 1.0 if v == v_stress else 0.0 if v == v_non else None


def show(rows, r, thr):
    df = pd.DataFrame(rows).set_index("min")
    ph_p.line_chart(df.dropna(axis=1, how="all"), height=250)
    if r["p"] is None:
        m_label.metric("State", "no signal")
    else:
        m_label.metric("State", "STRESS" if r["p"] >= thr else "non-stress")
        m_p.metric("P(stress)", f"{r['p']:.2f}")
    if "truth" in df and df["truth"].notna().any():
        d = df.dropna(subset=["P(stress)", "truth"])
        if len(d):
            m_acc.metric("Running accuracy", f"{((d['P(stress)'] >= thr) == (d['truth'] == 1)).mean():.1%}")


# ------------------------------------------------------------------- replay
if start:
    if not (os.path.exists(ckpt) and os.path.exists(bank)):
        st.error("Checkpoint or bank file not found.")
        st.stop()
    pred, rows = sp.StressPredictor(ckpt, bank, k=k, smooth=smooth), []

    if seg_mode:
        # windows are 15 s apart; reset smoothing if consecutive windows aren't contiguous
        H = sp.HOP
        for i, w in enumerate(windows):
            gap = i > 0 and not np.allclose(windows[i - 1][H:], w[:-H], atol=1e-4)
            r = pred.update_window(w, reset=gap)
            t_min = (i * sp.HOP_S + sp.WIN_S) / 60
            rows.append({"min": t_min, "P(stress)": r["p"],
                         "truth": to_truth(labels[i]) if labels is not None else None})
            m_t.metric("Window", f"{i + 1}/{len(windows)}")
            show(rows, r, thr)
            ph_sig.line_chart(w, height=150)
            progress.progress((i + 1) / len(windows))
            time.sleep(sp.HOP_S / speed)
    else:
        n = sp.FS  # 1 s chunks
        for i in range(0, len(signal) - n, n):
            r = pred.push(signal[i:i + n])
            t_min = (i + n) / sp.FS / 60
            if r is not None:
                truth = None
                if labels is not None:
                    t_c = ((i + n) - sp.WIN // 2) / sp.FS  # window centre, seconds
                    truth = to_truth(labels[min(max(int(t_c * lab_fs), 0), len(labels) - 1)])
                rows.append({"min": t_min, "P(stress)": r["p"], "truth": truth})
                show(rows, r, thr)
            m_t.metric("Elapsed (recording)", f"{t_min:.1f} min")
            if i % (sp.FS * 5) == 0:
                ph_sig.line_chart(signal[max(0, i + n - 10 * sp.FS):i + n], height=150)
            progress.progress(min((i + n) / len(signal), 1.0))
            time.sleep(1 / speed)
    st.success("Replay finished.")
