"""
GI Endoscopy Classifier - Streamlit app
Run:  streamlit run app.py
Put this file in (or point the sidebar at) the folder you extracted from gi_results.zip.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

st.set_page_config(page_title="GI Endoscopy Classifier", layout="wide")

CLASS_NAMES = ["dyed-lifted-polyps", "dyed-resection-margins", "esophagitis", "normal-cecum",
               "normal-pylorus", "normal-z-line", "polyps", "ulcerative-colitis"]  # alphabetical = training order
K = len(CLASS_NAMES)
pretty = lambda c: c.replace("-", " ").title()
PALETTE = {"main": "#0E7C86", "accent": "#F2A541", "bad": "#C8553D", "grey": "#8AA6B1"}


# ----------------------------------------------------------------------------- helpers
def wilson_ci(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def ece_score(P, y, bins=15):
    conf, pred = P.max(1), P.argmax(1); acc = (pred == y).astype(float)
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return e


def _patch_keras_compat():
    """Let an older Keras ignore config keys that a newer Keras wrote (e.g. quantization_config)."""
    from keras.src.layers.layer import Layer
    if getattr(Layer, "_compat_patched", False):
        return
    orig = Layer.__init__
    def patched(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        orig(self, *args, **kwargs)
    Layer.__init__ = patched
    Layer._compat_patched = True


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    import tensorflow as tf
    _patch_keras_compat()
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        st.error("Could not load the model. Please send the last lines of the error shown below.")
        raise


def model_size(model):
    return int(model.input_shape[1])


def prep(img, size):
    """Same preprocessing as training: bilinear resize, round to 0-255. Normalisation lives inside the model."""
    import tensorflow as tf
    t = tf.image.resize(np.array(img.convert("RGB")), (size, size))
    return tf.cast(tf.round(t), tf.float32).numpy()[None]


def predict(model, x, tta):
    p = model.predict(x, verbose=0)
    if tta:
        p = (p + model.predict(np.ascontiguousarray(x[:, :, ::-1, :]), verbose=0)) / 2
    return p


def gradcam(model, x, cls, img):
    """Grad-CAM on the backbone's final feature map; returns an RGB overlay."""
    import tensorflow as tf
    import matplotlib
    L = model.layers
    bi = next(i for i, l in enumerate(L) if isinstance(l, tf.keras.Model) and len(l.layers) > 50)
    xin = tf.convert_to_tensor(x)
    for l in L[1:bi]:
        if not l.name.startswith("augment"):
            xin = l(xin)
    feat = tf.cast(L[bi](xin, training=False), tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(feat); h = feat
        for l in L[bi + 1:]:
            h = l(h, training=False)
        score = tf.cast(h, tf.float32)[:, cls]
    g = tape.gradient(score, feat)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * feat, -1))[0].numpy()
    cam = cam / (cam.max() + 1e-8)
    rgb = img.convert("RGB"); W, H = rgb.size
    cam_img = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((W, H), Image.BILINEAR)) / 255.0
    heat = (matplotlib.colormaps["jet"](cam_img)[..., :3] * 255).astype(np.uint8)
    over = (0.55 * np.array(rgb) + 0.45 * heat).astype(np.uint8)
    return over, (cam_img * 255).astype(np.uint8)


def find_assets(folder: Path):
    root = folder / "gi_results" if (folder / "gi_results").is_dir() else folder
    return root, {
        "models": sorted(root.glob("*.keras")),
        "results": root / "results.json", "split": root / "split.csv", "probs": root / "test_probs_ensemble.npy",
        "logs": sorted(root.glob("*_log.csv")), "pngs": sorted(root.glob("*.png")),
    }


# ----------------------------------------------------------------------------- shared metric dashboard
def render_metrics(y, P, key):
    pred = P.argmax(1); n = len(y); k = int((pred == y).sum())
    from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_curve, roc_auc_score, confusion_matrix
    lo, hi = wilson_ci(k, n); f1m = f1_score(y, pred, average="macro"); e = ece_score(P, y)
    try:
        auc = roc_auc_score(np.eye(K)[y], P, average="macro", multi_class="ovr")
    except Exception:
        auc = float("nan")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{k / n:.2%}"); c[1].metric("95% CI (Wilson)", f"{lo:.1%} - {hi:.1%}")
    c[2].metric("Macro-F1", f"{f1m:.3f}"); c[3].metric("Calibration error (ECE)", f"{e:.3f}")
    c[4].metric("Macro AUC (OvR)", "n/a" if np.isnan(auc) else f"{auc:.3f}")
    st.caption(f"n = {n} images")

    a, b = st.columns(2)
    with a:
        norm = st.toggle("Normalise rows", value=False, key=key + "_norm")
        cm = confusion_matrix(y, pred, labels=range(K)).astype(float)
        if norm:
            cm = cm / cm.sum(1, keepdims=True)
        fig = px.imshow(cm, x=[pretty(c) for c in CLASS_NAMES], y=[pretty(c) for c in CLASS_NAMES], text_auto=".2f" if norm else True,
                        color_continuous_scale="Teal", aspect="auto", labels=dict(x="Predicted", y="True", color=""))
        fig.update_layout(title="Confusion matrix", height=480, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with b:
        p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
        d = pd.DataFrame({"Class": [pretty(c) for c in CLASS_NAMES], "Precision": p, "Recall": r, "F1": f})
        fig = px.bar(d.melt("Class", var_name="Metric", value_name="Score"), x="Class", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=[PALETTE["grey"], PALETTE["accent"], PALETTE["main"]], range_y=[0, 1.02])
        fig.update_layout(title="Per-class precision / recall / F1", height=480, margin=dict(t=50, b=10), xaxis_tickangle=-30)
        st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:  # reliability diagram
        conf, ok = P.max(1), (pred == y).astype(float); edges = np.linspace(0, 1, 11); xs, ys, cnt = [], [], []
        for lo_, hi_ in zip(edges[:-1], edges[1:]):
            m = (conf > lo_) & (conf <= hi_)
            if m.any():
                xs.append(conf[m].mean()); ys.append(ok[m].mean()); cnt.append(int(m.sum()))
        fig = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=PALETTE["grey"]), name="Perfect calibration"),
                         go.Scatter(x=xs, y=ys, mode="lines+markers", marker=dict(size=[6 + c_ ** 0.5 for c_ in cnt], color=PALETTE["main"]),
                                    line=dict(color=PALETTE["main"]), name="Model", text=[f"n={c_}" for c_ in cnt])])
        fig.update_layout(title=f"Reliability diagram (ECE {e:.3f})", xaxis_title="Mean predicted confidence", yaxis_title="Observed accuracy",
                          height=400, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
    with b:  # confidence histogram
        d = pd.DataFrame({"Confidence": conf, "Outcome": np.where(ok == 1, "Correct", "Incorrect")})
        fig = px.histogram(d, x="Confidence", color="Outcome", nbins=20, barmode="overlay", opacity=0.75,
                           color_discrete_map={"Correct": PALETTE["main"], "Incorrect": PALETTE["bad"]})
        fig.update_layout(title="Confidence of correct vs incorrect predictions", height=400, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    for i, cn in enumerate(CLASS_NAMES):
        if (y == i).any() and (y != i).any():
            fpr, tpr, _ = roc_curve(y == i, P[:, i]); fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=pretty(cn)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color=PALETTE["grey"]), name="Chance"))
    fig.update_layout(title="One-vs-rest ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420, margin=dict(t=50, b=10))
    st.plotly_chart(fig, width="stretch")


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("GI Endoscopy Classifier")
folder = Path(st.sidebar.text_input("Folder with the extracted gi_results files", value=str(Path(__file__).parent))).expanduser()
root, A = find_assets(folder)
finals = [m for m in A["models"] if m.name.endswith("_final.keras")] or A["models"]
chosen = st.sidebar.multiselect("Model(s) - several are averaged as an ensemble", [m.name for m in A["models"]], default=[m.name for m in finals])
tta = st.sidebar.toggle("Flip test-time augmentation", value=True)
thr = st.sidebar.slider("Review threshold (confidence)", 0.30, 0.99, 0.70, 0.01,
                        help="Predictions below this confidence are flagged for human review.")
show_cam = st.sidebar.toggle("Show Grad-CAM heat-map", value=True)
st.sidebar.caption(f"Reading from: {root}")
st.sidebar.warning("Research prototype - not a medical device.")
try:
    import tensorflow as _tf, keras as _k
    st.sidebar.caption(f"TensorFlow {_tf.__version__} | Keras {_k.__version__}")
except Exception:
    pass

tab_pred, tab_perf, tab_eval, tab_about = st.tabs(["Predict", "Model performance", "Evaluate a labelled folder", "About"])

# ----------------------------------------------------------------------------- predict
with tab_pred:
    st.header("Classify an endoscopic image")
    if not chosen:
        st.error(f"No .keras model found in {root}. Point the sidebar at the extracted gi_results folder.")
    files = st.file_uploader("Upload one or more images", type=["jpg", "jpeg", "png", "bmp"], accept_multiple_files=True)
    truth = st.selectbox("True class (optional, to mark the prediction right or wrong)", ["(unknown)"] + CLASS_NAMES)
    if chosen and files:
        models = [(n, load_model(str(root / n))) for n in chosen]
        for f in files:
            img = Image.open(f).convert("RGB")
            per_model = {n: predict(m, prep(img, model_size(m)), tta)[0] for n, m in models}
            P = np.mean(list(per_model.values()), axis=0)
            order = np.argsort(-P); top, second = int(order[0]), int(order[1])
            conf = float(P[top]); margin = float(P[top] - P[second])
            ent = float(-(P * np.log(P + 1e-12)).sum() / np.log(K))
            st.divider(); st.subheader(f.name)
            c1, c2, c3 = st.columns([1.1, 1.1, 1.2])
            with c1:
                st.image(img, caption="Input", width="stretch")
            with c2:
                if show_cam:
                    try:
                        name0, m0 = models[0]
                        over, _ = gradcam(m0, prep(img, model_size(m0)), top, img)
                        st.image(over, caption=f"Grad-CAM for '{pretty(CLASS_NAMES[top])}' ({name0})", width="stretch")
                    except Exception as ex:
                        st.info(f"Grad-CAM unavailable for this model ({type(ex).__name__}).")
            with c3:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=conf * 100, number={"suffix": "%", "valueformat": ".1f"},
                                             title={"text": "Prediction confidence"},
                                             gauge={"axis": {"range": [0, 100]}, "bar": {"color": PALETTE["main"] if conf >= thr else PALETTE["bad"]},
                                                    "steps": [{"range": [0, thr * 100], "color": "#F6D9D2"}, {"range": [thr * 100, 100], "color": "#D8EEEF"}],
                                                    "threshold": {"line": {"color": "#222", "width": 3}, "value": thr * 100}}))
                fig.update_layout(height=250, margin=dict(t=60, b=0, l=20, r=20))
                st.plotly_chart(fig, width="stretch")
            st.markdown(f"### Predicted: **{pretty(CLASS_NAMES[top])}**")
            if truth != "(unknown)":
                (st.success if CLASS_NAMES[top] == truth else st.error)(
                    f"{'Correct' if CLASS_NAMES[top] == truth else 'Incorrect'} - true class: {pretty(truth)}")
            (st.success if conf >= thr else st.warning)(
                "Confidence above the review threshold." if conf >= thr else
                "Low confidence - flag for human review (also possible if the image is not an endoscopic frame of these 8 classes).")
            m = st.columns(4)
            m[0].metric("Top-1 probability", f"{conf:.1%}"); m[1].metric("Margin over 2nd class", f"{margin:.1%}")
            m[2].metric("Normalised entropy", f"{ent:.2f}", help="0 = certain, 1 = uniform guess"); m[3].metric("2nd choice", pretty(CLASS_NAMES[second]))

            a, b = st.columns(2)
            with a:
                d = pd.DataFrame({"Class": [pretty(c) for c in CLASS_NAMES], "Probability": P}).sort_values("Probability")
                fig = px.bar(d, x="Probability", y="Class", orientation="h", text=d["Probability"].map("{:.1%}".format), range_x=[0, 1.05])
                fig.update_traces(marker_color=[PALETTE["main"] if c == pretty(CLASS_NAMES[top]) else PALETTE["grey"] for c in d["Class"]], textposition="outside")
                fig.update_layout(title="Class probabilities", height=380, margin=dict(t=50, b=10))
                st.plotly_chart(fig, width="stretch")
            with b:
                if len(per_model) > 1:
                    d = pd.DataFrame([{"Model": n.replace("_final.keras", ""), "Class": pretty(c), "Probability": p[i]}
                                      for n, p in per_model.items() for i, c in enumerate(CLASS_NAMES)])
                    fig = px.bar(d, x="Class", y="Probability", color="Model", barmode="group", color_discrete_sequence=[PALETTE["main"], PALETTE["accent"], PALETTE["grey"]])
                    fig.update_layout(title="Per-model agreement", height=380, margin=dict(t=50, b=10), xaxis_tickangle=-30)
                    st.plotly_chart(fig, width="stretch")
                else:
                    th = np.r_[P, P[:1]]; cats = [pretty(c) for c in CLASS_NAMES] + [pretty(CLASS_NAMES[0])]
                    fig = go.Figure(go.Scatterpolar(r=th, theta=cats, fill="toself", line_color=PALETTE["main"]))
                    fig.update_layout(title="Probability profile", height=380, polar=dict(radialaxis=dict(range=[0, 1])), margin=dict(t=50, b=10))
                    st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame({"Rank": range(1, 4), "Class": [pretty(CLASS_NAMES[i]) for i in order[:3]],
                                       "Probability": [f"{P[i]:.2%}" for i in order[:3]]}), hide_index=True)
    elif chosen:
        st.info("Upload an image to see the prediction, confidence gauge, Grad-CAM and probability charts.")

# ----------------------------------------------------------------------------- performance
with tab_perf:
    st.header("Held-out test performance")
    st.caption("Stratified, near-duplicate-aware 70/15/15 split; the test set was evaluated once.")
    if A["results"].exists():
        res = json.loads(A["results"].read_text())
        d = pd.DataFrame([{"Model": r["backbone"], "Seed": r["seed"], "Accuracy (plain)": r["plain"]["acc"], "Accuracy (TTA)": r["tta"]["acc"],
                           "CI low": r["tta"]["ci95"][0], "CI high": r["tta"]["ci95"][1], "Macro-F1": r["tta"]["macro_f1"], "ECE": r["tta"]["ece"]} for r in res])
        best = d.sort_values("Accuracy (TTA)", ascending=False).iloc[0]
        c = st.columns(4)
        c[0].metric("Best model", f"{best['Model']}"); c[1].metric("Accuracy (TTA)", f"{best['Accuracy (TTA)']:.2%}")
        c[2].metric("Macro-F1", f"{best['Macro-F1']:.3f}"); c[3].metric("ECE", f"{best['ECE']:.3f}")
        fig = go.Figure(go.Bar(x=[f"{m} (seed {s})" for m, s in zip(d["Model"], d["Seed"])], y=d["Accuracy (TTA)"], marker_color=PALETTE["main"],
                               error_y=dict(type="data", symmetric=False, array=d["CI high"] - d["Accuracy (TTA)"], arrayminus=d["Accuracy (TTA)"] - d["CI low"]),
                               text=d["Accuracy (TTA)"].map("{:.2%}".format), textposition="outside"))
        fig.update_layout(title="Test accuracy with 95% Wilson intervals", yaxis=dict(range=[0.8, 1.0], tickformat=".0%"), height=380, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(d.style.format({c_: "{:.4f}" for c_ in d.columns if c_ not in ("Model", "Seed")}), hide_index=True)
    else:
        st.info("results.json not found in the folder.")

    if A["probs"].exists() and A["split"].exists():
        st.subheader("Ensemble of all trained models (saved test probabilities)")
        sp = pd.read_csv(A["split"]); y = sp[sp.split == "test"]["label"].values; P = np.load(A["probs"])
        if len(y) == len(P):
            render_metrics(y, P, "ens")
        else:
            st.warning("test_probs_ensemble.npy and split.csv do not match in length.")
    else:
        st.caption("No test_probs_ensemble.npy found (it exists only when two or more models were trained). "
                   "Use the 'Evaluate a labelled folder' tab to compute the full diagrams on your own images.")

    if A["logs"]:
        st.subheader("Training curves")
        lg = st.selectbox("Log file", [p.name for p in A["logs"]]); ld = pd.read_csv(root / lg); ld["epoch"] = ld.index + 1
        a, b = st.columns(2)
        for col, (m1, m2, t) in zip((a, b), (("loss", "val_loss", "Loss"), ("accuracy", "val_accuracy", "Accuracy"))):
            if m1 in ld:
                fig = px.line(ld, x="epoch", y=[m1, m2], title=t, color_discrete_sequence=[PALETTE["main"], PALETTE["accent"]])
                fig.update_layout(height=340, margin=dict(t=50, b=10)); col.plotly_chart(fig, width="stretch")
    if A["pngs"]:
        with st.expander("Saved figures from training (confusion matrices, curves)"):
            for p in A["pngs"]:
                st.image(str(p), caption=p.name, width="stretch")

# ----------------------------------------------------------------------------- evaluate folder
with tab_eval:
    st.header("Evaluate on a labelled folder")
    st.caption("Folder layout: one sub-folder per class, named exactly as the 8 class names. Metrics and diagrams are computed live.")
    ef = st.text_input("Path to the labelled image folder", value="")
    mx = st.slider("Max images per class", 10, 500, 100, 10)
    if st.button("Run evaluation") and ef and chosen:
        base = Path(ef).expanduser(); items = []
        for i, cn in enumerate(CLASS_NAMES):
            d_ = base / cn
            if d_.is_dir():
                fs = sorted(p for p in d_.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))[:mx]
                items += [(p, i) for p in fs]
        if not items:
            st.error("No images found - check the folder and class sub-folder names.")
        else:
            models = [load_model(str(root / n)) for n in chosen]; probs, ys = [], []; bar = st.progress(0.0)
            for j, (p, yi) in enumerate(items):
                im = Image.open(p)
                probs.append(np.mean([predict(m, prep(im, model_size(m)), tta)[0] for m in models], axis=0)); ys.append(yi)
                if j % 10 == 0:
                    bar.progress((j + 1) / len(items))
            bar.empty(); render_metrics(np.array(ys), np.array(probs), "folder")

# ----------------------------------------------------------------------------- about
with tab_about:
    st.header("About")
    st.markdown(
        "- **Model:** ImageNet-pretrained EfficientNetB3 (and optionally ResNet101V2) fine-tuned in two stages on the Kvasir dataset (8 classes).\n"
        "- **Evaluation protocol:** near-duplicate frames grouped by perceptual hashing; stratified, group-aware 70/15/15 split; test set used once; Wilson 95% intervals; expected calibration error.\n"
        "- **Confidence:** softmax probability (optionally averaged over a horizontal flip and over models). The review threshold in the sidebar flags low-confidence cases.\n"
        "- **Closed set:** the model only knows these 8 classes. A non-endoscopic or unusual image will still receive a label, usually with low confidence or high entropy.\n"
        "- **Grad-CAM:** highlights image regions that most increased the predicted class score; it is an aid to inspection, not proof of causality.\n"
        "- **Not a medical device.** For research and demonstration only.")