"""Simple confidence dashboard. Run:  streamlit run test_dashboard.py"""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

CLASS_NAMES = ["dyed-lifted-polyps", "dyed-resection-margins", "esophagitis", "normal-cecum",
               "normal-pylorus", "normal-z-line", "polyps", "ulcerative-colitis"]

st.set_page_config(page_title="GI Prediction Test", layout="wide")
st.title("GI Endoscopy: Prediction Confidence Test")


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    import tensorflow as tf
    from keras.src.layers.layer import Layer
    if not getattr(Layer, "_compat_patched", False):   # older Keras vs newer-saved model
        orig = Layer.__init__
        def patched(self, *a, **kw):
            kw.pop("quantization_config", None)
            orig(self, *a, **kw)
        Layer.__init__ = patched
        Layer._compat_patched = True
    return tf.keras.models.load_model(path, compile=False)


here = Path(__file__).parent
models = sorted(list(here.glob("*_final.keras")) + list((here / "gi_results").glob("*_final.keras")))
if not models:
    st.error("No *_final.keras model found next to this file or in a gi_results folder.")
    st.stop()

with st.sidebar:
    model_path = st.selectbox("Model", models, format_func=lambda p: p.name)
    threshold = st.slider("Review threshold (confidence)", 0.50, 0.99, 0.90, 0.01)

file = st.file_uploader("Select an endoscopy image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
if file is None:
    st.info("Choose an image to see the prediction.")
    st.stop()

model = load_model(str(model_path))
size = int(model.input_shape[1])
img = Image.open(file).convert("RGB")
x = np.asarray(img.resize((size, size)), dtype="float32")[None]   # raw 0-255
probs = model.predict(x, verbose=0)[0]
order = np.argsort(probs)[::-1]
best, conf = int(order[0]), float(probs[order[0]])

left, right = st.columns([1, 2])
with left:
    st.image(img, caption=file.name, width="stretch")
with right:
    st.subheader(f"Prediction: {CLASS_NAMES[best]}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{conf*100:.2f}%")
    c2.metric("Runner-up", f"{CLASS_NAMES[int(order[1])]}", f"{probs[order[1]]*100:.2f}%", delta_color="off")
    c3.metric("Margin", f"{(probs[order[0]]-probs[order[1]])*100:.2f} pts")
    if conf >= threshold:
        st.success("High confidence")
    elif conf >= 0.70:
        st.warning("Medium confidence: double-check")
    else:
        st.error("Low confidence: needs human review")

g, b = st.columns(2)
with g:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf * 100, number={"suffix": "%"},
        title={"text": "Model confidence"},
        gauge={"axis": {"range": [0, 100]}, "threshold": {"value": threshold * 100, "line": {"width": 4}},
               "steps": [{"range": [0, 70], "color": "#f4c7c3"}, {"range": [70, threshold * 100], "color": "#fce8b2"},
                         {"range": [threshold * 100, 100], "color": "#b7e1cd"}]}))
    st.plotly_chart(fig, width="stretch")
with b:
    idx = np.argsort(probs)
    fig = go.Figure(go.Bar(x=probs[idx] * 100, y=[CLASS_NAMES[i] for i in idx], orientation="h",
                           marker_color=["#1a73e8" if i == best else "#9aa0a6" for i in idx],
                           text=[f"{probs[i]*100:.1f}%" for i in idx], textposition="outside"))
    fig.update_layout(title="Probability per class", xaxis_title="%", xaxis_range=[0, 105])
    st.plotly_chart(fig, width="stretch")