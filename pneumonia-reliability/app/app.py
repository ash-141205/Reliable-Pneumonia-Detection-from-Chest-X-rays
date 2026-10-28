"""Professional Streamlit interface for the trained pneumonia research model.

Run from the project directory with: streamlit run app/app.py
This interface is for research demonstration only, not medical use.
"""
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from pneumreliab.data import get_transform
from pneumreliab.evaluate import load_checkpoint
from pneumreliab.explain import gradcam, overlay
from pneumreliab.utils import get_device


st.set_page_config(
    page_title="Pneumonia Model Lab",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { --ink:#172b3a; --muted:#647887; --teal:#087e83; --line:#dce7e9; --wash:#f3f8f8; }
      .stApp, [data-testid="stAppViewContainer"] { background: linear-gradient(180deg,#f3f8f8 0%,#ffffff 34rem) !important; color:#172b3a !important; }
      .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 3rem; }
      html, body, [class*="css"] { font-family:'Segoe UI',Arial,sans-serif; }
      [data-testid="stHeadingWithActionElements"] h1,
      [data-testid="stHeadingWithActionElements"] h2,
      [data-testid="stHeadingWithActionElements"] h3,
      [data-testid="stMarkdownContainer"] h1,
      [data-testid="stMarkdownContainer"] h2,
      [data-testid="stMarkdownContainer"] h3 { font-family:'Segoe UI',Arial,sans-serif; color:#172b3a !important; letter-spacing:-.025em; }
      [data-testid="stMarkdownContainer"] p,
      [data-testid="stMarkdownContainer"] li,
      [data-testid="stMarkdownContainer"] strong { color:#172b3a; }
      [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color:#647887 !important; }
      [data-testid="stSidebar"] label, [data-testid="stSidebar"] label p { color:#172b3a !important; }
      .hero-kicker { color:var(--teal); font-size:.78rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.55rem; }
      .hero-sub { color:var(--muted); font-size:1.05rem; max-width:760px; line-height:1.65; }
      .notice { background:#fff8e8; border:1px solid #f1d99b; border-radius:12px; padding:14px 18px; color:#74531a; margin:1.25rem 0 1.5rem; }
      .section-label { color:var(--teal); font-size:.76rem; text-transform:uppercase; letter-spacing:.12em; font-weight:700; margin:1.25rem 0 .35rem; }
      .result-card { background:#fff; border:1px solid var(--line); border-radius:16px; padding:22px 24px; min-height:140px; box-shadow:0 5px 20px rgba(23,43,58,.04); }
      .result-label { color:var(--muted); font-size:.82rem; font-weight:600; margin-bottom:.55rem; }
      .result-value { color:var(--ink); font-family:'Segoe UI',Arial,sans-serif; font-size:1.75rem; line-height:1.2; font-weight:800; }
      .result-help { color:var(--muted); font-size:.86rem; margin-top:.55rem; }
      .panel-title { font-family:'Segoe UI',Arial,sans-serif; font-weight:700; font-size:1.05rem; color:var(--ink); margin-bottom:.2rem; }
      .panel-help { color:var(--muted); font-size:.86rem; margin-bottom:.7rem; }
      div[data-testid="stImage"] { background:#fff; border:1px solid var(--line); border-radius:14px; padding:8px; }
      [data-testid="stSidebar"] { background:#f7fafb !important; border-right:1px solid var(--line); }
      [data-testid="stSidebar"] h2 { font-size:1.15rem; }
      div[data-testid="stFileUploader"] { background:#fff; border:1px dashed #a8c5c8; border-radius:14px; padding:8px; }
      .footer-note { color:var(--muted); font-size:.83rem; border-top:1px solid var(--line); padding-top:1rem; margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def read_json(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_resource(show_spinner="Loading the selected model…")
def load_model(checkpoint_path):
    return load_checkpoint(checkpoint_path, get_device())


st.title("Pneumonia Model Lab")
st.markdown(
    '<div class="hero-sub">Explore how an own-trained convolutional neural network scores an uploaded chest X-ray, '
    'and inspect a Grad-CAM visualization of image regions associated with its selected score.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="notice"><strong>Research prototype only.</strong> This model is not a medical device, '
    'has not been clinically validated, and must not be used to diagnose or guide care.</div>',
    unsafe_allow_html=True,
)

checkpoint_paths = sorted(glob.glob(os.path.join(PROJECT_ROOT, "results", "*", "model.pt")))
run_by_label = {
    os.path.basename(os.path.dirname(path)): path for path in checkpoint_paths
}

with st.sidebar:
    st.markdown("## Model setup")
    st.caption("Choose a saved training run and set the demo’s abstention threshold.")
    if run_by_label:
        selected_run = st.selectbox("Saved checkpoint", list(run_by_label))
        checkpoint = run_by_label[selected_run]
    else:
        selected_run = ""
        checkpoint = ""
        st.info("No trained checkpoints found in the project’s results folder.")
    threshold = st.slider(
        "Abstain below confidence",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01,
        help="If the model’s top class probability is below this value, the demo marks the result uncertain.",
    )
    st.markdown("---")
    st.markdown("**How to use**")
    st.caption("1. Select a checkpoint.\n\n2. Upload a JPG or PNG X-ray.\n\n3. Review the scores and heatmap.")

if not checkpoint:
    st.info("Train a model first. Its checkpoint will appear in the sidebar.")
    st.stop()

run_dir = os.path.dirname(checkpoint)
config = read_json(os.path.join(run_dir, "config.json"))
metrics = read_json(os.path.join(run_dir, "metrics.json"))

try:
    model, temperature, architecture = load_model(checkpoint)
except Exception as exc:
    st.error(f"Could not load this checkpoint: {exc}")
    st.stop()

st.markdown('<div class="section-label">Image review</div>', unsafe_allow_html=True)
st.markdown("### Upload an image")
st.caption("Accepted formats: JPG, JPEG, or PNG. The image is processed locally by this app.")
uploaded = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded is None:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("### What you’ll see")
        st.write("A class score, an uncertainty status based on your selected threshold, and an image overlay generated with Grad-CAM.")
    with right:
        st.markdown("### Selected run")
        st.write(f"**Run:** `{selected_run}`  ")
        st.write(f"**Architecture:** {architecture}  ")
        st.write(f"**Training epochs:** {config.get('epochs', 'not recorded')}  ")
        st.write(f"**Device used for training:** {config.get('env', {}).get('gpu') or 'CPU'}")
    st.stop()

try:
    image = Image.open(uploaded).convert("L")
except (UnidentifiedImageError, OSError):
    st.error("That file could not be read as an image. Please upload a JPG or PNG file.")
    st.stop()

input_tensor = get_transform(False)(image).unsqueeze(0).to(get_device())
with torch.no_grad():
    probabilities = torch.softmax(model(input_tensor) / temperature, dim=1)[0].cpu().numpy()

predicted_class = int(probabilities.argmax())
confidence = float(probabilities[predicted_class])
pneumonia_probability = float(probabilities[1])
normal_probability = float(probabilities[0])
is_uncertain = confidence < threshold
cam = gradcam(model, input_tensor, predicted_class)
heatmap_overlay = overlay(image, cam)

st.markdown('<div class="section-label">Model output</div>', unsafe_allow_html=True)
st.markdown("### Prediction summary")
card1, card2, card3 = st.columns(3)
with card1:
    st.markdown(
        f'<div class="result-card"><div class="result-label">Model class</div>'
        f'<div class="result-value">{"Uncertain" if is_uncertain else ("Pneumonia" if predicted_class else "Normal")}</div>'
        f'<div class="result-help">{"Below the selected confidence threshold" if is_uncertain else "Highest-scoring class"}</div></div>',
        unsafe_allow_html=True,
    )
with card2:
    st.markdown(
        f'<div class="result-card"><div class="result-label">Pneumonia score</div>'
        f'<div class="result-value">{pneumonia_probability:.1%}</div>'
        f'<div class="result-help">Temperature-scaled model score</div></div>',
        unsafe_allow_html=True,
    )
with card3:
    st.markdown(
        f'<div class="result-card"><div class="result-label">Top-class confidence</div>'
        f'<div class="result-value">{confidence:.1%}</div>'
        f'<div class="result-help">Abstention threshold: {threshold:.0%}</div></div>',
        unsafe_allow_html=True,
    )

if is_uncertain:
    st.warning("The top-class score is below the selected threshold, so this demo is withholding a class prediction.")
elif predicted_class == 1:
    st.info("The model’s highest-scoring class is Pneumonia. This is a model output, not a diagnosis.")
else:
    st.info("The model’s highest-scoring class is Normal. This is a model output, not a diagnosis.")

prob_col1, prob_col2 = st.columns(2)
with prob_col1:
    st.caption("P(Normal)")
    st.progress(normal_probability)
    st.caption(f"{normal_probability:.1%}")
with prob_col2:
    st.caption("P(Pneumonia)")
    st.progress(pneumonia_probability)
    st.caption(f"{pneumonia_probability:.1%}")

st.markdown('<div class="section-label">Visual explanation</div>', unsafe_allow_html=True)
st.markdown("### Image and Grad-CAM")
st.caption("Grad-CAM highlights regions associated with the selected class score; it does not prove the model’s reasoning is medically valid.")
image_col, cam_col = st.columns(2, gap="large")
with image_col:
    st.markdown('<div class="panel-title">Uploaded X-ray</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-help">Original grayscale image</div>', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
with cam_col:
    st.markdown('<div class="panel-title">Grad-CAM overlay</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-help">Colored areas indicate stronger contribution to the selected score</div>', unsafe_allow_html=True)
    st.image(heatmap_overlay, use_container_width=True)

with st.expander("About this model run"):
    details_col1, details_col2 = st.columns(2)
    with details_col1:
        st.write(f"**Run:** {selected_run}")
        st.write(f"**Architecture:** {architecture}")
        st.write(f"**Seed:** {config.get('seed', 'not recorded')}")
        st.write(f"**Epochs configured:** {config.get('epochs', 'not recorded')}")
    with details_col2:
        st.write(f"**Temperature:** {temperature:.3f}")
        st.write(f"**Training device:** {config.get('env', {}).get('gpu') or 'CPU'}")
        test_metrics = metrics.get("internal_test", {})
        if test_metrics:
            st.write(f"**Internal test AUROC:** {test_metrics.get('calibrated', {}).get('auroc', float('nan')):.3f}")
            st.write(f"**Internal test images:** {test_metrics.get('n', 'not recorded')}")
    st.caption("Internal test results are from the same dataset source and do not establish performance on other populations.")

st.markdown(
    '<div class="footer-note">Research and education only · Not validated for clinical use · '
    'Do not use this model output to make health decisions.</div>',
    unsafe_allow_html=True,
)
