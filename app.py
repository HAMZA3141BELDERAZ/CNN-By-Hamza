from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from cnn_model import build_model


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "mnist_cnn.keras"
DIGITS = [str(digit) for digit in range(10)]

st.set_page_config(
    page_title="Digit Lab | MNIST CNN",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
      :root { --ink: #182321; --muted: #61716d; --paper: #f4f5ef; --lime: #d8f36a; --teal: #0d7567; }
      html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
      .stApp { background: var(--paper); color: var(--ink); }
      [data-testid="stSidebar"] { background: #e7ebe3; border-right: 1px solid #d6ddd4; }
      [data-testid="stSidebar"] h2 { font-size: 1.02rem; }
      .eyebrow { color: var(--teal); font: 500 0.72rem 'DM Mono', monospace; letter-spacing: 0.08em; text-transform: uppercase; }
      .hero { padding: 1.3rem 0 1.6rem; border-bottom: 1px solid #d6ddd4; margin-bottom: 1.7rem; }
      .hero h1 { font-size: clamp(2.2rem, 4vw, 3.6rem); line-height: 1; margin: .55rem 0 .7rem; letter-spacing: 0; }
      .hero p { max-width: 680px; color: var(--muted); font-size: 1rem; margin: 0; }
      .result { background: var(--ink); color: #f4f5ef; padding: 1.25rem 1.4rem; border-radius: 6px; }
      .result-label { color: var(--lime); font: 500 .7rem 'DM Mono', monospace; text-transform: uppercase; }
      .result-digit { font-size: 4rem; line-height: 1.05; font-weight: 700; margin: .25rem 0; }
      div.stButton > button[kind="primary"] { background: var(--teal); border-color: var(--teal); }
      div.stButton > button { border-radius: 4px; }
      [data-testid="stMetric"] { background: #e8ece4; padding: .8rem 1rem; border-radius: 5px; }
      .footnote { color: var(--muted); font-size: .84rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, modified_time: float):
    del modified_time
    return tf.keras.models.load_model(model_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Convert a photo or drawing into a centered, MNIST-like 28x28 tensor."""
    grayscale = ImageOps.grayscale(image)
    pixels = np.asarray(grayscale)
    if pixels.mean() > 127:
        grayscale = ImageOps.invert(grayscale)

    grayscale = ImageOps.autocontrast(grayscale)
    foreground = grayscale.point(lambda value: 255 if value > 30 else 0)
    bounds = foreground.getbbox()
    if bounds:
        grayscale = grayscale.crop(bounds)

    grayscale.thumbnail((20, 20), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    canvas.paste(grayscale, ((28 - grayscale.width) // 2, (28 - grayscale.height) // 2))
    return np.asarray(canvas, dtype=np.float32)[np.newaxis, ..., np.newaxis] / 255.0


def train_and_save_model() -> tuple[float, float]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
    x_test = x_test.astype("float32")[..., np.newaxis] / 255.0

    model = build_model()
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=0,
    )
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    load_model.clear()
    return float(loss), float(accuracy)


st.markdown(
    """
    <header class="hero">
      <div class="eyebrow">Computer vision / MNIST</div>
      <h1>Digit Lab<span style="color:#0d7567">.</span></h1>
      <p>A compact convolutional neural network that reads handwritten digits. Train it on MNIST, then test it with your own image.</p>
    </header>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="eyebrow">Model controls</div>', unsafe_allow_html=True)
    st.subheader("MNIST CNN")
    is_trained = MODEL_PATH.exists()
    st.write("Status: **Ready**" if is_trained else "Status: **Not trained**")
    st.caption("2 convolution layers · 5 epochs · 10 digit classes")

    if st.button("Train model" if not is_trained else "Retrain model", type="primary", use_container_width=True):
        with st.spinner("Training on 60,000 MNIST images..."):
            loss, accuracy = train_and_save_model()
        st.session_state["test_accuracy"] = accuracy
        st.session_state["test_loss"] = loss
        st.rerun()

    if "test_accuracy" in st.session_state:
        st.metric("MNIST test accuracy", f"{st.session_state['test_accuracy']:.1%}")
        st.caption(f"Test loss: {st.session_state['test_loss']:.4f}")

    st.divider()
    st.markdown("**Input tips**")
    st.caption("Upload a clear image with one dark digit on a light background, or a white digit on a dark background. PNG and JPEG are supported.")

left, right = st.columns([1.05, 0.95], gap="large")
with left:
    st.subheader("1. Add a digit")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"],
        help="For best results, use a centered handwritten digit with minimal background clutter.",
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
    else:
        st.info("Your uploaded digit will appear here.")

with right:
    st.subheader("2. Prediction")
    if not MODEL_PATH.exists():
        st.warning("Train the model from the sidebar before running a prediction.")
    elif uploaded_file:
        model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
        probabilities = model.predict(preprocess_image(image), verbose=0)[0]
        predicted_digit = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_digit])

        st.markdown(
            f'<section class="result"><div class="result-label">Predicted digit</div>'
            f'<div class="result-digit">{predicted_digit}</div>'
            f'<div>{confidence:.1%} confidence</div></section>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown("**Class probabilities**")
        st.bar_chart({"Confidence": probabilities}, horizontal=True, height=280)
    else:
        st.info("Upload an image after training to see the prediction and class probabilities.")

st.markdown(
    '<p class="footnote">Built with TensorFlow and Streamlit · Predictions are most reliable on simple, centered handwritten digits.</p>',
    unsafe_allow_html=True,
)
