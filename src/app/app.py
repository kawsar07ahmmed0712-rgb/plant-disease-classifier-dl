from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# --- Constants (robust across torchvision versions) ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

APP_TITLE = "🌿 Plant Disease Classifier"
DEFAULT_CKPT_PATH = Path("models/best_model.pt")


# --- Caching: load model once ---
@st.cache_resource
def load_model(ckpt_path: str):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path.resolve()}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label_names = ckpt["label_names"]
    img_size = int(ckpt.get("img_size", 224))

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(label_names))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    meta = {
        "arch": ckpt.get("arch", "resnet18"),
        "img_size": img_size,
        "labels": label_names,
        "ckpt_path": str(ckpt_path),
    }
    return model, tf, meta


@torch.no_grad()
def predict(model, tf, pil_img: Image.Image):
    x = tf(pil_img.convert("RGB")).unsqueeze(0)  # [1, C, H, W]
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return pred_idx, probs


def class_badge(label: str):
    # You can tweak these labels later if you use another dataset
    if "healthy" in label.lower():
        return "✅ Healthy"
    if "rust" in label.lower():
        return "🟠 Rust"
    if "spot" in label.lower():
        return "🔴 Leaf Spot"
    return f"🔎 {label}"


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🌿", layout="centered")

    st.title(APP_TITLE)
    st.caption("Upload a leaf image to predict disease class using a fine-tuned ResNet18 model.")

    # Sidebar (clean + informative)
    st.sidebar.header("⚙️ Settings")
    ckpt_path = st.sidebar.text_input("Model checkpoint path", value=str(DEFAULT_CKPT_PATH))
    top_k = st.sidebar.slider("Top-K probabilities", min_value=2, max_value=5, value=3)
    auto_predict = st.sidebar.checkbox("Auto-predict after upload", value=True)

    # Load model artifacts
    try:
        model, tf, meta = load_model(ckpt_path)
    except Exception as e:
        st.error("Model could not be loaded.")
        st.code(str(e))
        st.info("Make sure you trained the model and the checkpoint exists at the given path.")
        st.stop()

    with st.expander("📌 Model info", expanded=False):
        st.write(f"**Architecture:** {meta['arch']}")
        st.write(f"**Image size:** {meta['img_size']}×{meta['img_size']}")
        st.write(f"**Classes:** {', '.join(meta['labels'])}")
        st.write(f"**Checkpoint:** {meta['ckpt_path']}")

    st.divider()

    uploaded = st.file_uploader("📤 Upload a leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to start.")
        st.stop()

    # Preview
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded image", use_container_width=True)

    predict_clicked = st.button("🔮 Predict", type="primary") if not auto_predict else True

    if predict_clicked:
        pred_idx, probs = predict(model, tf, img)
        labels = meta["labels"]

        pred_label = labels[pred_idx]
        confidence = float(probs[pred_idx])

        st.subheader("Result")
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown(f"### {class_badge(pred_label)}")
            st.write(f"**Predicted class:** `{pred_label}`")
            st.write(f"**Confidence:** `{confidence:.2%}`")

            # Simple guidance text (nice for portfolio demo)
            if "healthy" in pred_label.lower():
                st.success("Looks healthy. Maintain good watering and monitoring.")
            else:
                st.warning("Potential disease detected. Consider isolating plant and checking treatment options.")

        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
            st.progress(min(max(confidence, 0.0), 1.0))

        st.divider()
        st.subheader("Probabilities")

        # Top-K table
        order = np.argsort(-probs)
        top_idx = order[:top_k]
        rows = [{"class": labels[i], "probability": float(probs[i])} for i in top_idx]

        st.dataframe(rows, use_container_width=True)

        # Bar chart (all classes)
        chart_data = {labels[i]: float(probs[i]) for i in range(len(labels))}
        st.bar_chart(chart_data)

        with st.expander("Show raw probabilities"):
            st.json({labels[i]: float(probs[i]) for i in range(len(labels))})


if __name__ == "__main__":
    main()
