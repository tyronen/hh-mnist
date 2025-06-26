import inspect
import random
import string

from create_composite_images import START_TOKEN, END_TOKEN, PAD_TOKEN

import streamlit as st
import torch
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from torchvision import transforms

import models
import utils

device = utils.get_device()


@st.cache_resource
def load_model():
    checkpoint = torch.load(utils.COMPLEX_MODEL_FILE, map_location=device)
    config = checkpoint["config"]

    # keep only the arguments that ComplexTransformer’s __init__ expects
    ctor_keys = inspect.signature(models.ComplexTransformer).parameters
    ctor_cfg = {k: v for k, v in config.items() if k in ctor_keys}

    model = models.ComplexTransformer(**ctor_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)  # keep parameters on the same device as inputs
    model.eval()
    return model


def preprocess_image(image_data):
    """Preprocess one 280×280 canvas → (1, 28, 28) normalised tensor."""
    image = Image.fromarray(image_data).convert("L")

    # tight crop around the drawing
    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    # centre‑pad to square then resize to 28×28
    w, h = image.size
    side = max(w, h) + 40
    pad = (
        (side - w) // 2,
        (side - h) // 2,
        side - w - (side - w) // 2,
        side - h - (side - h) // 2,
    )
    image = ImageOps.expand(image, border=pad, fill=0)

    tfm = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.Lambda(
                lambda img: img.point(lambda p: 255 if p > 50 else 0, "L")
            ),
            transforms.GaussianBlur(kernel_size=3, sigma=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return tfm(image).squeeze(0)  # (28, 28)


def assemble_composite(tl, tr, bl, br):
    """Stack four (28,28) tensors into one (1,56,56) composite."""
    composite = torch.zeros(56, 56)
    composite[:28, :28] = tl
    composite[:28, 28:] = tr
    composite[28:, :28] = bl
    composite[28:, 28:] = br
    return composite.unsqueeze(0).unsqueeze(0)  # (1,1,56,56)


def random_string():
    return "".join(
        random.choice(string.ascii_letters + string.digits) for _ in range(8)
    )


TEMPERATURE = 2.0

INTRO = """
This is a demonstration app showing simple handwriting recognition.

This app uses a deep-learning model trained on the MNIST public dataset of 
handwritten digits using the Pytorch library.

Draw digits (0-9) in the black boxes and press Predict. The model will then 
attempt to guess what digits you have entered, and how confident it is in that
guess as a percentage."""


def make_canvas(key_index):
    return st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key=st.session_state.canvas_keys[key_index],
    )


def main():
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "has_prediction" not in st.session_state:
        st.session_state.has_prediction = False
    if "canvas_keys" not in st.session_state:
        st.session_state.canvas_keys = [
            random_string(),
            random_string(),
            random_string(),
            random_string(),
        ]

    st.title("Digit Recogniser")
    st.markdown(INTRO)

    left, right = st.columns(2)
    with left:
        canvasTL = make_canvas(0)
        canvasBL = make_canvas(2)
    with right:
        canvasTR = make_canvas(1)
        canvasBR = make_canvas(3)

    model = load_model()

    if st.button("Predict", type="primary"):
        if not all(
            [c.image_data is not None for c in (canvasTL, canvasTR, canvasBL, canvasBR)]
        ):
            st.error("Please draw a digit in **all four** squares before predicting.")
        else:
            tl = preprocess_image(canvasTL.image_data)
            tr = preprocess_image(canvasTR.image_data)
            bl = preprocess_image(canvasBL.image_data)
            br = preprocess_image(canvasBR.image_data)

            composite = assemble_composite(tl, tr, bl, br).to(device)

            with torch.no_grad():
                # greedy autoregressive decode ─ up to 4 digits or <END>
                input_seq = torch.full(
                    (1, 5), PAD_TOKEN, device=device, dtype=torch.long
                )
                input_seq[0, 0] = START_TOKEN
                tokens = []
                for pos in range(4):  # max 4 digits
                    logits = model(composite, input_seq)  # (1, 5, vocab)
                    next_pos = (
                        (input_seq[0] == PAD_TOKEN).nonzero(as_tuple=False)[0].item()
                    )
                    next_token = logits.argmax(-1)[
                        0, next_pos
                    ].item()  # token at current position
                    if next_token == END_TOKEN:
                        break
                    tokens.append(next_token)
                    input_seq[0, next_pos] = next_token  # feed it back

                st.session_state.prediction = tokens
            st.session_state.has_prediction = True

    if st.session_state.has_prediction:
        pred_str = " ".join(str(d) for d in st.session_state.prediction)
        st.write(f"**Predicted digits (TL→TR→BL→BR):** {pred_str}")


if __name__ == "__main__":
    main()
