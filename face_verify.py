import os
import tempfile
import io
from PIL import Image
from deepface import DeepFace

MODEL_NAME = "SFace"          # much lighter than the default VGG-Face
DETECTOR = "opencv"           # fastest detector backend
MAX_DIM = 640                 # downscale selfies before matching

_model = None


def get_model():
    global _model
    if _model is None:
        _model = DeepFace.build_model(MODEL_NAME)
    return _model


def warm_up():
    """Load model weights once so the first real verification isn't slow."""
    try:
        get_model()
        return True
    except Exception:
        return False


def _shrink(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((MAX_DIM, MAX_DIM))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def verify_face(known_image_path, live_image_bytes):
    """
    known_image_path: path to the registered biometric photo on disk
    live_image_bytes: raw bytes from a fresh st.camera_input capture
    Returns (verified: bool, message: str)
    """
    if not known_image_path or not os.path.exists(known_image_path):
        return False, "No registered biometric photo on file. Complete your profile first."

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(_shrink(live_image_bytes))
        live_path = tmp.name

    try:
        get_model()
        result = DeepFace.verify(
            img1_path=known_image_path,
            img2_path=live_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,
            enforce_detection=True,
        )
        if result["verified"]:
            return True, "Face verified."
        return False, "Face does not match your registered profile."
    except ValueError:
        return False, "Could not detect a face clearly. Try again with better lighting."
    finally:
        os.remove(live_path)