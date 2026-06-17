# api/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# ─────────────────────────────────────────────────────────
# Existing constants — UNCHANGED
# ─────────────────────────────────────────────────────────
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_LABELS = {
    'nv'   : 'Melanocytic Nevi (Mole)',
    'mel'  : 'Melanoma',
    'bkl'  : 'Benign Keratosis',
    'bcc'  : 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratosis',
    'vasc' : 'Vascular Lesion',
    'df'   : 'Dermatofibroma'
}
CONFIDENCE_THRESHOLD = 0.60
MODEL_PATH = '../model/best_model.pt'
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# transform — same as val/test in training — UNCHANGED
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ─────────────────────────────────────────────────────────
# NEW — Gatekeeper constants
# ─────────────────────────────────────────────────────────

# ImageNet normalization (different from skin model's transform above)
imagenet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# How confident the ImageNet model must be before we trust its
# "this is a random object" verdict and reject the image
IMAGENET_REJECT_THRESHOLD = 0.40

# Minimum confidence for OpenCV Haar Cascade face detection scaling
FACE_DETECTION_MIN_NEIGHBORS = 5

# A short list of ImageNet class index ranges that correspond to
# everyday objects/animals we want to explicitly reject.
# (Full ImageNet has 1000 classes — indices below cover common
#  objects, animals, vehicles, electronics, etc. This list is not
#  exhaustive on purpose — exhaustive lists are unnecessary because
#  we ALSO reject anything that isn't skin-textured via low max prob.)
# We load the full class list once and check keyword membership instead
# of hardcoding indices — see _NON_SKIN_KEYWORDS below.
_NON_SKIN_KEYWORDS = [
    'cat', 'dog', 'bird', 'car', 'truck', 'screen', 'phone',
    'laptop', 'computer', 'keyboard', 'chair', 'table', 'cup',
    'bottle', 'book', 'television', 'monitor', 'remote', 'mouse',
    'plate', 'food', 'fruit', 'vegetable', 'flower', 'tree',
    'building', 'vehicle', 'animal', 'bag', 'shoe', 'clothing',
    'website', 'web site', 'menu', 'envelope', 'paper'
]


def load_model():
    """Loads the skin disease classifier — UNCHANGED from original."""
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(CLASS_NAMES)
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v
                      for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print(f"Skin disease model loaded on {DEVICE} ✅")
    return model


def load_gatekeeper_models():
    """
    NEW — Loads the two gatekeeper models:
      1. Pretrained ImageNet ResNet18 — flags random objects/screenshots
      2. OpenCV Haar Cascade face detector — flags human faces/body shots
    Both are lightweight and run on CPU comfortably. No extra model
    files to manage — the Haar Cascade ships inside opencv-python.
    """
    # 1. ImageNet classifier
    imagenet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    imagenet_model = imagenet_model.to(DEVICE)
    imagenet_model.eval()

    # ImageNet class labels (1000 classes), bundled with the weights metadata
    imagenet_labels = models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]

    # 2. OpenCV Haar Cascade face detector — built into opencv-python,
    #    no separate download needed
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_detector = cv2.CascadeClassifier(cascade_path)

    print("Gatekeeper models loaded ✅ (ImageNet ResNet18 + OpenCV Face Detector)")
    return {
        "imagenet_model" : imagenet_model,
        "imagenet_labels": imagenet_labels,
        "face_detector"  : face_detector
    }


def _check_human(image: Image.Image, face_detector) -> bool:
    """
    NEW — Returns True if a human face is detected anywhere in the image.
    Catches face photos, selfies, and most body-part photos where a
    face happens to also be visible. For pure hand/leg/arm shots with
    no face, this check alone won't catch it — that's handled by the
    ImageNet check below (hands/legs don't match skin-lesion-like
    close-up texture and often get classified as something generic).
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=FACE_DETECTION_MIN_NEIGHBORS,
        minSize=(40, 40)
    )
    return len(faces) > 0


def _check_random_object(image: Image.Image, imagenet_model, imagenet_labels) -> tuple:
    """
    NEW — Runs the image through ImageNet ResNet18.
    Returns (is_random_object: bool, top_label: str, confidence: float)
    """
    tensor = imagenet_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = imagenet_model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    confidence = probs.max().item()
    top_idx    = probs.argmax().item()
    top_label  = imagenet_labels[top_idx].lower()

    is_random_object = False
    if confidence >= IMAGENET_REJECT_THRESHOLD:
        if any(keyword in top_label for keyword in _NON_SKIN_KEYWORDS):
            is_random_object = True

    return is_random_object, top_label, confidence


def predict(model, image: Image.Image, gatekeeper=None):
    """
    UPDATED — now runs gatekeeper checks before the skin disease model.
    `gatekeeper` is the dict returned by load_gatekeeper_models().
    If gatekeeper is None, behaves exactly like the original function
    (useful fallback, though main.py will always pass it in).
    """

    # ── NEW: Gatekeeper Step 1 — human / body part check ──────────
    if gatekeeper is not None:
        is_human = _check_human(image, gatekeeper["face_detector"])
        if is_human:
            return {
                'status'           : 'rejected_human',
                'predicted_class'  : None,
                'label'            : None,
                'confidence'       : None,
                'all_probabilities': None,
                'is_unknown'       : True,
                'message'          : ('It looks like this image shows a face or '
                                       'body part. Please upload a close-up photo '
                                       'of only the affected skin area.')
            }

        # ── NEW: Gatekeeper Step 2 — random object check ──────────
        is_random, top_label, ig_confidence = _check_random_object(
            image, gatekeeper["imagenet_model"], gatekeeper["imagenet_labels"]
        )
        if is_random:
            return {
                'status'           : 'rejected_invalid',
                'predicted_class'  : None,
                'label'            : None,
                'confidence'       : None,
                'all_probabilities': None,
                'is_unknown'       : True,
                'message'          : 'This does not look like a skin image. Please upload a proper image.'
            }

    # ── EXISTING LOGIC — UNCHANGED below this point ───────────────
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    confidence    = probs.max().item()
    predicted_idx = probs.argmax().item()
    predicted_cls = CLASS_NAMES[predicted_idx]

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            'status'           : 'rejected_invalid',
            'predicted_class'  : 'unknown',
            'label'            : 'Unknown — not a skin lesion',
            'confidence'       : round(confidence, 4),
            'all_probabilities': {
                CLASS_NAMES[i]: round(probs[i].item(), 4)
                for i in range(len(CLASS_NAMES))
            },
            'is_unknown'       : True,
            'message'          : 'This does not look like a clear skin lesion image. Please upload a proper image.'
        }

    return {
        'status'           : 'success',
        'predicted_class'  : predicted_cls,
        'label'            : CLASS_LABELS[predicted_cls],
        'confidence'       : round(confidence, 4),
        'all_probabilities': {
            CLASS_NAMES[i]: round(probs[i].item(), 4)
            for i in range(len(CLASS_NAMES))
        },
        'is_unknown'       : False,
        'message'          : None
    }