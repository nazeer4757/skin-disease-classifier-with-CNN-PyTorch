# api/model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# constants
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

# transform — same as val/test in training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_model():
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, len(CLASS_NAMES)
    )
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # handle DataParallel prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v
                      for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE} ✅")
    return model


def predict(model, image: Image.Image):
    # preprocess
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    # inference
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]

    confidence    = probs.max().item()
    predicted_idx = probs.argmax().item()
    predicted_cls = CLASS_NAMES[predicted_idx]

    # confidence threshold check
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            'predicted_class'  : 'unknown',
            'label'            : 'Unknown — not a skin lesion',
            'confidence'       : round(confidence, 4),
            'all_probabilities': {
                CLASS_NAMES[i]: round(probs[i].item(), 4)
                for i in range(len(CLASS_NAMES))
            },
            'is_unknown'       : True
        }

    return {
        'predicted_class'  : predicted_cls,
        'label'            : CLASS_LABELS[predicted_cls],
        'confidence'       : round(confidence, 4),
        'all_probabilities': {
            CLASS_NAMES[i]: round(probs[i].item(), 4)
            for i in range(len(CLASS_NAMES))
        },
        'is_unknown'       : False
    }