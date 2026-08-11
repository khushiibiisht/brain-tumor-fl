# app.py
# Flask web application for Brain Tumor FL project

import os
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

# ─────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Flask app using device: {device}')

# ─────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────
def load_model():
    from model import get_model

    model = get_model(device)
    weights_path = 'logs/global_weights.npy'

    if os.path.exists(weights_path):
        print('Loading FL trained weights from global_weights.npy...')

        # Load numpy arrays saved by server.py after each FL round
        ndarrays = np.load(
            weights_path,
            allow_pickle=True
        )

        # Pair layer names with loaded numpy arrays
        params_dict = zip(
            model.state_dict().keys(),
            ndarrays
        )

        # Convert numpy arrays → PyTorch tensors
        state_dict = {
            k: torch.tensor(v)
            for k, v in params_dict
        }

        # Load into model
        model.load_state_dict(state_dict, strict=True)
        print('FL weights loaded successfully!')

    else:
        print('WARNING: logs/global_weights.npy not found!')
        print('Run FL training first (python server.py + clients)')
        print('Using pretrained ImageNet weights only — predictions will be poor')

    model.eval()
    return model


# Load model once when Flask starts
try:
    model = load_model()
    MODEL_LOADED = True
    print('Model ready for inference!')
except Exception as e:
    print(f'Model load error: {e}')
    MODEL_LOADED = False


# ─────────────────────────────────────────
# CLASS DEFINITIONS
# ─────────────────────────────────────────
# Order MUST match training folder order (alphabetical)
# glioma=0, meningioma=1, notumor=2, pituitary=3
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

CLASS_DESCRIPTIONS = {
    'Glioma'    : 'A tumor that starts in the glial cells of the brain or spine. Most common primary brain tumor.',
    'Meningioma': 'A tumor that forms on the membranes covering the brain and spinal cord. Usually slow growing.',
    'No Tumor'  : 'No tumor detected in the MRI scan. Brain tissue appears normal.',
    'Pituitary' : 'A tumor that forms in the pituitary gland at the base of the brain. Affects hormone production.'
}

CLASS_COLORS = {
    'Glioma'    : '#fc8181',
    'Meningioma': '#f6ad55',
    'No Tumor'  : '#68d391',
    'Pituitary' : '#63b3ed'
}

# ─────────────────────────────────────────
# IMAGE TRANSFORM
# Must match exactly what was used during training
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives uploaded MRI image.
    Returns prediction + confidence scores as JSON.
    """
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded. Run FL training first.'
        }), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Open image and convert to RGB
        # (MRI scans sometimes come as grayscale — RGB handles both)
        img = Image.open(file.stream).convert('RGB')

        # Apply same transforms used during training
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Run inference — no gradient needed
        with torch.no_grad():
            outputs = model(img_tensor)

            # Softmax converts raw scores to probabilities (sum=1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            # Get highest probability class
            confidence, predicted = probabilities.max(1)

        # Build response
        pred_idx   = predicted.item()
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf  = confidence.item() * 100
        all_probs  = probabilities[0].tolist()

        return jsonify({
            'prediction'       : pred_class,
            'confidence'       : round(pred_conf, 2),
            'description'      : CLASS_DESCRIPTIONS[pred_class],
            'color'            : CLASS_COLORS[pred_class],
            'all_probabilities': {
                CLASS_NAMES[i]: round(all_probs[i] * 100, 2)
                for i in range(len(CLASS_NAMES))
            },
            'model_loaded_from': 'FL trained weights (global_weights.npy)'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/api/results')
def api_results():
    """Returns all strategy results as JSON for charts."""
    results_files = {
        'FedAvg' : 'logs/results.json',
        'FedProx': 'logs/fedprox_results.json',
        'FedBN'  : 'logs/fedbn_results.json'
    }
    all_results = {}
    for strategy, path in results_files.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                all_results[strategy] = json.load(f)
    return jsonify(all_results)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health():
    """Health check endpoint — useful for Docker."""
    return jsonify({
        'status'      : 'running',
        'model_loaded': MODEL_LOADED,
        'device'      : str(device),
        'weights_exist': os.path.exists('logs/global_weights.npy')
    })


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == '__main__':
    print('\n' + '='*50)
    print('  BRAIN TUMOR FL — FLASK UI')
    print(f'  Device       : {device}')
    print(f'  Model loaded : {MODEL_LOADED}')
    print(f'  Weights exist: {os.path.exists("logs/global_weights.npy")}')
    print('='*50)
    print('\nOpen http://localhost:5000 in your browser\n')

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )