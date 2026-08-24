# Brain Tumor Detection using Federated Learning

A privacy-preserving brain tumor classification system using 
Federated Learning. Patient MRI data never leaves individual 
hospital nodes —  model weights are shared only .

## Project Structure

brain-tumor-fl/
├── data/ ← download dataset here (see below)
├── logs/ ← training results saved here
├── model.py ← ResNet-18 architecture
├── train_utils.py ← train/test functions
├── split_data.py ← splits dataset into hospital portions
├── client.py ← hospital FL client
├── server.py ← central FL aggregator
└── requirements.txt

## Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/khushiibiisht/brain-tumor-fl.git
cd brain-tumor-fl
```

### 2. Create virtual environment (use Python 3.11)
```bash
py -3.11 -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install packages
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download from Kaggle:
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Extract into the `data/` folder so structure looks like:

data/
├── Training/
│ ├── glioma/
│ ├── meningioma/
│ ├── notumor/
│ └── pituitary/
└── Testing/
├── glioma/
├── meningioma/
├── notumor/
└── pituitary/

### 5. Split dataset into hospital clients
```bash
python split_data.py
```

### 6. Run Federated Learning

Open 4 terminals. In each, activate venv first.

**Terminal 1 — Start server:**
```bash
python server.py
```

**Terminal 2, 3, 4 — Start clients:**
```bash
python client.py 0
python client.py 1
python client.py 2
```

## Model
- Architecture: ResNet-18 (pretrained on ImageNet)
- Task: 4-class MRI classification
- Classes: glioma, meningioma, notumor, pituitary

## FL Strategy
- FedAvg (baseline)
- FedProx (handles unequal data)
- FedBN (handles scanner heterogeneity)

## Team
