# 🏥 Gain Extraction Model - Patient Progress Prediction

A deep learning project that analyzes Patient Discharge Summaries to predict health progress and goal achievement using fine-tuned Bio_ClinicalBERT transformer models.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.0+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Usage](#-usage)
- [Training Process](#-training-process)
- [Results](#-results)
- [Visualization](#-visualization)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

The **Gain Extraction Model** is an advanced NLP system designed to automatically assess patient health improvements from clinical discharge summaries. By leveraging the Bio_ClinicalBERT transformer model, this project achieves high accuracy in determining whether patients have met their healthcare goals and made meaningful progress during treatment.

### Key Objectives

- 📊 Analyze patient discharge summaries efficiently
- 🎯 Predict patient health progress with high accuracy
- 🏥 Support healthcare providers in tracking patient outcomes
- ⚡ Provide automated, scalable health assessment tools

---

## ✨ Features

- **State-of-the-Art NLP**: Utilizes Bio_ClinicalBERT, specifically trained on MIMIC-III clinical notes
- **Automated Preprocessing**: Comprehensive text cleaning and normalization pipeline
- **Class Imbalance Handling**: Intelligent majority downsampling techniques
- **GPU Optimization**: CUDA-accelerated training and inference
- **Comprehensive Evaluation**: Detailed metrics including confusion matrices and accuracy scores
- **TensorBoard Integration**: Real-time training visualization
- **Production-Ready**: Modular code structure for easy deployment

---

## 📁 Project Structure

```
.
├── brief-review-of-gain-prediction.ipynb  # Main analysis notebook
├── brog_train.csv                          # Training dataset
├── brog_test.csv                           # Testing dataset
├── requirements.txt                        # Python dependencies
├── readme.txt                              # Project notes
├── save_tokenizer_bcb/                     # Saved tokenizer files
├── save_model_bcb/                         # Pre-trained model files
└── fine_tuned_model_bcb/                   # Fine-tuned model checkpoints
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/bilalhameed248/Brief-Review-Of-Gain-Prediction-Model.git
cd gain-prediction-project
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

---

## 📊 Dataset

### Data Description

The project uses clinical discharge summaries with binary classification:

- **Class 0**: No significant health improvement
- **Class 1**: Positive health progress/goal achievement

### Data Files

- `brog_train.csv`: Training dataset with labeled examples
- `brog_test.csv`: Test dataset for model evaluation

### Data Preprocessing

The preprocessing pipeline includes:

1. **Text Cleaning**
   - Removal of special characters and punctuation
   - Lowercase conversion
   - Pattern removal

2. **Linguistic Processing**
   - Stop word removal
   - Lemmatization using WordNet
   - Word tokenization

3. **Class Balancing**
   - Majority downsampling
   - Removal of low-information samples (< 8 words)

---

## 🧠 Model Architecture

### Bio_ClinicalBERT

**Base Model**: [emilyalsentzer/Bio_ClinicalBERT](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT)

**Training Data**: MIMIC-III database (~880M words from ICU patient notes)

**Architecture Details**:
- Transformer-based encoder
- 768 hidden dimensions
- 12 attention heads
- Fine-tuned for binary sequence classification

### Model Configuration

```python
- Input: Tokenized text sequences (max_length=130)
- Output: Binary classification (0/1)
- Loss Function: Cross-entropy
- Optimizer: AdamW
- Learning Rate: 1e-5
```

---

## 💻 Usage

### Quick Start

1. **Open the Jupyter notebook**
```bash
jupyter notebook brief-review-of-gain-prediction.ipynb
```

2. **Run all cells sequentially** or execute specific sections:
   - Data Loading & Preprocessing
   - Model Training
   - Evaluation & Visualization

### Inference Example

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('./save_tokenizer_bcb/')
model = AutoModelForSequenceClassification.from_pretrained('./fine_tuned_model_bcb/')

# Prepare input
text = "Patient shows significant improvement in mobility and pain management."
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=130)

# Get prediction
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    
print(f"Prediction: {'Positive Progress' if prediction == 1 else 'No Significant Progress'}")
```

---

## 🎓 Training Process

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 4 |
| Learning Rate | 1e-5 |
| Max Steps | 1000 |
| Warmup Steps | 500 |
| Evaluation Strategy | Every 5 steps |
| Max Sequence Length | 130 tokens |

### Training Pipeline

1. **Data Split**: 80% train, 10% validation, 10% test
2. **Tokenization**: Padding and truncation to max_length
3. **Fine-tuning**: Transfer learning from Bio_ClinicalBERT
4. **Evaluation**: Continuous validation monitoring
5. **Checkpoint Saving**: Best model selection

### Monitoring Training

Launch TensorBoard to visualize training metrics:

```bash
cd fine_tuned_model_bcb
tensorboard --logdir=runs
```

Access at: http://localhost:6006/

---

## 📈 Results

### Performance Metrics

| Metric | Pre-trained Model | Fine-tuned Model |
|--------|------------------|------------------|
| **Accuracy** | 50% | **90%** |
| **Improvement** | Baseline | **+40%** |

### Confusion Matrix

The model demonstrates strong performance in both positive and negative class predictions, with detailed confusion matrices available in the notebook visualizations.

### Key Findings

- ✅ Significant accuracy improvement after fine-tuning
- ✅ Balanced performance across both classes
- ✅ Effective handling of clinical terminology
- ✅ Robust to various text lengths and formats

---

## 📊 Visualization

The project includes comprehensive visualizations:

- **Class Distribution**: Bar plots showing label balance
- **Common Phrases**: Most frequent terms per class
- **Confusion Matrices**: Prediction accuracy breakdown
- **Training Curves**: Loss and accuracy over time (TensorBoard)

---

## 📦 Requirements

See [requirements.txt](requirements.txt) for full details. Key dependencies:

```txt
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
seaborn>=0.12.0
matplotlib>=3.7.0
nltk>=3.8
accelerate>=0.20.0
evaluate>=0.4.0
pynvml>=11.5.0
tensorboard>=2.13.0
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Bio_ClinicalBERT**: Emily Alsentzer et al. for the pre-trained clinical BERT model
- **MIMIC-III**: Beth Israel Deaconess Medical Center for the clinical database
- **Hugging Face**: For the Transformers library and model hub
- **PyTorch**: For the deep learning framework

---

## 📧 Contact

For questions, issues, or collaboration opportunities:

- **Project Repository**: [GitHub Link](your-github-link)
- **Issue Tracker**: [Issues Page](your-github-link/issues)

---

## 🔮 Future Work

- [ ] Multi-class classification for different progress levels
- [ ] Integration with real-time EHR systems
- [ ] Deployment as REST API
- [ ] Support for multiple languages
- [ ] Explainability features (LIME, SHAP)
- [ ] Model compression for edge deployment

---

<div align="center">

**Made with ❤️ for better healthcare outcomes**

⭐ Star this repo if you find it helpful!

</div>