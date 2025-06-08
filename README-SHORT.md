# 🧠 EmotiSense-AI: Emotion Classification System

> **AI-powered text emotion classification using DistilBERT transformers**

## 📊 Dataset
- **Source**: [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
- **Size**: 20,000 samples (16K train, 2K validation, 2K test)
- **Classes**: 6 emotions - Joy, Sadness, Anger, Fear, Love, Surprise
- **Format**: Social media text posts in English

## 🎯 Approach
1. **Baseline Models**: TF-IDF + Logistic Regression/SVM (81.2% accuracy)
2. **Transformer Model**: Fine-tuned DistilBERT (93.5% accuracy)
3. **Class Imbalance Handling**: Balanced weights for fair representation
4. **Interactive Web App**: Streamlit interface for real-time predictions

## 📈 Results
| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| SVM Baseline | 81.2% | 0.808 | 5 min |
| **DistilBERT** | **93.5%** | **0.934** | 15 min |

## 🛠️ Dependencies
```bash
# Core ML
torch==2.0.1
transformers==4.30.2
scikit-learn==1.3.0
pandas==2.0.3

# Visualization
matplotlib==3.7.1
seaborn==0.12.2

# Web App
streamlit==1.24.0
```

## 🚀 Quick Start
```bash
# Clone repository
git clone https://github.com/Sharnabh/EmotiSense-AI.git
cd EmotiSense-AI

# Install dependencies
pip install -r requirements.txt

# Launch web app
streamlit run app/streamlit_app.py
```

## 📁 Project Structure
```
├── notebooks/          # EDA & model training
├── app/               # Streamlit web interface  
├── outputs/           # Trained models & visualizations
└── requirements.txt   # Dependencies
```

## 🎯 Key Features
- **93.5% accuracy** with fine-tuned DistilBERT
- **Real-time predictions** via interactive web interface
- **Comprehensive analysis** with confusion matrices and visualizations
- **Professional codebase** with Git LFS for large files

---

**Author**: Sharnabh Banerjee | [GitHub](https://github.com/Sharnabh/EmotiSense-AI) | [LinkedIn](https://linkedin.com/in/sharnabh)
