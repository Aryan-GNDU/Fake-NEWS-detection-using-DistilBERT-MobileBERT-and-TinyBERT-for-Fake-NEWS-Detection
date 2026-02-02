<p align="center">
  <h1 align="center">📰 Fake News Detection using Fine-Tuned DistilBERT</h1>
  <h3 align="center">Production-Ready NLP System | Transformer Fine-Tuning | GPU Training</h3>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Model-DistilBERT-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-HuggingFace-yellow?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Backend-PyTorch-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Accelerator-CUDA-green?style=for-the-badge"/>
</p>

---

# 🎯 Executive Summary

This project implements an **end-to-end Transformer-based NLP system** for detecting fake news.

Instead of using a prebuilt API, this project:

- Fine-tunes a pretrained DistilBERT model
- Optimizes hyperparameters
- Trains on GPU (CUDA)
- Evaluates with balanced classification metrics
- Deploys a production-ready inference app via Streamlit

This demonstrates **real-world ML engineering capability**, not just inference usage.

---


### What the Project Does:
- User enters headline
- Model predicts FAKE or REAL
- Confidence score displayed
- Clean UX with visual indicators

---

# 🏗️ System Architecture

```
                    ┌────────────────────────┐
                    │      Raw News Text      │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │    Text Preprocessing   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  DistilBERT Tokenizer   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Fine-Tuned DistilBERT  │
                    │ (Classification Head)   │
                    └────────────┬───────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  Prediction + Score     │
                    └────────────────────────┘
```

---

# 🧠 Fine-Tuning Strategy (Core Engineering Work)

## 🔹 Why Fine-Tuning?

Training from scratch is expensive.  
Instead, we leveraged **transfer learning** from pretrained DistilBERT and adapted it to Fake News classification.

This allows:
- Faster convergence
- Better generalization
- Lower data requirements

---

## 🔹 Model Initialization

```python
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
```

We replaced the original classification head with a new 2-label head.

---

## 🔹 Tokenization Pipeline

```python
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding=True,
        truncation=True,
        max_length=512
    )
```

---

## 🔹 Training Configuration

```python
training_args = TrainingArguments(
    output_dir="bert_base_train",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True
)
```

### Engineering Considerations

- Larger batch size → better gradient stability
- Weight decay → reduced overfitting
- Epoch evaluation → monitoring convergence
- Best model loading → optimal checkpoint selection

---

## 🔹 GPU Acceleration

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

Transformer models are matrix-heavy → GPU significantly improves training speed.

---

# 📊 Model Evaluation

Because Fake News datasets are often imbalanced, accuracy alone is misleading.

We used:

- Accuracy
- Precision
- Recall
- F1-Score

---

# 🧠 Confusion Matrix Visualization

Add this to your notebook:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = trainer.predict(encoded_test).predictions.argmax(-1)
y_true = encoded_test["label"]

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["REAL","FAKE"],
            yticklabels=["REAL","FAKE"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---


### Why This Matters

The confusion matrix reveals:

- False Positives (real predicted fake)
- False Negatives (fake predicted real)

This is critical in misinformation detection systems.

---

# 🌐 Deployment Architecture

```
User Input
     │
     ▼
Streamlit UI
     │
     ▼
HuggingFace Pipeline
     │
     ▼
Fine-Tuned Model
     │
     ▼
Classification + Confidence
```

Run locally:

```bash
streamlit run app.py
```

---

# 📁 Project Structure

```
.
├── fine tuning.ipynb        # Model training & evaluation
├── app.py                   # Streamlit deployment
├── fake_news/               # Saved trained model
├── demo.gif                 # App demo animation
├── confusion_matrix.png     # Model evaluation visualization
└── README.md
```

---

# 🏆 Engineering Highlights

✔ Transfer learning with Transformers  
✔ Hyperparameter tuning  
✔ GPU-based training  
✔ Proper evaluation strategy  
✔ Model checkpointing  
✔ Deployment-ready system  
✔ Clean inference pipeline  

---

# 💼 Skills Demonstrated (Portfolio Ready)

- Transformer Architecture Understanding
- HuggingFace Trainer API
- PyTorch & CUDA
- Evaluation Metric Engineering
- Model Deployment
- Production Thinking
- End-to-End ML Pipeline Design

---

# 🚀 Future Improvements

- Model quantization for faster inference
- Deploy on HuggingFace Spaces
- Add model explainability (SHAP)
- Add real-time API backend (FastAPI)

---

# 👨‍💻 Author

**Aryan Katoch**  
Electronics & Computer Engineering  
NLP | Transformers | Applied AI Engineering  

---

<p align="center">
  ⭐ If this project impressed you, consider giving it a star!
</p>
