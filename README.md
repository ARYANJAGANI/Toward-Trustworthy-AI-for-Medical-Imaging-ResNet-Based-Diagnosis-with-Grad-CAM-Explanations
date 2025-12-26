# Toward Trustworthy AI for Medical Imaging  
### ResNet-Based Skin Lesion Diagnosis with Grad-CAM++ Explanations

---

## 📌 Project Overview

Deep learning models achieve high accuracy in medical image classification, but their adoption in clinical practice is limited due to a lack of interpretability. In high-stakes domains such as **skin cancer diagnosis**, clinicians require not only accurate predictions but also **clear and trustworthy explanations**.

This project presents a **ResNet-50–based skin lesion classification system** enhanced with **Explainable AI (XAI)** techniques. We focus on **Grad-CAM++** to generate fine-grained, class-discriminative heatmaps and compare it against **Grad-CAM** and **Score-CAM**. Additionally, we introduce a **text-based explanation engine** that summarizes model attention in human-readable terms to further improve interpretability and trust.

This work was completed as the **final project for IS 755 – Intelligent Systems** at the **University of Maryland, Baltimore County (UMBC)**.

---

## 👥 Authors

- **Aryan Jagani** – University of Maryland, Baltimore County  
- **Vishvakumar Patel** – University of Maryland, Baltimore County  

---

## 🎯 Problem Statement

Although CNN-based models such as ResNet-50 demonstrate strong performance in skin lesion classification, they are often treated as **black boxes**. Existing explanation methods like Grad-CAM frequently produce **coarse and diffuse heatmaps**, which may highlight irrelevant regions and reduce clinical trust.

Key challenges addressed:
- Lack of intrinsic interpretability in deep learning models
- Coarse visual explanations in standard Grad-CAM
- Limited quantitative evaluation of explanation quality
- Poorly calibrated confidence in clinical AI systems

---

## 🧠 Hypotheses

- **Primary Hypothesis:**  
  Advanced CAM-based methods (Grad-CAM++, Score-CAM) provide more precise and clinically relevant localization than standard Grad-CAM.

- **Secondary Hypothesis:**  
  Combining visual explanations with concise text-based summaries improves interpretability and perceived trustworthiness.

- **Performance Hypothesis:**  
  Incorporating explainability does not significantly degrade classification accuracy.

---

## 📂 Dataset

- **Dataset:** ISIC 2019 Skin Lesion Dataset  
- **Samples:** 25,331 dermoscopic images  
- **Classes (9):**  
  MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK  
- **Metadata:** Age, sex, anatomical site  
- **Challenges:** Strong class imbalance (NV, BKL dominant)

---

## 🔧 Methodology

### 1. Data Preprocessing
- Image resizing to **224 × 224**
- Normalization using ImageNet statistics
- Data augmentation (training only):
  - Random resized crops
  - Horizontal/vertical flips
  - Color jitter
  - Random rotation
- Stratified **80/20 train–validation split**

### 2. Model Architecture
- **Backbone:** ResNet-50 (ImageNet pretrained)
- Final fully connected layer replaced for 9-class classification
- End-to-end fine-tuning

### 3. Training Setup
- Optimizer: Adam (lr = 1e-4)
- Loss: Cross-Entropy
- Batch size: 64
- Epochs: 10
- Mixed Precision Training (AMP)
- Environment: Google Colab GPU

---

## 🔍 Explainability Techniques (XAI)

We implemented and compared the following CAM-based methods:

- **Grad-CAM**  
  Gradient-weighted class activation maps (baseline)

- **Grad-CAM++**  
  Uses higher-order gradients for improved localization and multi-instance focus

- **Score-CAM** *(planned/partially explored)*  
  Gradient-free, perturbation-based explanation method

### Layer-wise Analysis
Explanations were generated from:
- Early layers (texture-level features)
- Intermediate layers (lesion structure)
- Deep layers (clinically relevant localization)

---

## 📝 Text-Based Explanation Engine

To complement visual explanations, we designed a rule-based text generator that summarizes:
- Region size (small / medium / large)
- Location (upper/lower, left/right)
- Shape (round / elongated)
- Color intensity
- Boundary properties

These summaries accompany the predicted class and confidence score, making explanations more accessible to non-technical users.

---

## 📊 Results

### Classification Performance
- **Training Accuracy:** 81%
- **Validation Accuracy:** 82%
- Best performance on frequent classes (NV, BKL)
- Reduced performance on rare malignant classes (MEL, SCC)

### Explainability Findings
- Grad-CAM++ produces **sharper, more lesion-focused heatmaps** than Grad-CAM
- Deep-layer explanations are the most clinically meaningful
- Text-based summaries enhance interpretability beyond visual inspection alone

---

## ⚠️ Limitations

- Class imbalance affects minority class performance
- CAM methods are post-hoc and not causally guaranteed
- Quantitative XAI metrics (IoU, pointing game, deletion curves) are outlined but not fully implemented
- Metadata features were not fully leveraged

---

## 🚀 Future Work

- Integrate EfficientNet and Vision Transformer architectures
- Improve minority class performance via reweighting and oversampling
- Fully implement quantitative XAI metrics
- Add model calibration analysis
- Develop an interactive Streamlit-based visualization tool for clinicians

---

## 🛠 Tools & Technologies

- Python
- PyTorch
- ResNet-50
- pytorch-grad-cam
- NumPy, Pandas
- Matplotlib
- Google Colab (GPU)

---


---

## 📄 Course Information

- **Course:** IS 755 – Advanced AI
- **Institution:** University of Maryland, Baltimore County  

---

## 👤 Author Contribution

**Aryan Jagani**
- Model architecture & training pipeline
- Grad-CAM++ implementation
- Literature review and XAI comparison
- Integration of text-based explanations

**Vishvakumar Patel**
- Dataset preprocessing and augmentation
- Explainability metrics planning
- Robustness analysis support

---

## 📚 References

Key references include Grad-CAM, Grad-CAM++, Score-CAM, and ISIC dataset publications.  
See the full report for a detailed bibliography.

---


