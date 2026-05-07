---
marp: true
theme: default
class: list
paginate: true
header: 'CS541 AI Project'
footer: 'Homomorphic Encryption in Machine Learning'
---

# Privacy-Preserving Machine Learning
## Exploring Homomorphic Encryption for Medical Diagnosis
**CS541 AI Project**

---

## Project Overview

- **Goal**: Evaluate the performance and viability of **Homomorphic Encryption (HE)** in practical machine learning tasks.
- **Approach**: Compare plaintext (traditional) ML inference against encrypted inference.
- **Models**: Logistic Regression and Support Vector Machines (Linear).
- **Tooling**: `scikit-learn` for plaintext training and `TenSEAL` for HE inference.

---

## The Dataset

**UCI Diabetes Dataset**
- **Objective**: Binary classification (predicting the presence or absence of diabetes).
- **Features**: 8 medical attributes (e.g., glucose, blood pressure, BMI, age).
- **Scale**: ~768 samples (614 training, 154 testing).
- **Preprocessing**: StandardScaler normalization (crucial for managing values prior to encryption).

---

## What is Homomorphic Encryption (HE)?

- **Definition**: A cryptographic technique that allows mathematical operations to be performed directly on *encrypted* data without decrypting it first.
- **Benefit**: Unlocks privacy-preserving ML (e.g., a hospital can send encrypted patient data to a cloud API that returns an encrypted diagnosis, without the cloud ever seeing the raw data).
- **The Catch**: HE adds noise to the data and is exceptionally computationally expensive.

---

## The CKKS Scheme

To support the fractional math required by ML, we use the **CKKS (Cheon-Kim-Kim-Song)** scheme:
- Supports addition and multiplication of approximate numbers.
- Requires scaling to handle fractional arithmetic.
- Noise grows with each operation.

We test two encryption parameter variants to see the effect of precision vs. noise:
1. **CKKS (8192)**: High precision, larger polynomial degree.
2. **CKKS (4096)**: Lower precision, faster but much noisier.

---

## Methodology

1. **Train**: Train ML models (Logistic Regression, Linear SVM) on *plaintext* training data.
2. **Setup**: Create CKKS contexts with predefined polynomial moduli and scales.
3. **Calibrate**: Compute a decision boundary threshold in the encrypted domain using a subset of training data to offset homomorphic noise.
4. **Inference**: Perform dot product operations (`Weights · Features + Bias`) mathematically in the ciphertext space.
5. **Compare**: Measure Accuracy, F1-Score, and Time.

---

## Why Linear Models?

You might wonder why we aren't using deep Neural Networks with complex activations:
- Non-linear activation functions (like ReLU or exact Sigmoid) are extremely difficult and expensive to compute natively in the encrypted domain.
- **Linear models** only require dot products (multiplications and additions) and a final threshold mapping, making them highly compatible with current HE schemes.

---

## Security vs. Accuracy vs. Speed

The fundamental trade-off observed in the project:
- **Plaintext**: Fast inference (< 1 ms), perfect baseline accuracy.
- **CKKS 8192**: Retains accuracy near the plaintext baseline, but inference is exponentially slower (~0.5 - 1.0 seconds per sample).
- **CKKS 4096**: Faster than 8192, but lower "noise budget" results in heavy signal loss and flatlined learning curves.

*HE operations are typically 1,000x to 10,000x slower than plaintext operations.*

---

## Key Conclusions

- **Homomorphic Encryption is viable** for privacy-critical, latency-tolerant medical analysis.
- **Noise management is critical**: If parameters aren't large enough, encryption noise swallows the ML signal, destroying predictions.
- **Future Work**: Requires hardware acceleration or highly optimized approximate HE schemes to be practical for real-time inference.
