# Project.py Documentation

## Overview

`Project.py` is a comprehensive machine learning pipeline that demonstrates **Homomorphic Encryption (HE)** applied to medical classification tasks. The project compares the performance of two ML models (Logistic Regression and SVM) across three execution environments:

1. **Plaintext** - Standard unencrypted inference (baseline)
2. **CKKS (8192)** - High-precision encrypted inference
3. **CKKS (4096)** - Lower-precision encrypted inference

The comparison evaluates **accuracy, F1 score, execution time, and learning curves** to assess the practical viability of HE in machine learning applications.

---

## Dataset

- **Source**: UCI Diabetes Dataset (`diabetes.csv`)
- **Task**: Binary classification (Diabetes presence vs. absence)
- **Features**: 8 medical attributes (glucose, blood pressure, BMI, etc.)
- **Samples**: ~768 total (614 train, 154 test at 80/20 split)
- **Preprocessing**: StandardScaler normalization

---

## Dependencies

```
pandas
numpy
scikit-learn (LogisticRegression, SVC, train_test_split, StandardScaler, metrics)
tenseal      # Homomorphic Encryption library
matplotlib   # Visualization
```

Install via:
```bash
pip install pandas numpy scikit-learn tenseal matplotlib
```

---

## Architecture

### Section 1: Data Loading & Preprocessing (Lines 1-24)

```python
df = pd.read_csv('Project/diabetes.csv')
X_train, X_test = train_test_split(...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
```

- Loads diabetes dataset and splits into 80% training, 20% testing
- Applies StandardScaler to normalize features (mean=0, std=1)
- **Critical**: Scaling must happen before encryption to keep values manageable

---

### Section 2: Plaintext Model Training (Lines 26-43)

```python
lr = LogisticRegression(max_iter=1000, random_state=13)
svm = SVC(kernel='linear', random_state=13)
```

- Trains two linear models on plaintext data
- **Logistic Regression**: Probabilistic classifier with learned weights
- **SVM (Linear Kernel)**: Maximum margin classifier
- Both are compatible with HE because they only require dot products + bias (no non-linear activations)

**Why Linear Models?** Non-linear activation functions (ReLU, sigmoid) are expensive to compute in the encrypted domain and introduce significant noise/performance penalties.

---

### Section 3: CKKS Context Setup (Lines 45-72)

#### CKKS (8192) - High Precision
```python
poly_mod_degree_8192 = 8192
coeff_mod_bit_sizes_8192 = [60, 40, 40, 60]
enc_training_ckks_8192.global_scale = 2 ** 40
```

- **Polynomial Degree (N=8192)**: Larger N = higher capacity and precision, but slower computation
- **Coefficient Moduli [60, 40, 40, 60]**: Larger bit sizes provide more "room" for numbers before overflow
- **Global Scale (2^40)**: Fractional precision in encrypted arithmetic (~1 trillion scale factor)

#### CKKS (4096) - Lower Precision
```python
poly_mod_degree_4096 = 4096
coeff_mod_bit_sizes_4096 = [40, 21, 21, 21, 21, 40]
enc_training_ckks_4096.global_scale = 2 ** 20
```

- **Polynomial Degree (N=4096)**: Half the capacity of 8192, faster but noisier
- **Smaller coefficient moduli**: Less precision available
- **Lower scale (2^20)**: Reduced fractional precision (~1 million scale factor)

**Purpose**: This creates an intentional "precision gap" to demonstrate how HE performance degrades with reduced security margins.

---

### Section 4: Threshold Calibration (Lines 74-93)

```python
threshold_lr_ckks = np.mean(y_lin_enc_train_lr)
```

- Computes the encrypted predictions on 50 training samples
- Takes the **mean** as the decision boundary
- **Why?** In HE, decryption can introduce minor noise. We calibrate the threshold to the encrypted domain to ensure fair comparison.

---

### Section 5 & 6: Homomorphic Inference (Lines 95-165)

For each model (LR/SVM) and context (8192/4096):

```python
sample_enc = ts.ckks_vector(enc_training_ckks_8192, X_test_scaled[i])
y_lin_enc = sample_enc.dot(lr_weights) + lr_bias  # Encrypted dot product
y_lin_dec = y_lin_enc.decrypt()                    # Decrypt result
y_pred = 1 if y_lin_dec[0] > threshold else 0     # Thresholded classification
```

- **ts.ckks_vector()**: Encrypts plaintext feature vector
- **.dot()**: Computes encrypted dot product (weights · features)
- **decrypt()**: Returns ciphertext to plaintext (only private key can do this)
- All operations are **mathematically equivalent** to plaintext computation, but with added noise

---

### Section 7: Results Storage & Display (Lines 167-223)

Stores metrics in nested dictionaries:
```python
metrics_he = {
    'CKKS (8192)': {
        'Logistic Regression': {'acc': ..., 'f1': ..., 'time': ...},
        'SVM (Linear)': {...}
    },
    'CKKS (4096)': {...}
}
```

Displays a comparison table showing all 6 model/context combinations.

---

### Section 8: Learning Curves (Lines 228-369)

Iterates over 5 training set sizes (20%, 40%, 60%, 80%, 100%):

1. **Trains new models** on each subset
2. **Calibrates new thresholds** for both CKKS contexts (separate calibration per subset)
3. **Evaluates all 6 combinations** on the full test set
4. **Stores F1 scores** to track how performance changes with training data volume

**Key Insight**: The learning curves reveal whether HE models can improve with more training data or if they're fundamentally limited by encryption noise.

---

### Section 9: Visualization (Lines 371-429)

Generates a 2×2 subplot figure:

| Plot | Description |
|------|-------------|
| **[0,0]** | Logistic Regression learning curve across all contexts |
| **[0,1]** | SVM learning curve across all contexts |
| **[1,0]** | F1 score bar chart (full training data) |
| **[1,1]** | Inference timing comparison (log scale) |

---

## Key Concepts

### Homomorphic Encryption (HE)

Allows computation on encrypted data **without decrypting it**:
- Plaintext: `z = a·x + b`
- Encrypted: `Enc(z) = Enc(a)·Enc(x) + Enc(b)` ✓ Same result when decrypted

### CKKS Scheme

- **Approximate arithmetic** (slight rounding errors)
- **Supports addition and multiplication** on encrypted numbers
- **Scales by 2^p** to handle fractional arithmetic
- **Noise grows** with each operation; eventually swallows signal

### Decision Boundary in HE

In plaintext: `decision = (w·x + b) > 0`  
In HE: Same computation, but `(w·x + b)` contains noise that can flip predictions near the decision boundary.

---

## Running the Code

```bash
cd /Users/m01010100/Documents/CS541_AI/Project
python Project.py
```

**Expected Runtime**: 10-15 minutes (HE operations are computationally expensive)

**Output**:
1. Console table comparing all models
2. Matplotlib figure with 4 subplots
3. Progress messages showing learning curve evaluation

---

## Interpreting Results

### Performance Gap Analysis

Plaintext vs. CKKS performance gaps indicate:
- **8192 > 4096**: Higher polynomial degree reduces noise
- **Small gap (<5%)**: HE viable for this task
- **Large gap (>20%)**: Noise floor dominates; dataset/model not suitable for HE

### Learning Curves

- **Plaintext curves**: Should monotonically increase (more data = better)
- **HE curves (flat)**: Noise prevents improvement; limited by precision, not data
- **HE curves (rising)**: HE can still learn; precision sufficient to capture signal

### Timing Comparison

- Plaintext: <1 millisecond per sample
- HE: 500ms - 1s per sample
- Speedup needed: ~1000-10000x to make HE practical for real-time inference

---

## Configuration Parameters

### To Improve HE Accuracy

1. **Increase polynomial degree**:
   ```python
   poly_mod_degree_8192 = 16384  # Double capacity
   coeff_mod_bit_sizes_8192 = [60, 40, 40, 40, 40, 40, 40, 60]
   ```

2. **Increase global scale**:
   ```python
   enc_training_ckks_8192.global_scale = 2 ** 50  # More precision
   ```

3. **Improve dataset**:
   - Use MinMaxScaler instead of StandardScaler
   - Apply PCA to reduce dimensionality
   - Choose datasets with larger class margins

### To Speed Up Inference

1. **Reduce polynomial degree** (sacrifices precision)
2. **Use batching** (encrypt multiple samples at once)
3. **Implement approximate HE** (lower security for speed)

---

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Terrible HE accuracy (50%) | Scale too small; noise dominates | Increase `global_scale` |
| CKKS (4096) flat line | Precision insufficient | Use higher polynomial degree |
| Out of memory | Large ciphertext | Reduce `poly_mod_degree` |
| Negative accuracy | Threshold miscalibration | Re-run calibration loop |
| Extreme slowdown | Too many multiplications | Reduce feature count (PCA) |

---

## Mathematical Details

### CKKS Encoding

Plaintext value `x` → Encoded as: `m = round(x × 2^p)` where `p = log2(scale)`

### Encryption Noise

After `k` multiplications:
- Noise magnitude ≈ $N × 2^k$ (N = context-dependent constant)
- Signal magnitude ≈ value being encrypted
- Signal lost when: Noise > Signal

---

## References

- **TenSEAL**: https://github.com/OpenMined/TenSEAL
- **CKKS Scheme**: Cheon et al. (2017) "Homomorphic Encryption for Arithmetic of Approximate Numbers"
- **Diabetes Dataset**: UCI ML Repository

---

## Author Notes

This project demonstrates the **fundamental tradeoff in HE**:
- **Security**: Data stays encrypted; server never sees plaintext
- **Accuracy**: Noise from encryption reduces model performance
- **Speed**: Encrypted operations 1000x+ slower than plaintext

HE is ideal for **privacy-critical, latency-tolerant** applications (e.g., outsourced analysis of sensitive medical data). For real-time inference, traditional encryption (TLS) + secure enclaves remain superior.
