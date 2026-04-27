# Homomorphic Encryption ML Pipeline - Diabetes Classification

## Project Overview

This project demonstrates **Machine Learning inference under Homomorphic Encryption (HE)** using the CKKS scheme in TenSEAL. It compares plaintext and encrypted inference for two linear ML models on the UCI Diabetes dataset, highlighting the performance trade-offs between privacy and speed.

---

## Project Architecture

### Step 1: Load & Prepare Data
**What's happening:**
- Load the UCI Diabetes dataset (diabetes.csv)
- Features: 8 medical attributes (glucose, BMI, blood pressure, etc.)
- Target: Binary classification (diabetes diagnosis: 0 or 1)
- Split: 80% training, 20% testing
- Normalize: StandardScaler to zero-mean, unit-variance (required for HE numerical stability)

**Why normalization matters:**
CKKS encryption works best with values in a small range. Scaling prevents numerical errors during homomorphic operations.

---

### Step 2: Train Models on Plaintext
**What's happening:**
- Train **Logistic Regression** on plaintext (unencrypted) data
- Train **SVM with Linear Kernel** on plaintext data
- Store weights and bias terms for later use in encrypted inference

**Why these models?**
Both use **linear decision boundaries** of the form:
$$\text{prediction} = \text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$$

This simplicity is crucial for HE compatibility.

---

### Step 3: Plaintext Inference & Timing
**What's happening:**
- Run predictions on test data using both models
- Measure inference speed (baseline performance)
- Calculate accuracy and F1-score (ground truth comparison)

**Metrics captured:**
- **Accuracy**: Percentage of correct predictions
- **F1-Score**: Harmonic mean of precision and recall (better for imbalanced data)
- **Time**: Inference latency in seconds

---

### Step 4: Setup CKKS Context
**What's happening:**
- Initialize CKKS (Cheon-Kim-Kim-Song) cryptographic scheme
- Parameters:
  - `poly_mod_degree = 8192`: Polynomial modulus degree (higher = more security, slower)
  - `coeff_mod_bit_sizes`: Bit-widths for coefficient moduli (controls precision & depth)
- Generate key material:
  - **Galois keys**: Enable rotations and permutations
  - **Relin keys**: Enable relinearization after multiplication

**Why these parameters?**
- `poly_mod_degree=8192`: Balances 128-bit security with computational feasibility
- `[40, 21, 21, 21, 21, 21, 21, 21, 21, 40]`: 9 levels of multiplication depth
  - Top/bottom (40-bit) layers for initialization/cleanup
  - Middle layers (21-bit) for homomorphic operations
  - More levels = more complex computations possible, but slower

---

### Step 5: Homomorphic Inference for Logistic Regression
**What's happening:**
- Encrypt test data and model weights under CKKS
- Perform **encrypted dot product**: $\mathbf{w} \cdot \mathbf{x}$
- Add encrypted bias: $\mathbf{w} \cdot \mathbf{x} + b$
- Apply **sigmoid approximation** (encrypted)
- Decrypt and threshold at 0.5 for classification

**Sigmoid Approximation:**
$$\sigma(x) \approx 0.5 + 0.1975x$$

Linear approximation around $x=0$. This avoids expensive polynomial evaluations that consume multiplication depth.

**Decryption:**
Only the data owner (with secret key) decrypts results. Server never sees plaintext data.

---

### Step 6: Homomorphic Inference for SVM
**What's happening:**
- Similar to Logistic Regression, but simpler
- No sigmoid needed: use SVM decision boundary directly
- Threshold at 0 (instead of 0.5)
- Encrypted computation: $\mathbf{w} \cdot \mathbf{x} + b > 0$?

**Why SVM works well:**
- Fewer operations (no sigmoid approximation)
- Faster encrypted inference
- Same linear structure as Logistic Regression

---

## Why Certain Models Don't Work with CKKS

### ❌ Random Forest (Decision Trees)
**The Problem:** Decision trees require **comparison operations** and **conditional branching**
```
**Why it fails:**
- CKKS supports: `+, -, ×, ≤ (limited)`
- CKKS does NOT support: `>, <, if-then-else` on encrypted data
- Would require **homomorphic comparison circuits**, which are impractical for CKKS
- Alternative (BFV with bootstrapping) is 1000x slower

---

### ❌ Neural Networks (ReLU Activation)
**The Problem:** ReLU requires non-linear operations
$$\text{ReLU}(x) = \max(0, x)$$

**Why it fails:**
- CKKS supports polynomial operations (multiplication)
- ReLU is **piecewise linear** (non-smooth)
- Approximating ReLU requires high-degree polynomials
- Consumes all available multiplication depth very quickly

**Workaround:** Use linear activations (not practical for deep networks)

---

### ❌ Gradient Boosting (XGBoost, LightGBM)
**The Problem:** Like Random Forests, relies on trees and comparisons
- Sequential decision trees
- Threshold-based splits
- Can't be evaluated homomorphically

---

### ❌ K-Means Clustering
**The Problem:** Requires distance comparisons (argmin)

---

## ✅ Why Linear Models Work

### Logistic Regression
- Only matrix multiplication and addition
- No comparisons on encrypted data
- Sigmoid approximation keeps depth low

### SVM (Linear Kernel)
- Even simpler than Logistic Regression
- Single dot product per sample
- Minimal encrypted operations

---

## Homomorphic Encryption Trade-offs

| Aspect | Plaintext | Encrypted |
|--------|-----------|-----------|
| **Speed** | ~0.0005s/sample | ~1s/sample |
| **Accuracy** | 77.9% | 78.0% (similar) |
| **Privacy** | None | Full (data encrypted) |
| **Computation** | Server | Server (on ciphertexts) |
| **Operations** | Any | Linear only |

**Key Insight:** Privacy costs ~1000x slowdown for linear models. Nonlinear models cost ~10000x+.

---

## Technical Concepts

### CKKS Scheme
- **Purpose:** Approximate encrypted arithmetic over real numbers
- **Advantages:** Efficient for linear operations, reasonable accuracy
- **Disadvantages:** Limited to polynomials, doesn't support comparisons natively

### Modulus Switching Depth
Each multiplication "uses up" one level:
````
This is the code block that represents the suggested code change:
````
More multiplications = need deeper modulus chain = slower encryption/decryption.
````
### Why We Can't Compare on Encrypted Data
Comparison requires examining bit-level structure:
```python
# Plaintext (easy):
if x > y:
    # examine actual values
    
# Encrypted (impossible):
if E(x) > E(y):
    # don't know x or y!
    # Server cannot compare without decrypting
````
`````
This is the code block that represents the suggested code change:
```markdown
=== Plaintext vs Encrypted Inference Comparison ===
Model              Plaintext Acc  Plaintext F1  Plaintext Time  Encrypted Acc  Encrypted F1  Encrypted Time
Logistic Regression      0.779221      0.653061          0.000044       0.778701      0.652740        0.991936
SVM (Linear)             0.772727      0.646465          0.001976       0.774026      0.648936        0.981452
```