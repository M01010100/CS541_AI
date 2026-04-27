import os

slides_content = r"""---
marp: true
theme: default
paginate: true
math: mathjax
---

<style>
h2 {
  border-bottom: 4px solid #0056b3;    /* Adds a blue color bar */
  padding-bottom: 10px;                /* Spacing between text and bar */
  margin-bottom: 30px;                 /* Spacing below the bar */
}
</style>

# Machine Learning Inference under Homomorphic Encryption
## Diabetes Classification using CKKS

**CS541 AI Project**

---

## 1. Introduction and Motivation

- **Privacy in Machine Learning:** Health data (e.g., patient records) is highly sensitive. Sharing it with external ML services risks data breaches.
- **Homomorphic Encryption (HE):** A cryptographic approach that allows computations on ciphertext without decrypting it first.
- **Goal:** Compare plaintext and HE-encrypted inference speeds and accuracies using the UCI Diabetes dataset.
- **Models Evaluated:** Logistic Regression and Support Vector Machines (Linear Kernel).

---

## 2. Theoretical Background: Logistic Regression

- **Hypothesis Function:** 
  For an input vector $\mathbf{x}$, the output probability is given by the sigmoid function:
  $$ h_\theta(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}} $$
- **HE Challenge:** 
  The standard sigmoid involves non-linear operations (exponentiation and division) which are prohibitively expensive or unsupported in many HE schemes.
- **HE Solution:** 
  Linear approximation around $x=0$:
  $$ \sigma(x) \approx 0.5 + 0.1975x $$
  This simple polynomial form requires minimal multiplication depth.

---

## 3. Theoretical Background: Support Vector Machine (Linear)

- **Hypothesis Function:** 
  A linear discriminant function without probabilities:
  $$ h(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b) $$
- **Decision Rule:**
  - If $\mathbf{w}^T \mathbf{x} + b > 0$, predict Class 1.
  - Otherwise, predict Class 0.
- **HE Advantage:** 
  SVM using a linear kernel is extremely well-suited for HE since it solely relies on the dot product, requiring just one level of polynomial multiplication and addition.

---

## 4. The CKKS Encryption Scheme

- **Cheon-Kim-Kim-Song (CKKS):** An HE scheme optimized for approximate arithmetic over continuous values (floating-point numbers).
- **Core Operations Supported:** Addition, Subtraction, and Multiplication.
- **Key Parameters in this Project:**
  - `poly_mod_degree = 8192`: Determines the ring dimension (balances security vs. performance).
  - `coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 21, 21, 40]`: Multiplication depth.
  - `global_scale = 2^{21}`: Controls precision.
- **Limitation:** Non-linear operations like ReLU ($\max(0, x)$), decision trees, and value comparisons cannot be evaluated efficiently.

---

## 5. Experimental Setup

- **Dataset:** UCI Diabetes Classification (binary outcome).
- **Data Preprocessing:** Standard scaling (mean $0$, variance $1$) applied to features. This ensures data is centered, maximizing the accuracy of the polynomial sigmoid approximation and maintaining numerical stability in CKKS.
- **Implementation:** Python using `scikit-learn` for baseline plaintext models and `TenSEAL` for the HE pipeline.

---

## 6. Results and Observations

### Inference Comparison
| Model | Plaintext Acc | Plaintext F1 | Encrypted Acc | Encrypted F1 | Time (PT) | Time (Enc) |
|-------|---------------|--------------|---------------|--------------|-----------|------------|
| LogReg| ~77.9%        | ~65.3%       | ~77.9%        | ~65.3%       | 0.000s    | ~1.0s      |
| SVM   | ~77.3%        | ~64.6%       | ~77.4%        | ~64.9%       | 0.002s    | ~0.98s     |

- **Accuracy Preserved:** Encrypted inference produces nearly identical performance to plaintext.
- **Performance Trade-off:** Encrypted inference is ~1000-5000x slower per sample.

---

## 7. Conclusion

- HE (via CKKS) effectively protects patient privacy while maintaining ML diagnostic accuracy.
- Linear models (Logistic Regression, SVM) are highly compatible with CKKS, avoiding unsupported comparisons.
- The principal barrier remains computational latency, though scaling mechanisms and optimized polynomial approximations provide viable paths forward.

"""

if __name__ == "__main__":
    with open("Project/SLIDES.md", "w") as f:
        f.write(slides_content)
    print("Slides successfully generated at 'Project/SLIDES.md'.")
    print("You can view the slides using Marp (https://marp.app/) or a compatible Markdown presentation viewer.")