# Presenter Notes: Equation Explanations

This document provides a detailed breakdown of the mathematical equations presented in the slide deck. Use these notes to explain the underlying mechanics of the models and the Homomorphic Encryption (HE) adaptations during your presentation.

---

## Slide 2: Theoretical Background: Logistic Regression

### Equation 1: The Hypothesis Function
$$ h_\theta(\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}} $$

**How to explain it:**
*   **$\mathbf{x}$**: This is the input feature vector (e.g., patient health metrics like BMI, glucose levels, etc.).
*   **$\mathbf{w}^T$**: This represents the transposed weight vector learned during plaintext training. 
*   **$\mathbf{w}^T \mathbf{x} + b$**: This is the core linear combination—the dot product of the weights and the input features, shifted by the bias term $b$. It outputs a single scalar value.
*   **$\sigma$ (Sigmoid Function)**: The mathematical function $\frac{1}{1 + e^{-z}}$. Its purpose is to take any real-valued number and "squash" it into a range between 0 and 1. This converts our raw linear output into a probability score (e.g., 0.82 means an 82% predicted chance of diabetes).

### Equation 2: The HE Sigmoid Approximation
$$ \sigma(x) \approx 0.5 + 0.1975x $$

**How to explain it:**
*   **The Problem:** The CKKS Homomorphic Encryption scheme only supports addition, subtraction, and multiplication natively. It cannot compute division or exponentiation ($e^{-x}$), meaning we cannot calculate the exact sigmoid function on encrypted data.
*   **The Solution:** We use a simple linear polynomial approximation. 
*   **The Components:**
    *   At $x = 0$, the standard sigmoid function evaluates exactly to $0.5$. Our approximation captures this intercept correctly.
    *   The $0.1975$ is the slope (derived via minimax polynomial approximation or related methods) that best fits the curve near the center values of the dataset.
*   **Why it works here:** Because we standardized our data (mean 0, variance 1) during preprocessing, most of the values for $\mathbf{w}^T \mathbf{x} + b$ will fall near $0$, right where this approximation is the most accurate. It also requires only a single scalar multiplication, saving precious HE multiplication depth.

---

## Slide 3: Theoretical Background: Support Vector Machine (Linear)

### Equation 3: The Hypothesis Function
$$ h(\mathbf{x}) = \text{sign}(\mathbf{w}^T \mathbf{x} + b) $$

**How to explain it:**
*   Unlike Logistic Regression, an SVM doesn't output a probability; it outputs a strict class assignment (often mapped to +1 or -1, or 1 and 0 in our binary case).
*   **$\mathbf{w}^T \mathbf{x} + b$**: This defines a hyperplane (a geometric boundary) separating the two classes in our multi-dimensional feature space. The result of this calculation is the distance measuring how far the point $\mathbf{x}$ is from that boundary.
*   **$\text{sign}(\cdot)$**: This simply evaluates whether the distance is mathematically positive or negative.

### Equation 4: The Decision Rule
*   If $\mathbf{w}^T \mathbf{x} + b > 0$, predict Class 1.
*   Otherwise, predict Class 0.

**How to explain it:**
*   This is the practical execution of the `sign` function.
*   **HE Context:** In Homomorphic Encryption, we calculate the dot product $\mathbf{w}^T \mathbf{x} + b$ purely in the encrypted domain. The server holding the encrypted data never runs the "`> 0`" check, because comparison operations are not natively supported in CKKS without incredibly expensive boolean logic circuits. 
*   Instead, the server sends the encrypted result back to the client. The client decrypts this single scalar value and then performs the `> 0` check in plaintext. This maintains the computation-privacy while keeping the model blistering fast compared to encrypted tree-based models.