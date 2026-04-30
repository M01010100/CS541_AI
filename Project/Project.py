import pandas as pd
import numpy as np
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time

# 1. Load & Prepare Data
df = pd.read_csv('Project/diabetes.csv')
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Models on Plaintext
lr = LogisticRegression(max_iter=1000, random_state=13)
svm = SVC(kernel='linear', random_state=13)  # Linear SVM for HE compatibility

lr.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

# 3. Plaintext Inference & Timing
def time_predict(model, X):
    start = time.time()
    y_pred = model.predict(X)
    return y_pred, time.time() - start

y_pred_lr_pt, time_lr_pt = time_predict(lr, X_test_scaled)
y_pred_svm_pt, time_svm_pt = time_predict(svm, X_test_scaled)

# Plaintext Metrics
metrics_pt = {
    'Logistic Regression': {'acc': accuracy_score(y_test, y_pred_lr_pt), 'f1': f1_score(y_test, y_pred_lr_pt), 'time': time_lr_pt},
    'SVM (Linear)': {'acc': accuracy_score(y_test, y_pred_svm_pt), 'f1': f1_score(y_test, y_pred_svm_pt), 'time': time_svm_pt}
}

# 4. Setup CKKS Context
poly_mod_degree = 8192
coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 21, 21, 40]

enc_training = ts.context(
    ts.SCHEME_TYPE.CKKS, 
    poly_mod_degree, 
    coeff_mod_bit_sizes, 
)
enc_training.global_scale = 2 ** 21
enc_training.generate_galois_keys()
enc_training.generate_relin_keys()

# 5. Homomorphic Inference for Logistic Regression
lr_weights = np.array(lr.coef_[0]).tolist()  # Convert to list explicitly
lr_bias = lr.intercept_[0]

start_he_lr = time.perf_counter()
y_pred_lr_he = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(lr_weights) + lr_bias
    
    x_norm = y_lin_enc * 0.1
    y_sigmoid_enc = 0.5 + 0.1975 * x_norm
    
    y_sigmoid_dec = y_sigmoid_enc.decrypt()
    y_pred_lr_he.append(1 if y_sigmoid_dec[0] > 0.5 else 0)

y_pred_lr_he = np.array(y_pred_lr_he)
time_lr_he = time.perf_counter() - start_he_lr

# 6. Homomorphic Inference for SVM
svm_weights = svm.coef_[0]
svm_bias = svm.intercept_[0]

start_he_svm = time.perf_counter()
y_pred_svm_he = []
svm_dec_values = []

for i in range(X_test_scaled.shape[0]):
    sample_plain = X_test_scaled[i]
    
    # Encrypt the sample
    sample_enc = ts.ckks_vector(enc_training, sample_plain.tolist())
    
    # Use .dot() directly
    y_lin_enc = sample_enc.dot(svm_weights.tolist())
    
    y_dot_dec = y_lin_enc.decrypt()
    dot_val = y_dot_dec[0] if isinstance(y_dot_dec, list) else y_dot_dec
    
    # Add bias
    y_lin_result = dot_val + svm_bias
    
    svm_dec_values.append(y_lin_result)
    y_pred_svm_he.append(1 if y_lin_result > 0 else 0)

y_pred_svm_he = np.array(y_pred_svm_he)
time_svm_he = time.perf_counter() - start_he_svm

print(f"Plaintext dot products (first 5): {[np.dot(X_test_scaled[i], svm_weights) for i in range(5)]}")
print(f"Encrypted dot products (first 5): {[svm_dec_values[i] - svm_bias for i in range(5)]}")
print(f"Plaintext decision (first 5): {svm.decision_function(X_test_scaled)[:5]}")
print(f"Encrypted prediction (first 5): {svm_dec_values[:5]}")

# DEBUG: Print statistics
print(f"SVM decrypted values - Min: {min(svm_dec_values):.6f}, Max: {max(svm_dec_values):.6f}, Mean: {np.mean(svm_dec_values):.6f}")
print(f"SVM plaintext decision values (first 5): {svm.decision_function(X_test_scaled)[:5]}")
print(f"SVM encrypted decision values (first 5): {svm_dec_values[:5]}")

# DEBUG: Compare
print(f"Plaintext SVM bias: {svm_bias}")
print(f"Encrypted dot products (before bias) - first 5: {svm_dec_values[:5]}")
print(f"Encrypted values (after bias) - first 5: {svm_dec_values[:5]}")
print(f"Plaintext decision function - first 5: {svm.decision_function(X_test_scaled)[:5]}")

metrics_he = {
    'Logistic Regression': {'acc': accuracy_score(y_test, y_pred_lr_he), 'f1': f1_score(y_test, y_pred_lr_he), 'time': time_lr_he},
    'SVM (Linear)': {'acc': accuracy_score(y_test, y_pred_svm_he), 'f1': f1_score(y_test, y_pred_svm_he), 'time': time_svm_he}
}

# 7. Comparison Matrix
comparison = pd.DataFrame([
    {
        'Model': 'Logistic Regression',
        'Plaintext Acc': metrics_pt['Logistic Regression']['acc'],
        'Plaintext F1': metrics_pt['Logistic Regression']['f1'],
        'Plaintext Time (s)': metrics_pt['Logistic Regression']['time'],
        'Encrypted Acc': metrics_he['Logistic Regression']['acc'],
        'Encrypted F1': metrics_he['Logistic Regression']['f1'],
        'Encrypted Time (s)': metrics_he['Logistic Regression']['time']
    },
    {
        'Model': 'SVM (Linear)',
        'Plaintext Acc': metrics_pt['SVM (Linear)']['acc'],
        'Plaintext F1': metrics_pt['SVM (Linear)']['f1'],
        'Plaintext Time (s)': metrics_pt['SVM (Linear)']['time'],
        'Encrypted Acc': metrics_he['SVM (Linear)']['acc'],
        'Encrypted F1': metrics_he['SVM (Linear)']['f1'],
        'Encrypted Time (s)': metrics_he['SVM (Linear)']['time']
    }
])

print("=== Plaintext vs Encrypted Inference Comparison ===")
print(comparison.to_string(index=False))