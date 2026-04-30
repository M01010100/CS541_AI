import pandas as pd
import numpy as np
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
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
ridge = Ridge(alpha=1.0, random_state=13)  # Ridge Regression

lr.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)
ridge.fit(X_train_scaled, y_train)

# 3. Plaintext Inference & Timing
def time_predict(model, X):
    start = time.time()
    y_pred = model.predict(X)
    return y_pred, time.time() - start

y_pred_lr_pt, time_lr_pt = time_predict(lr, X_test_scaled)
y_pred_svm_pt, time_svm_pt = time_predict(svm, X_test_scaled)
ridge_pred_pt, time_ridge_pt = time_predict(ridge, X_test_scaled)

# Plaintext Metrics
metrics_pt = {
    'Logistic Regression': {'acc': accuracy_score(y_test, y_pred_lr_pt), 'f1': f1_score(y_test, y_pred_lr_pt), 'time': time_lr_pt},
    'SVM (Linear)': {'acc': accuracy_score(y_test, y_pred_svm_pt), 'f1': f1_score(y_test, y_pred_svm_pt), 'time': time_svm_pt},
    'Ridge Regression': {'acc': accuracy_score(y_test, (ridge_pred_pt > 0.5).astype(int)), 'f1': f1_score(y_test, (ridge_pred_pt > 0.5).astype(int)), 'time': time_ridge_pt}
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
lr_weights = lr.coef_[0]
lr_bias = lr.intercept_[0]

w_lr_enc = ts.ckks_vector(enc_training, lr_weights)

start_he_lr = time.perf_counter()
y_pred_lr_he = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(w_lr_enc) + lr_bias
    
    # Linear sigmoid approximation
    x_norm = y_lin_enc * 0.1
    y_sigmoid_enc = 0.5 + 0.1975 * x_norm
    
    y_sigmoid_dec = y_sigmoid_enc.decrypt()
    y_pred_lr_he.append(1 if y_sigmoid_dec[0] > 0.5 else 0)

y_pred_lr_he = np.array(y_pred_lr_he)
time_lr_he = time.perf_counter() - start_he_lr

# 6. Homomorphic Inference for SVM
svm_weights = svm.coef_[0]
svm_bias = svm.intercept_[0]

w_svm_enc = ts.ckks_vector(enc_training, svm_weights)

start_he_svm = time.perf_counter()
y_pred_svm_he = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(w_svm_enc) + svm_bias
    
    # Simple threshold at 0 (SVM decision boundary)
    y_sigmoid_dec = y_lin_enc.decrypt()
    y_pred_svm_he.append(1 if y_sigmoid_dec[0] > 0 else 0)

y_pred_svm_he = np.array(y_pred_svm_he)
time_svm_he = time.perf_counter() - start_he_svm

# 7. Homomorphic Inference for Ridge Regression
ridge_weights = ridge.coef_
ridge_bias = ridge.intercept_

w_ridge_enc = ts.ckks_vector(enc_training, ridge_weights)

start_he_ridge = time.perf_counter()
y_pred_ridge_he = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(w_ridge_enc) + ridge_bias
    
    # Threshold at 0.5 for classification
    y_sigmoid_dec = y_lin_enc.decrypt()
    y_pred_ridge_he.append(1 if y_sigmoid_dec[0] > 0.5 else 0)

y_pred_ridge_he = np.array(y_pred_ridge_he)
time_ridge_he = time.perf_counter() - start_he_ridge

metrics_he = {
    'Logistic Regression': {'acc': accuracy_score(y_test, y_pred_lr_he), 'f1': f1_score(y_test, y_pred_lr_he), 'time': time_lr_he},
    'SVM (Linear)': {'acc': accuracy_score(y_test, y_pred_svm_he), 'f1': f1_score(y_test, y_pred_svm_he), 'time': time_svm_he},
    'Ridge Regression': {'acc': accuracy_score(y_test, y_pred_ridge_he), 'f1': f1_score(y_test, y_pred_ridge_he), 'time': time_ridge_he}
}

# 8. Comparison Matrix
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
    },
    {
        'Model': 'Ridge Regression',
        'Plaintext Acc': metrics_pt['Ridge Regression']['acc'],
        'Plaintext F1': metrics_pt['Ridge Regression']['f1'],
        'Plaintext Time (s)': metrics_pt['Ridge Regression']['time'],
        'Encrypted Acc': metrics_he['Ridge Regression']['acc'],
        'Encrypted F1': metrics_he['Ridge Regression']['f1'],
        'Encrypted Time (s)': metrics_he['Ridge Regression']['time']
    }
])

print("=== Plaintext vs Encrypted Inference Comparison ===")
print(comparison.to_string(index=False))

# 9. Generate HE Comparison Graphs
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Plaintext vs Homomorphic Encrypted Inference Comparison', fontsize=16)

models = ['Logistic Regression', 'SVM (Linear)', 'Ridge Regression']
plaintext_acc = [metrics_pt[m]['acc'] for m in models]
encrypted_acc = [metrics_he[m]['acc'] for m in models]
plaintext_f1 = [metrics_pt[m]['f1'] for m in models]
encrypted_f1 = [metrics_he[m]['f1'] for m in models]
plaintext_time = [metrics_pt[m]['time'] for m in models]
encrypted_time = [metrics_he[m]['time'] for m in models]

# Accuracy Comparison
ax = axes[0, 0]
x = np.arange(len(models))
width = 0.35
ax.bar(x - width/2, plaintext_acc, width, label='Plaintext', alpha=0.8)
ax.bar(x + width/2, encrypted_acc, width, label='Encrypted', alpha=0.8)
ax.set_ylabel('Accuracy', fontsize=11)
ax.set_title('Accuracy Comparison', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()
ax.set_ylim([0, 1])

# F1 Score Comparison
ax = axes[0, 1]
ax.bar(x - width/2, plaintext_f1, width, label='Plaintext', alpha=0.8)
ax.bar(x + width/2, encrypted_f1, width, label='Encrypted', alpha=0.8)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('F1 Score Comparison', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()
ax.set_ylim([0, 1])

# Execution Time Comparison
ax = axes[1, 0]
ax.bar(x - width/2, plaintext_time, width, label='Plaintext', alpha=0.8)
ax.bar(x + width/2, encrypted_time, width, label='Encrypted', alpha=0.8)
ax.set_ylabel('Time (seconds)', fontsize=11)
ax.set_title('Execution Time Comparison', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend()

# Accuracy Loss (Plaintext - Encrypted)
ax = axes[1, 1]
accuracy_loss = [plaintext_acc[i] - encrypted_acc[i] for i in range(len(models))]
colors = ['green' if loss == 0 else 'orange' if loss < 0.05 else 'red' for loss in accuracy_loss]
ax.bar(models, accuracy_loss, color=colors, alpha=0.8)
ax.set_ylabel('Accuracy Loss', fontsize=11)
ax.set_title('Accuracy Loss (Plaintext - Encrypted)', fontsize=12)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax.set_ylim([-0.05, max(accuracy_loss) + 0.05])

plt.tight_layout()
plt.savefig('Project/HE_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: Project/HE_comparison.png")

# Additional detailed visualization for timing
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(models))
width = 0.35
bars1 = ax.bar(x - width/2, plaintext_time, width, label='Plaintext', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, encrypted_time, width, label='Encrypted', alpha=0.8, color='coral')

ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title('Inference Time: Plaintext vs Homomorphic Encrypted', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=11)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}s', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('Project/HE_timing_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Project/HE_timing_comparison.png")