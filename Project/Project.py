import pandas as pd
import numpy as np
import tenseal as ts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import time
import matplotlib.pyplot as plt

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

# 4. Setup CKKS Contexts (two variants)
# CKKS (8192) - Higher precision
poly_mod_degree_8192 = 8192
coeff_mod_bit_sizes_8192 = [60, 40, 40, 60] # Larger primes = more room for precision

enc_training_ckks_8192 = ts.context(
    ts.SCHEME_TYPE.CKKS, 
    poly_mod_degree_8192, 
    coeff_mod_bit_sizes_8192, 
)
enc_training_ckks_8192.global_scale = 2 ** 40 # Radically higher fractional precision
enc_training_ckks_8192.generate_galois_keys()
enc_training_ckks_8192.generate_relin_keys()

# CKKS (4096) - Lower precision variant for comparison
poly_mod_degree_4096 = 4096
coeff_mod_bit_sizes_4096 = [40, 21, 21, 21, 21, 40]

enc_training_ckks_4096 = ts.context(
    ts.SCHEME_TYPE.CKKS, 
    poly_mod_degree_4096, 
    coeff_mod_bit_sizes_4096, 
)
enc_training_ckks_4096.global_scale = 2 ** 20
enc_training_ckks_4096.generate_galois_keys()
enc_training_ckks_4096.generate_relin_keys()

#---

# 5. Homomorphic Inference with CKKS (8192)
lr_weights = np.array(lr.coef_[0]).tolist()
lr_bias = lr.intercept_[0]
svm_weights = np.array(svm.coef_[0]).tolist()
svm_bias = svm.intercept_[0]

# Calibrate thresholds on encrypted training data
train_sample_size = min(50, X_train_scaled.shape[0])
y_lin_enc_train_lr = []
y_lin_enc_train_svm = []

for i in range(train_sample_size):
    sample_enc = ts.ckks_vector(enc_training_ckks_8192, X_train_scaled[i])
    y_lin_enc_lr = sample_enc.dot(lr_weights) + lr_bias
    y_lin_enc_train_lr.append(y_lin_enc_lr.decrypt()[0])
    
    y_lin_enc_svm = sample_enc.dot(svm_weights) + svm_bias
    y_lin_enc_train_svm.append(y_lin_enc_svm.decrypt()[0])

threshold_lr_ckks = np.mean(y_lin_enc_train_lr)
threshold_svm_ckks = np.mean(y_lin_enc_train_svm)

# CKKS (8192) - Logistic Regression
start_he_lr_ckks_8192 = time.perf_counter()
y_pred_lr_ckks_8192 = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training_ckks_8192, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(lr_weights) + lr_bias
    y_lin_dec = y_lin_enc.decrypt()
    y_pred_lr_ckks_8192.append(1 if y_lin_dec[0] > threshold_lr_ckks else 0)

y_pred_lr_ckks_8192 = np.array(y_pred_lr_ckks_8192)
time_lr_ckks_8192 = time.perf_counter() - start_he_lr_ckks_8192

# CKKS (8192) - SVM
start_he_svm_ckks_8192 = time.perf_counter()
y_pred_svm_ckks_8192 = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training_ckks_8192, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(svm_weights) + svm_bias
    y_lin_dec = y_lin_enc.decrypt()
    y_pred_svm_ckks_8192.append(1 if y_lin_dec[0] > threshold_svm_ckks else 0)

y_pred_svm_ckks_8192 = np.array(y_pred_svm_ckks_8192)
time_svm_ckks_8192 = time.perf_counter() - start_he_svm_ckks_8192

#---

# 6. Homomorphic Inference with CKKS (4096)
# Calibrate thresholds for 4096 variant
y_lin_enc_train_lr_4096 = []
y_lin_enc_train_svm_4096 = []

for i in range(train_sample_size):
    sample_enc_4096 = ts.ckks_vector(enc_training_ckks_4096, X_train_scaled[i])
    y_lin_enc_lr_4096 = (sample_enc_4096.dot(lr_weights) + lr_bias).decrypt()[0]
    y_lin_enc_train_lr_4096.append(y_lin_enc_lr_4096)
    
    y_lin_enc_svm_4096 = (sample_enc_4096.dot(svm_weights) + svm_bias).decrypt()[0]
    y_lin_enc_train_svm_4096.append(y_lin_enc_svm_4096)

threshold_lr_ckks_4096 = np.mean(y_lin_enc_train_lr_4096)
threshold_svm_ckks_4096 = np.mean(y_lin_enc_train_svm_4096)

# CKKS (4096) - Logistic Regression
start_he_lr_ckks_4096 = time.perf_counter()
y_pred_lr_ckks_4096 = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training_ckks_4096, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(lr_weights) + lr_bias
    y_lin_dec = y_lin_enc.decrypt()
    y_pred_lr_ckks_4096.append(1 if y_lin_dec[0] > threshold_lr_ckks_4096 else 0)

y_pred_lr_ckks_4096 = np.array(y_pred_lr_ckks_4096)
time_lr_ckks_4096 = time.perf_counter() - start_he_lr_ckks_4096

# CKKS (4096) - SVM
start_he_svm_ckks_4096 = time.perf_counter()
y_pred_svm_ckks_4096 = []

for i in range(X_test_scaled.shape[0]):
    sample_enc = ts.ckks_vector(enc_training_ckks_4096, X_test_scaled[i])
    y_lin_enc = sample_enc.dot(svm_weights) + svm_bias
    y_lin_dec = y_lin_enc.decrypt()
    y_pred_svm_ckks_4096.append(1 if y_lin_dec[0] > threshold_svm_ckks_4096 else 0)

y_pred_svm_ckks_4096 = np.array(y_pred_svm_ckks_4096)
time_svm_ckks_4096 = time.perf_counter() - start_he_svm_ckks_4096


metrics_he = {
    'CKKS (8192)': {
        'Logistic Regression': {'acc': accuracy_score(y_test, y_pred_lr_ckks_8192), 'f1': f1_score(y_test, y_pred_lr_ckks_8192), 'time': time_lr_ckks_8192},
        'SVM (Linear)': {'acc': accuracy_score(y_test, y_pred_svm_ckks_8192), 'f1': f1_score(y_test, y_pred_svm_ckks_8192), 'time': time_svm_ckks_8192}
    },
    'CKKS (4096)': {
        'Logistic Regression': {'acc': accuracy_score(y_test, y_pred_lr_ckks_4096), 'f1': f1_score(y_test, y_pred_lr_ckks_4096), 'time': time_lr_ckks_4096},
        'SVM (Linear)': {'acc': accuracy_score(y_test, y_pred_svm_ckks_4096), 'f1': f1_score(y_test, y_pred_svm_ckks_4096), 'time': time_svm_ckks_4096}
    }
}

# 7. Comprehensive Comparison Matrix
comparison = pd.DataFrame([
    # Plaintext Baseline
    {
        'Model': 'Logistic Regression',
        'Library': 'Plaintext',
        'F1 Score': metrics_pt['Logistic Regression']['f1'],
        'Accuracy': metrics_pt['Logistic Regression']['acc'],
        'Time (s)': metrics_pt['Logistic Regression']['time']
    },
    {
        'Model': 'SVM (Linear)',
        'Library': 'Plaintext',
        'F1 Score': metrics_pt['SVM (Linear)']['f1'],
        'Accuracy': metrics_pt['SVM (Linear)']['acc'],
        'Time (s)': metrics_pt['SVM (Linear)']['time']
    },
    # CKKS (8192) Encrypted
    {
        'Model': 'Logistic Regression',
        'Library': 'CKKS (8192)',
        'F1 Score': metrics_he['CKKS (8192)']['Logistic Regression']['f1'],
        'Accuracy': metrics_he['CKKS (8192)']['Logistic Regression']['acc'],
        'Time (s)': metrics_he['CKKS (8192)']['Logistic Regression']['time']
    },
    {
        'Model': 'SVM (Linear)',
        'Library': 'CKKS (8192)',
        'F1 Score': metrics_he['CKKS (8192)']['SVM (Linear)']['f1'],
        'Accuracy': metrics_he['CKKS (8192)']['SVM (Linear)']['acc'],
        'Time (s)': metrics_he['CKKS (8192)']['SVM (Linear)']['time']
    },
    # CKKS (4096) Encrypted
    {
        'Model': 'Logistic Regression',
        'Library': 'CKKS (4096)',
        'F1 Score': metrics_he['CKKS (4096)']['Logistic Regression']['f1'],
        'Accuracy': metrics_he['CKKS (4096)']['Logistic Regression']['acc'],
        'Time (s)': metrics_he['CKKS (4096)']['Logistic Regression']['time']
    },
    {
        'Model': 'SVM (Linear)',
        'Library': 'CKKS (4096)',
        'F1 Score': metrics_he['CKKS (4096)']['SVM (Linear)']['f1'],
        'Accuracy': metrics_he['CKKS (4096)']['SVM (Linear)']['acc'],
        'Time (s)': metrics_he['CKKS (4096)']['SVM (Linear)']['time']
    }
])

print("=== Model Performance Comparison: Plaintext vs CKKS (8192) vs CKKS (4096) ===")
print(comparison.to_string(index=False))
print("\n")

print("\n--- Generating Learning Curves ---")

# Proportions of the training set to use
train_sizes = np.linspace(0.2, 1.0, 5)

# Dictionaries to store F1 scores
f1_scores = {
    'LR_Plaintext': [],
    'LR_CKKS_8192': [],
    'LR_CKKS_4096': [],
    'SVM_Plaintext': [],
    'SVM_CKKS_8192': [],
    'SVM_CKKS_4096': []
}

actual_train_sizes = []

for frac in train_sizes:
    # 1. Subset the training data
    subset_size = int(frac * len(X_train_scaled))
    actual_train_sizes.append(subset_size)
    X_train_sub = X_train_scaled[:subset_size]
    y_train_sub = y_train[:subset_size]
    
    # 2. Train Plaintext Models
    lr_sub = LogisticRegression(max_iter=1000, random_state=13)
    svm_sub = SVC(kernel='linear', random_state=13)
    
    lr_sub.fit(X_train_sub, y_train_sub)
    svm_sub.fit(X_train_sub, y_train_sub)
    
    # 3. Plaintext Test Inference
    y_pred_lr_pt = lr_sub.predict(X_test_scaled)
    y_pred_svm_pt = svm_sub.predict(X_test_scaled)
    
    f1_scores['LR_Plaintext'].append(f1_score(y_test, y_pred_lr_pt))
    f1_scores['SVM_Plaintext'].append(f1_score(y_test, y_pred_svm_pt))
    
    # Extract weights for HE inference
    lr_weights_sub = np.array(lr_sub.coef_[0]).tolist()
    lr_bias_sub = lr_sub.intercept_[0]
    
    svm_weights_sub = np.array(svm_sub.coef_[0]).tolist()
    svm_bias_sub = svm_sub.intercept_[0]

    train_sample_size_sub = min(50, len(X_train_sub))
    
    # 8192 Calibration
    y_lin_train_lr_sub = []
    y_lin_train_svm_sub = []
    
    for i in range(train_sample_size_sub):
        sample_enc = ts.ckks_vector(enc_training_ckks_8192, X_train_sub[i])
        y_lin_train_lr_sub.append((sample_enc.dot(lr_weights_sub) + lr_bias_sub).decrypt()[0])
        y_lin_train_svm_sub.append((sample_enc.dot(svm_weights_sub) + svm_bias_sub).decrypt()[0])

    thresh_lr_8192 = np.mean(y_lin_train_lr_sub)
    thresh_svm_8192 = np.mean(y_lin_train_svm_sub)
    
    # 4096 Calibration
    y_lin_train_lr_4096_sub = []
    y_lin_train_svm_4096_sub = []
    
    for i in range(train_sample_size_sub):
        sample_enc_4096 = ts.ckks_vector(enc_training_ckks_4096, X_train_sub[i])
        y_lin_train_lr_4096_sub.append((sample_enc_4096.dot(lr_weights_sub) + lr_bias_sub).decrypt()[0])
        y_lin_train_svm_4096_sub.append((sample_enc_4096.dot(svm_weights_sub) + svm_bias_sub).decrypt()[0])

    thresh_lr_4096 = np.mean(y_lin_train_lr_4096_sub)
    thresh_svm_4096 = np.mean(y_lin_train_svm_4096_sub)
    
    # 4. CKKS (8192) HE Inference
    y_pred_lr_ckks_8192_sub = []
    y_pred_svm_ckks_8192_sub = []
    
    for i in range(X_test_scaled.shape[0]):
        sample_enc = ts.ckks_vector(enc_training_ckks_8192, X_test_scaled[i])
        
        # CKKS (8192) LR
        y_lin_enc_lr = sample_enc.dot(lr_weights_sub) + lr_bias_sub
        y_lin_dec_lr = y_lin_enc_lr.decrypt()
        y_pred_lr_ckks_8192_sub.append(1 if y_lin_dec_lr[0] > thresh_lr_8192 else 0)        
        # CKKS (8192) SVM
        y_lin_enc_svm = sample_enc.dot(svm_weights_sub) + svm_bias_sub
        y_lin_dec_svm = y_lin_enc_svm.decrypt()
        y_pred_svm_ckks_8192_sub.append(1 if y_lin_dec_svm[0] > thresh_svm_8192 else 0)
        
    f1_scores['LR_CKKS_8192'].append(f1_score(y_test, y_pred_lr_ckks_8192_sub))
    f1_scores['SVM_CKKS_8192'].append(f1_score(y_test, y_pred_svm_ckks_8192_sub))
    
    # 5. CKKS (4096) HE Inference
    y_pred_lr_ckks_4096_sub = []
    y_pred_svm_ckks_4096_sub = []
    
    for i in range(X_test_scaled.shape[0]):
        sample_enc = ts.ckks_vector(enc_training_ckks_4096, X_test_scaled[i])
        
        # CKKS (4096) LR
        y_lin_enc_lr = sample_enc.dot(lr_weights_sub) + lr_bias_sub
        y_lin_dec_lr = y_lin_enc_lr.decrypt()
        y_pred_lr_ckks_4096_sub.append(1 if y_lin_dec_lr[0] > thresh_lr_4096 else 0)
        
        # CKKS (4096) SVM
        y_lin_enc_svm = sample_enc.dot(svm_weights_sub) + svm_bias_sub
        y_lin_dec_svm = y_lin_enc_svm.decrypt()
        y_pred_svm_ckks_4096_sub.append(1 if y_lin_dec_svm[0] > thresh_svm_4096 else 0)
        
    f1_scores['LR_CKKS_4096'].append(f1_score(y_test, y_pred_lr_ckks_4096_sub))
    f1_scores['SVM_CKKS_4096'].append(f1_score(y_test, y_pred_svm_ckks_4096_sub))
    
    print(f"Evaluated on {subset_size} training samples (Plaintext + CKKS (8192) + CKKS (4096)).")

# 8. Plotting Learning Curves
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Logistic Regression
axes[0, 0].plot(actual_train_sizes, f1_scores['LR_Plaintext'], 'b-o', label='Plaintext', linewidth=2, markersize=8)
axes[0, 0].plot(actual_train_sizes, f1_scores['LR_CKKS_8192'], 'r--s', label='CKKS (8192)', linewidth=2, markersize=8)
axes[0, 0].plot(actual_train_sizes, f1_scores['LR_CKKS_4096'], 'g:^', label='CKKS (4096)', linewidth=2, markersize=8)
axes[0, 0].set_title('Learning Curve: Logistic Regression', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Number of Training Samples')
axes[0, 0].set_ylabel('Test F1 Score')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# SVM
axes[0, 1].plot(actual_train_sizes, f1_scores['SVM_Plaintext'], 'b-o', label='Plaintext', linewidth=2, markersize=8)
axes[0, 1].plot(actual_train_sizes, f1_scores['SVM_CKKS_8192'], 'r--s', label='CKKS (8192)', linewidth=2, markersize=8)
axes[0, 1].plot(actual_train_sizes, f1_scores['SVM_CKKS_4096'], 'g:^', label='CKKS (4096)', linewidth=2, markersize=8)
axes[0, 1].set_title('Learning Curve: SVM (Linear)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Number of Training Samples')
axes[0, 1].set_ylabel('Test F1 Score')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# F1 Score Comparison at Full Training Data
models = ['LR', 'SVM']
plaintext_f1 = [f1_scores['LR_Plaintext'][-1], f1_scores['SVM_Plaintext'][-1]]
ckks_8192_f1 = [f1_scores['LR_CKKS_8192'][-1], f1_scores['SVM_CKKS_8192'][-1]]
ckks_4096_f1 = [f1_scores['LR_CKKS_4096'][-1], f1_scores['SVM_CKKS_4096'][-1]]

x = np.arange(len(models))
width = 0.25

axes[1, 0].bar(x - width, plaintext_f1, width, label='Plaintext', color='blue', alpha=0.8)
axes[1, 0].bar(x, ckks_8192_f1, width, label='CKKS (8192)', color='red', alpha=0.8)
axes[1, 0].bar(x + width, ckks_4096_f1, width, label='CKKS (4096)', color='green', alpha=0.8)
axes[1, 0].set_ylabel('F1 Score')
axes[1, 0].set_title('F1 Score Comparison (Full Training Data)', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(models)
axes[1, 0].legend()
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Timing Comparison
timing_data = {
    'LR_Plaintext': time_lr_pt,
    'LR_CKKS_8192': time_lr_ckks_8192,
    'LR_CKKS_4096': time_lr_ckks_4096,
    'SVM_Plaintext': time_svm_pt,
    'SVM_CKKS_8192': time_svm_ckks_8192,
    'SVM_CKKS_4096': time_svm_ckks_4096
}

libraries = ['Plaintext', 'CKKS (8192)', 'CKKS (4096)']
lr_times = [timing_data['LR_Plaintext'], timing_data['LR_CKKS_8192'], timing_data['LR_CKKS_4096']]
svm_times = [timing_data['SVM_Plaintext'], timing_data['SVM_CKKS_8192'], timing_data['SVM_CKKS_4096']]

x = np.arange(len(libraries))
axes[1, 1].bar(x - width/2, lr_times, width, label='LR', color='skyblue', alpha=0.8)
axes[1, 1].bar(x + width/2, svm_times, width, label='SVM', color='orange', alpha=0.8)
axes[1, 1].set_ylabel('Time (seconds)')
axes[1, 1].set_title('Inference Time Comparison', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(libraries)
axes[1, 1].legend()
axes[1, 1].set_yscale('log')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()