import tenseal as ts
import numpy as np

ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, [40, 21, 21, 21, 21, 21, 21, 40])
ctx.generate_galois_keys()
ctx.generate_relin_keys()

# Try different encryption approaches
test_data = np.array([1.0, 2.0, 3.0])

# Attempt 1: Direct context method
try:
    enc1 = ctx.encrypt(test_data)
    print("Method 1 (ctx.encrypt) works:", type(enc1))
except Exception as e:
    print("Method 1 failed:", e)

# Attempt 2: Using encryptor
try:
    encryptor = ctx.encryptor()
    enc2 = encryptor.encrypt(test_data)
    print("Method 2 (encryptor.encrypt) works:", type(enc2))
except Exception as e:
    print("Method 2 failed:", e)

# Attempt 3: Check if ts module has encryption functions
print("\nAvailable in ts module:", [attr for attr in dir(ts) if 'encrypt' in attr.lower() or 'vector' in attr.lower()])