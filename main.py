import tenseal as ts
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib

# process data 
def load_and_preprocess_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
    df = pd.read_csv(url, names=column_names)

    df = df.replace('?', np.nan).dropna().astype(float)
    X = df.drop("target", axis=1)
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

# logistic regression model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    joblib.dump(model, "heart_disease_model.pkl")
    return model

# homomorphic encryption 
def setup_encryption():
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[40, 21, 21, 21, 21, 40])
    context.global_scale = 2**21
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context

def encrypt_sample(sample, context):
    encrypted_sample = ts.ckks_vector(context, sample)
    return encrypted_sample

def encrypt_model_coefficients(model, context):
    encrypted_coefficients = ts.ckks_vector(context, model.coef_[0])
    return encrypted_coefficients

def encrypted_prediction(encrypted_sample, encrypted_model_coefficients, context):
    encrypted_result = encrypted_sample.dot(encrypted_model_coefficients) 
    return encrypted_result

def decrypt_prediction(encrypted_result, context):
    decrypted_result = encrypted_result.decrypt()
    return decrypted_result[0]

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    model = train_model(X_train, y_train)
    context = setup_encryption()
    sample = X_test[0]
    encrypted_sample = encrypt_sample(sample, context)
    encrypted_coefficients = encrypt_model_coefficients(model, context)
    encrypted_result = encrypted_prediction(encrypted_sample, encrypted_coefficients, context)
    decrypted_prediction = decrypt_prediction(encrypted_result, context)
    
    print("Decrypted prediction (log-odds):", decrypted_prediction)
    print("Decrypted prediction (probability):", 1 / (1 + np.exp(-decrypted_prediction)))
