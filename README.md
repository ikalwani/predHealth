# Privacy-Preserving Healthcare Predictions

This project demonstrates a secure prediction system for healthcare data using **homomorphic encryption**. Leveraging the **TenSEAL** library, the system performs logistic regression predictions on encrypted patient data, ensuring that sensitive information remains confidential.

## Real-Life Scenario

In a hospital setting, protecting patient privacy is critical. This project addresses this by using homomorphic encryption to process healthcare predictions securely, ensuring compliance with privacy standards while analyzing sensitive patient data.

## Features

- **Homomorphic Encryption**: Uses TenSEAL to encrypt patient data, allowing computations on encrypted data without exposing the data itself.
- **Logistic Regression**: Applies a logistic regression model to predict health outcomes based on the UCI heart disease dataset.
- **Privacy Compliance**: Maintains healthcare privacy standards by keeping patient data confidential throughout the prediction process.

## How It Works

1. **Data Preparation**: Loads and preprocesses the UCI heart disease dataset, including standardization.
2. **Model Training**: Trains a logistic regression model on the preprocessed data.
3. **Encryption Setup**: Configures TenSEAL for homomorphic encryption, generating encryption keys and setting up the encryption context.
4. **Secure Prediction**: Encrypts input data, performs predictions on encrypted data, and decrypts the results to obtain predictions.
