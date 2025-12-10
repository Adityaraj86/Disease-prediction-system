import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load data
df = pd.read_csv("Thyroid_Diff.csv")

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Encode categorical variables
categorical_cols = ['Gender', 'Smoking', 'Hx Smoking', 'Hx Radiothreapy', 
                    'Thyroid Function', 'Physical Examination', 'Adenopathy', 
                    'Pathology', 'Focality', 'Risk', 'T', 'N', 'M', 'Stage', 'Response']

label_encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Encode target variable
target_encoder = LabelEncoder()
df['Recurred'] = target_encoder.fit_transform(df['Recurred'])

# Prepare features and target
X = df.drop("Recurred", axis=1)
y = df["Recurred"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42)
}

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, zero_division=0),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

# Train and evaluate models
results = {}
trained_models = {}

for name, model in models.items():
    if name in ["KNN", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        results[name] = evaluate_model(model, X_test_scaled, y_test)
    else:
        model.fit(X_train, y_train)
        trained_models[name] = model
        results[name] = evaluate_model(model, X_test, y_test)

# Save models and preprocessing objects
with open('thyroid_disease_models.pkl', 'wb') as f:
    pickle.dump({
        'models': trained_models,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'target_encoder': target_encoder,
        'results': results,
        'feature_names': X.columns.tolist()
    }, f)

# Print results
evaluation_df = pd.DataFrame({
    model: {
        "Accuracy": results[model]["Accuracy"],
        "Precision": results[model]["Precision"],
        "Recall": results[model]["Recall"],
        "F1 Score": results[model]["F1 Score"]
    }
    for model in results
}).T

print("Thyroid Disease Model Evaluation:")
print(evaluation_df)
print("\nConfusion Matrices:")
for model in results:
    print(f"\n{model} Confusion Matrix:")
    print(results[model]["Confusion Matrix"])

