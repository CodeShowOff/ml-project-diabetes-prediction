from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)



# Load and clean data
DATA_PATH = 'diabetes.csv'
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df = pd.read_csv(DATA_PATH)
for col in cols_with_zero:
    df[col] = df[col].replace(0, np.nan)
    df[col].fillna(df[col].mean(), inplace=True)



# Remove outliers (rows where any feature is more than 3 std from mean)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Outcome')
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]



# EDA: Data statistics
print("\n--- Data Statistics ---")
print(df.describe())

# EDA: Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

# EDA: Feature histograms
df.hist(bins=15, figsize=(12,8))
plt.tight_layout()
plt.savefig('feature_histograms.png')
# Remove outliers (rows where any feature is more than 3 std from mean)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols.remove('Outcome')
for col in numeric_cols:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
plt.close()




# Feature engineering: add interaction and polynomial features
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['BMI_Squared'] = df['BMI'] ** 2
# Features and target (add engineered features for training)
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Glucose_BMI', 'BMI_Squared']]
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train models
dt_model = DecisionTreeClassifier(random_state=42, class_weight='balanced')
dt_model.fit(X_train_scaled, y_train)
lr_model = LogisticRegression(max_iter=200, class_weight='balanced')
lr_model.fit(X_train_scaled, y_train)
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)



# Evaluate models and select the best one
print('\n--- Decision Tree ---')
dt_pred = dt_model.predict(X_test_scaled)
dt_acc = accuracy_score(y_test, dt_pred)
print('Accuracy:', dt_acc)
print('Confusion Matrix:\n', confusion_matrix(y_test, dt_pred))
print('Classification Report:\n', classification_report(y_test, dt_pred))

print('\n--- Logistic Regression ---')
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print('Accuracy:', lr_acc)
print('Confusion Matrix:\n', confusion_matrix(y_test, lr_pred))
print('Classification Report:\n', classification_report(y_test, lr_pred))

print('\n--- Naive Bayes ---')
nb_pred = nb_model.predict(X_test_scaled)
nb_acc = accuracy_score(y_test, nb_pred)
print('Accuracy:', nb_acc)
print('Confusion Matrix:\n', confusion_matrix(y_test, nb_pred))
print('Classification Report:\n', classification_report(y_test, nb_pred))

# Select the best model (highest accuracy)
best_model_name = 'Decision Tree'
best_model = dt_model
best_acc = dt_acc
if lr_acc > best_acc:
    best_model_name = 'Logistic Regression'
    best_model = lr_model
    best_acc = lr_acc
if nb_acc > best_acc:
    best_model_name = 'Naive Bayes'
    best_model = nb_model
    best_acc = nb_acc
print(f'\nBest model: {best_model_name} (Accuracy: {best_acc:.2f})')

# ROC Curve
plt.figure(figsize=(7,5))
models = {
    'Decision Tree': dt_model,
    'Logistic Regression': lr_model,
    'Naive Bayes': nb_model
}
for name, model in models.items():
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_score = model.decision_function(X_test_scaled)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# Simple API for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Compute engineered features for prediction
    glucose_bmi = data['Glucose'] * data['BMI']
    bmi_squared = data['BMI'] ** 2
    input_data = np.array([
        data['Pregnancies'], data['Glucose'], data['BloodPressure'],
        data['SkinThickness'], data['Insulin'], data['BMI'],
        data['DiabetesPedigreeFunction'], data['Age'],
        glucose_bmi, bmi_squared
    ]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    # Use the best model for prediction
    result = best_model.predict(input_scaled)[0]
    if result == 0:
        return jsonify({'result': '✅ Non-Diabetic'})
    else:
        return jsonify({'result': '⚠️ Diabetic - High Risk'})

# Serve the HTML form
        glucose_bmi, bmi_squared
@app.route('/')
def home():
    return render_template_string(open('index.html').read())

if __name__ == '__main__':
    app.run()