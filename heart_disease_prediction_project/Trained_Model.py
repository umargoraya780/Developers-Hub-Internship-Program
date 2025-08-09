import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import joblib

df = pd.read_csv("heart_disease_uci_cleaned.csv")

# Converting target column 'num' to binary classification
df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)

df = df.drop(['id', 'dataset', 'num'], axis=1)

# One-hot encode categorical features
df = pd.get_dummies(df, drop_first=True)

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
# Probability for ROC curve
y_pred_proba = model.predict_proba(X_test)[:, 1]  

print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ROC-AUC score calculation
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# ROC Curve plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Random guess line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
fig = plt.gcf()  
fig.canvas.manager.set_window_title("ROC Curve")
plt.show()

# important features
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})
feature_importance['Absolute Importance'] = feature_importance['Coefficient'].abs()

# Sorting
feature_importance = feature_importance.sort_values(by='Absolute Importance', ascending=False)

print("\nTop Features Affecting Prediction:\n", feature_importance.head(10))

joblib.dump(model, "logistic_regression_model.pkl")
print("Model saved!")
