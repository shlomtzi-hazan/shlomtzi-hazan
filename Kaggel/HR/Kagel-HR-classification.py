import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import shuffle
import numpy as np

#TODO: Add your files' location
folder = "Your files location HERE"

# Load training datasets
time_train = pd.read_csv(folder + "time_domain_features_train.csv")
freq_train = pd.read_csv(folder + "frequency_domain_features_train.csv")
nonlinear_train = pd.read_csv(folder + "heart_rate_non_linear_features_train.csv")

# Merge and shuffle training datasets
train_data = time_train.merge(freq_train, on="uuid").merge(nonlinear_train, on="uuid")
train_data = shuffle(train_data, random_state=42)  # Shuffle the dataset

# Load testing datasets
time_test = pd.read_csv(folder + "time_domain_features_test.csv")
freq_test = pd.read_csv(folder + "frequency_domain_features_test.csv")
nonlinear_test = pd.read_csv(folder + "heart_rate_non_linear_features_test.csv")

# Merge testing datasets
test_data = time_test.merge(freq_test, on="uuid").merge(nonlinear_test, on="uuid")

# Extract target variable (y) and features (X) for training and testing
y_train = train_data["condition"]  # Categorical column to predict
X_train = train_data.drop(columns=["uuid", "HR", "condition", "datasetId"])

y_test = test_data["condition"]
X_test = test_data.drop(columns=["uuid", "condition", "datasetId"])

# Encode target variable
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)  # Encode training target classes into numeric values
y_test_encoded = le.transform(y_test)  # Encode test target classes using the same encoder

# Handle missing values (if any)
X_train.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_test.mean(), inplace=True)

# Visualize feature correlations
plt.figure(figsize=(12, 8))
sns.heatmap(X_train.corr(), annot=False, cmap="coolwarm", cbar=True)
plt.title("Feature Correlation Heatmap")
plt.show

# Feature importance via mutual information
mi_scores = mutual_info_classif(X_train, y_train_encoded, random_state=42)
mi_scores = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
mi_scores.plot(kind="bar", color="skyblue")
plt.title("Feature Importance via Mutual Information")
plt.ylabel("Mutual Information Score")
plt.xlabel("Features")
plt.show

# Split training data into train and validation sets
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train_encoded, test_size=0.2, random_state=42
)

# Hardcoded best hyperparameters for Random Forest
# Pay attention: The hyperparameters can cause over fitting if misscalculated 
rf_best_params = {'max_depth': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}

# Train the best Random Forest model
rf_classifier = RandomForestClassifier(**rf_best_params, random_state=42)
rf_classifier.fit(X_train_split, y_train_split)

# Cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_classifier, X_train_split, y_train_split, cv=5, scoring='accuracy')
print(f"Random Forest Cross-Validation Accuracy: {rf_cv_scores.mean():.2f} (+/- {rf_cv_scores.std():.2f})")

# Feature importance from Random Forest
rf_importances = rf_classifier.feature_importances_
rf_indices = np.argsort(rf_importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(X_train.shape[1]), rf_importances[rf_indices], align="center", color="lightgreen")
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in rf_indices], rotation=90)
plt.title("Feature Importance from Random Forest")
plt.show()

# Validate Random Forest on the validation set
y_val_pred_rf = rf_classifier.predict(X_val)
rf_val_accuracy = accuracy_score(y_val, y_val_pred_rf)
print(f"Random Forest Validation Accuracy: {rf_val_accuracy:.2f}")
print("\nRandom Forest Validation Classification Report:\n", classification_report(y_val, y_val_pred_rf, target_names=le.classes_))

# Test Random Forest on the test set
y_test_pred_rf = rf_classifier.predict(X_test)
rf_test_accuracy = accuracy_score(y_test_encoded, y_test_pred_rf)
print(f"Random Forest Test Accuracy: {rf_test_accuracy:.2f}")
print("\nRandom Forest Test Classification Report:\n", classification_report(y_test_encoded, y_test_pred_rf, target_names=le.classes_))

# Add Gradient Boosting Classifier
gb_best_params = {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3}

# Train Gradient Boosting model
gb_classifier = GradientBoostingClassifier(**gb_best_params, random_state=42)
gb_classifier.fit(X_train_split, y_train_split)

# Cross-validation for Gradient Boosting
gb_cv_scores = cross_val_score(gb_classifier, X_train_split, y_train_split, cv=5, scoring='accuracy')
print(f"Gradient Boosting Cross-Validation Accuracy: {gb_cv_scores.mean():.2f} (+/- {gb_cv_scores.std():.2f})")

# Feature importance from Gradient Boosting
gb_importances = gb_classifier.feature_importances_
gb_indices = np.argsort(gb_importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(X_train.shape[1]), gb_importances[gb_indices], align="center", color="orange")
plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in gb_indices], rotation=90)
plt.title("Feature Importance from Gradient Boosting")
plt.show()

# Validate Gradient Boosting on the validation set
y_val_pred_gb = gb_classifier.predict(X_val)
gb_val_accuracy = accuracy_score(y_val, y_val_pred_gb)
print(f"Gradient Boosting Validation Accuracy: {gb_val_accuracy:.2f}")
print("\nGradient Boosting Validation Classification Report:\n", classification_report(y_val, y_val_pred_gb, target_names=le.classes_))

# Test Gradient Boosting on the test set
y_test_pred_gb = gb_classifier.predict(X_test)
gb_test_accuracy = accuracy_score(y_test_encoded, y_test_pred_gb)
print(f"Gradient Boosting Test Accuracy: {gb_test_accuracy:.2f}")
print("\nGradient Boosting Test Classification Report:\n", classification_report(y_test_encoded, y_test_pred_gb, target_names=le.classes_))

# Confusion matrix for Random Forest
def plot_confusion_matrix(conf_matrix, title, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.show()

plot_confusion_matrix(confusion_matrix(y_test_encoded, y_test_pred_rf), "Confusion Matrix (Random Forest)", le.classes_)
plot_confusion_matrix(confusion_matrix(y_test_encoded, y_test_pred_gb), "Confusion Matrix (Gradient Boosting)", le.classes_)

# Per-class ROC curves for Gradient Boosting
fpr_gb = {}
tpr_gb = {}
roc_auc_gb = {}

for i in range(len(le.classes_)):
    y_test_binary = (y_test_encoded == i).astype(int)
    y_score_gb = gb_classifier.predict_proba(X_test)[:, i]
    fpr_gb[i], tpr_gb[i], _ = roc_curve(y_test_binary, y_score_gb)
    roc_auc_gb[i] = roc_auc_score(y_test_binary, y_score_gb)

plt.figure(figsize=(10, 8))
for i in range(len(le.classes_)):
    plt.plot(fpr_gb[i], tpr_gb[i], label=f"Class {le.classes_[i]} (AUC = {roc_auc_gb[i]:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Per-Class ROC Curves (Gradient Boosting)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.show()

# Validate Gradient Boosting on the validation set
y_val_pred_gb = gb_classifier.predict(X_val)
gb_val_accuracy = accuracy_score(y_val, y_val_pred_gb)
print(f"Gradient Boosting Validation Accuracy: {gb_val_accuracy:.2f}")
print("\nGradient Boosting Validation Classification Report:\n", classification_report(y_val, y_val_pred_gb, target_names=le.classes_))

# Test Gradient Boosting on the test set
y_test_pred_gb = gb_classifier.predict(X_test)
gb_test_accuracy = accuracy_score(y_test_encoded, y_test_pred_gb)
print(f"Gradient Boosting Test Accuracy: {gb_test_accuracy:.2f}")
print("\nGradient Boosting Test Classification Report:\n", classification_report(y_test_encoded, y_test_pred_gb, target_names=le.classes_))
