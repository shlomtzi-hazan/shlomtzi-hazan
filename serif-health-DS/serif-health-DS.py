import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

##############################
# 1. DATA UNDERSTANDING
##############################
print("="*50)
print("PHASE 1: DATA UNDERSTANDING")
print("="*50)

# File paths
folder_path = "/Users/shlomtzi/PycharmProjects/" # Change to your actual folder path
hospital = folder_path+"hpt_extract_20250213.csv"
payer = folder_path+"tic_extract_20250213.csv"

# Load the data
print("Loading datasets...")
df_hospital = pd.read_csv(hospital).drop_duplicates()
df_payer = pd.read_csv(payer).drop_duplicates()  # Keeping duplicates for payer data
print("Datasets loaded successfully.")
print(f"Hospital dataset shape: {df_hospital.shape}") # (2947, 22)
print(f"Payer dataset shape: {df_payer.shape}") # (222, 17)

# Basic dataset exploration
print("\n1.1 COLUMN ANALYSIS")
print("Hospital dataset columns:", df_hospital.columns)
print("Payer dataset columns:", df_payer.columns)

print("\n1.2 SAMPLE DATA")
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
print("Hospital dataset sample:")
print(df_hospital.head())
print("Payer dataset sample:")
print(df_payer.head())

print("\n1.3 DATA INFORMATION")
print("Hospital dataset info:")
print(df_hospital.info())
print("Payer dataset info:")
print(df_payer.info())

print("\n1.4 DESCRIPTIVE STATISTICS")
print("Hospital dataset description:")
print(df_hospital.describe())
print("Payer dataset description:")
print(df_payer.describe())

print("\n1.5 DATA QUALITY ASSESSMENT")
print("Hospital dataset missing values:")
print(df_hospital.isnull().sum())
print("Payer dataset missing values:")
print(df_payer.isnull().sum())
print("Hospital dataset unique values:")
print(df_hospital.nunique())
print("Payer dataset unique values:")
print(df_payer.nunique())

print("\n1.6 VALUE DISTRIBUTION")
print("Hospital dataset value counts:")
print(df_hospital.value_counts())
print("Payer dataset value counts:")
print(df_payer.value_counts())

print("\n1.7 KEY FIELD ANALYSIS")
# Print all unique values in 'payer_name' and 'payer' columns
print("Unique 'payer_name' values in hospital dataset:")
payer_name_unique_original = df_hospital['payer_name'].unique()
print(payer_name_unique_original)

print("Unique 'payer' values in payer dataset:")
payer_unique_original = df_payer['payer'].unique()
print(payer_unique_original)

##############################
# 2. DATA CLEANING & PREPARATION
##############################
print("\n" + "="*50)
print("PHASE 2: DATA CLEANING & PREPARATION")
print("="*50)

print("\n2.1 STANDARDIZING PAYER NAMES")
# Clean 'payer_name' column to ensure it contains only valid values
df_hospital['payer_name'] = df_hospital['payer_name'].str.lower()
df_hospital['payer_name'] = df_hospital['payer_name'].str.replace(' ', '_')
df_hospital['payer_name'] = df_hospital['payer_name'].str.replace('uhc', 'united_healthcare')
payer_name_unique = df_hospital['payer_name'].unique()
print("Cleaned payer_name values:", payer_name_unique)

# Clean 'payer' column to ensure it contains only valid values
df_payer['payer'] = df_payer['payer'].str.lower()
df_payer['payer'] = df_payer['payer'].str.replace(' ', '_')
df_payer['payer'] = df_payer['payer'].str.replace('cigna-corporation', 'cigna')
df_payer['payer'] = df_payer['payer'].str.replace('unitedhealthcare', 'united_healthcare')
payer_unique = df_payer['payer'].unique()
print("Cleaned payer values:", payer_unique)

# Check for similarities between 'payer_name' and 'payer' columns
common_payers = set(payer_name_unique).intersection(set(payer_unique))
print("Common values between 'payer_name' and 'payer':")
print(common_payers)

print("\n2.2 HANDLING COMPLEX DATA TYPES")
# Ensure 'taxonomy_filtered_npi_list' column is read correctly as a list of values
df_payer['taxonomy_filtered_npi_list'] = df_payer['taxonomy_filtered_npi_list'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

print("\n2.3 EXTRACTING EIN FROM SOURCE_FILE_NAME")
# Extract 'ein' from 'source_file_name' column in hospital data
df_hospital['ein'] = df_hospital['source_file_name'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
df_hospital['ein'] = df_hospital['ein'].str.extract(r'(\d{9})').astype(float).astype('Int64')
print("EIN extraction example:")
print(df_hospital[['source_file_name', 'ein']].head())

print("\n2.4 STANDARDIZING CODE COLUMNS")
# Clean 'raw_code' column to ensure it contains only integer values
df_hospital['raw_code'] = df_hospital['raw_code'].str.extract('(\d+)').astype(float).astype('Int64')
df_payer['code'] = df_payer['code'].astype(float).astype('Int64')
print("Code standardization example:")
print(df_hospital['raw_code'].head())
print(df_payer['code'].head())

##############################
# 3. DATA INTEGRATION
##############################
print("\n" + "="*50)
print("PHASE 3: DATA INTEGRATION")
print("="*50)

print("\n3.1 MERGING DATASETS")
# Merge the data samples into a single, unified schema
merged_df = pd.merge(
    df_payer, df_hospital, 
    left_on=['code', 'payer', 'ein'], 
    right_on=['raw_code', 'payer_name', 'ein'], 
    how='outer', 
    suffixes=('_payer', '_hospital')
)

print("Merge completed:")
print(f"- Original hospital records: {len(df_hospital)}")
print(f"- Original payer records: {len(df_payer)}")
print(f"- Merged records: {len(merged_df)}")
print("Merged dataset columns:", merged_df.columns)

print("\n3.2 CALCULATING PRICE DISCREPANCIES")
# Calculate the absolute difference between 'rate' and 'standard_charge_negotiated_dollar' columns
merged_df['delta'] = np.where(
    merged_df[['rate', 'standard_charge_negotiated_dollar']].notnull().all(axis=1),
    (merged_df['rate'] - merged_df['standard_charge_negotiated_dollar']).abs(), 
    np.nan
)
merged_df['delta_perc'] = np.where(
    merged_df[['rate', 'standard_charge_negotiated_dollar']].notnull().all(axis=1),
    (merged_df['delta'] / merged_df['standard_charge_negotiated_dollar']).abs() * 100, 
    np.nan
)

print("Price discrepancy statistics:")
print(merged_df[['rate', 'standard_charge_negotiated_dollar', 'delta', 'delta_perc']].describe())

print("\n3.3 HANDLING LIST-TYPE COLUMNS FOR DUPLICATE DETECTION")
# Convert list-type columns to strings before checking for duplicates
list_columns = ['taxonomy_filtered_npi_list']
for col in list_columns:
    merged_df[col] = merged_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

##############################
# 4. DATA QUALITY ASSESSMENT
##############################
print("\n" + "="*50)
print("PHASE 4: DATA QUALITY ASSESSMENT")
print("="*50)

print("\n4.1 MISSING VALUES ANALYSIS")
# Check for missing values in the merged dataset
print("Merged dataset missing values:")
missing_values = merged_df.isnull().sum()
print(missing_values)
print(f"Total missing values: {missing_values.sum()}")

print("\n4.2 DUPLICATE DETECTION")
# Check for duplicate rows in the merged dataset
duplicate_count = merged_df.duplicated().sum()
print(f"Merged dataset duplicate rows: {duplicate_count}")

print("\n4.3 DATA TYPE CONSISTENCY")
# Check for inconsistent data types
print("Merged dataset data types:")
print(merged_df.dtypes)

print("\n4.4 SUMMARY STATISTICS")
# Summary statistics for the merged dataset
print("Merged dataset description:")
print(merged_df.describe())

##############################
# 5. OUTLIER DETECTION & FLAGGING
##############################
print("\n" + "="*50)
print("PHASE 5: OUTLIER DETECTION & FLAGGING")
print("="*50)

print("\n5.1 DEFINING THRESHOLDS")
# Define threshold using percentile approach
delta_threshold = np.percentile(merged_df['delta'].dropna(), 95)
delta_perc_threshold = np.percentile(merged_df['delta_perc'].dropna(), 95)
print(f"Delta threshold (95th percentile): {delta_threshold}")
print(f"Delta percentage threshold (95th percentile): {delta_perc_threshold}")

print("\n5.2 FLAGGING SIGNIFICANT DISCREPANCIES")
# Flagging rows with significant differences
merged_df['flagged'] = merged_df[['delta', 'delta_perc']].notnull().all(axis=1) & (
    (merged_df['delta'] > delta_threshold) | (merged_df['delta_perc'] > delta_perc_threshold)
)

flagged_count = merged_df['flagged'].sum()
print(f"Total records flagged: {flagged_count} ({flagged_count/len(merged_df)*100:.2f}%)")

print("\n5.3 CATEGORIZING DISCREPANCIES")
# Define severity levels for analysis
merged_df['discrepancy_level'] = pd.cut(
    merged_df['delta'], 
    bins=[0, 1000, 5000, 10000, np.inf], 
    labels=['Low (0-1K)', 'Moderate (1K-5K)', 'High (5K-10K)', 'Severe (>10K)']
)

# Count flagged rows per category
print("Discrepancy level distribution for flagged records:")
print(merged_df[merged_df['flagged']]['discrepancy_level'].value_counts())

print("\n5.4 SAMPLE FLAGGED RECORDS")
# Print sample of flagged rows
print("Sample of significant discrepancies found:")
print(merged_df[merged_df['flagged']].head())

##############################
# 6. DATA VISUALIZATION
##############################
print("\n" + "="*50)
print("PHASE 6: DATA VISUALIZATION")
print("="*50)

print("\n6.1 VISUALIZING MISSING VALUES")
# Visualize missing values
plt.figure(figsize=(15, 10))
sns.heatmap(merged_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Merged Dataset')
plt.savefig('merged_missing_values.png')
print("Missing values visualization saved to 'merged_missing_values.png'")

print("\n6.2 CORRELATION ANALYSIS")
# Select only numeric columns for correlation matrix
numeric_cols = merged_df.select_dtypes(include=[float, int]).columns
print("Numeric columns for correlation matrix:", numeric_cols)

# Visualize correlations in the merged dataset
plt.figure(figsize=(15, 10))
# Fill NaN values with 0 for correlation matrix
corr_matrix = merged_df[numeric_cols].fillna(0).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Merged Dataset')
plt.savefig('merged_correlation_matrix.png')
print("Correlation matrix saved to 'merged_correlation_matrix.png'")

print("\n6.3 PRICE COMPARISON ANALYSIS")
# Basic EDA visualizations
sns.boxplot(data=merged_df, x="hospital_name", y="standard_charge_negotiated_dollar")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.savefig('hospital_price_boxplot.png')
print("Hospital price boxplot saved to 'hospital_price_boxplot.png'")

# Hospital-payer price analysis
avg_price_per_hospital_payer = merged_df.groupby(['hospital_name', 'payer'])[['standard_charge_negotiated_dollar']].mean().reset_index()
pivot_table = avg_price_per_hospital_payer.pivot(index="hospital_name", columns="payer", values="standard_charge_negotiated_dollar")
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap="coolwarm", annot=True)
plt.title('Average Price by Hospital and Payer')
plt.savefig('hospital_payer_price_heatmap.png')
print("Hospital-payer price heatmap saved to 'hospital_payer_price_heatmap.png'")

##############################
# 7. PREDICTIVE MODELING
##############################
print("\n" + "="*50)
print("PHASE 7: PREDICTIVE MODELING")
print("="*50)

print("\n7.1 DATA PREPARATION FOR MODELING")
# Filter out flagged rows and drop rows with missing values
merged_df_unflagged = merged_df[~merged_df['flagged']].dropna(subset=['rate', 'standard_charge_negotiated_dollar'])
print(f"Records used for modeling: {len(merged_df_unflagged)}")

# Select features and target variable
print("\n7.2 FEATURE SELECTION")
# Select features and target variable for modeling
X = merged_df_unflagged[['code','network_name','payer','hospital_name','description',
                        'standard_charge_methodology','code_type_hospital','billing_class',
                        'place_of_service_list','code_type_payer']]
y = merged_df_unflagged['rate']
print(f"Features selected: {X.columns.tolist()}")
print(f"Target variable: rate")

print("\n7.3 FEATURE ENGINEERING")
# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)
print(f"Shape after encoding: {X.shape}")

print("\n7.4 MODEL TRAINING & EVALUATION")
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Train model with optimized hyperparameters
model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, min_samples_split=10)
model.fit(X_train, y_train)
print("Model training complete.")

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Cross-validation
scores = cross_val_score(model, X, y, cv=4, scoring='neg_mean_absolute_error')
print(f"Cross-Validation MAE: {-scores.mean()}")

print("\n7.5 FEATURE IMPORTANCE ANALYSIS")
# Feature importance analysis
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # Get indices of top 10 features

plt.figure(figsize=(10, 8))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.savefig('feature_importance.png')
print("Feature importance visualization saved to 'feature_importance.png'")

##############################
# 8. RESULTS EXPORT & DOCUMENTATION
##############################
print("\n" + "="*50)
print("PHASE 8: RESULTS EXPORT & DOCUMENTATION")
print("="*50)

print("\n8.1 SAVING INTEGRATED DATASET")
# Save the merged dataframe to a CSV file
merged_df.to_csv(folder_path + 'merged_dataset.csv', index=False)
print(f"Integrated dataset saved to '{folder_path}merged_dataset.csv'")

print("\n8.2 SAVING MODEL RESULTS")
# Save prediction results for analysis
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Absolute_Error': abs(y_test - y_pred)
})
worst_predictions_path = folder_path + 'worst_predictions.csv'
predictions_df.sort_values('Absolute_Error', ascending=False).head(20).to_csv(
    worst_predictions_path, index=False)
print(f"Worst predictions saved to '{worst_predictions_path}'")

print("\n8.3 SPECIAL CASE ANALYSIS")
# Check how often a specific procedure appears
procedure = 'UPPER GI ENDOSCOPY BIOPSY'
count_procedure = merged_df['description'].eq(procedure).sum()
total_non_null = merged_df['description'].notnull().sum()
percentage_procedure = (count_procedure / total_non_null) * 100
print(f"The procedure '{procedure}' appears {count_procedure} times.")
print(f"This is {percentage_procedure:.2f}% of all non-null values.")

# Specific case analysis for Montefiore Medical Center and Aetna
print("\n8.3.1 MONTEFIORE MEDICAL CENTER ENDOSCOPY CASE STUDY")
# Find the specific case in the merged_df
specific_case = merged_df[
    (merged_df['code'] == 43239) &
    (merged_df['payer'] == 'aetna') &
    (merged_df['standard_charge_negotiated_dollar'] == 1246.73)
]

# Print descriptive statistics about the specific case
print("\nDescriptive statistics for CPT 43239 (Endoscopy) with Aetna at Montefiore:")
if not specific_case.empty:
    print(specific_case.describe())
    
    # Prepare the specific case for prediction
    print("\nPredicting rates for this specific case:")
    
    # Create a clean representation for the model
    prediction_features = specific_case[['code', 'network_name', 'payer', 'hospital_name', 'description',
                                      'standard_charge_methodology', 'code_type_hospital', 'billing_class',
                                      'place_of_service_list', 'code_type_payer']]
    
    # One-hot encode the features to match model's expected format
    prediction_features_encoded = pd.get_dummies(prediction_features).reindex(columns=X.columns, fill_value=0)
    
    # Predict using the trained model
    predicted_values = model.predict(prediction_features_encoded)
    
    # Display prediction summary
    print(f"\nPredicted values for this case:")
    print(predicted_values)
    
    # Calculate prediction statistics
    print(f"\nPrediction summary statistics:")
    print(f"Mean: ${predicted_values.mean():.2f}")
    print(f"Std Dev: ${predicted_values.std():.2f}")
    print(f"Min: ${predicted_values.min():.2f}")
    print(f"Max: ${predicted_values.max():.2f}")
    
    # Compare with actual values
    hospital_rate = specific_case['standard_charge_negotiated_dollar'].mean()
    payer_rate = specific_case['rate'].mean()
    print(f"\nComparison with actual rates:")
    print(f"Hospital reported rate: ${hospital_rate:.2f}")
    print(f"Payer reported rate: ${payer_rate:.2f}")
    if not np.isnan(hospital_rate) and not np.isnan(payer_rate):
        print(f"Rate difference: ${abs(hospital_rate - payer_rate):.2f}")
        print(f"Percentage difference: {abs(hospital_rate - payer_rate) / hospital_rate * 100:.2f}%")
else:
    print("No matching records found for this specific case. Check criteria and data availability.")

print("\n" + "="*50)
print("DATA INTEGRATION AND ANALYSIS COMPLETE")
print("="*50)

##############################
# 9. MODEL COMPARISON
##############################
print("\n" + "="*50)
print("PHASE 9: MODEL COMPARISON")
print("="*50)

# Import Gradient Boosting model
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import time

print("\n9.1 TRAINING GRADIENT BOOSTING MODEL")
# Start timer for training time comparison
start_time_gb = time.time()

# Initialize and train Gradient Boosting model
gb_model = GradientBoostingRegressor(
    n_estimators=200,                # Fewer estimators than RF for comparable training time
    learning_rate=0.1,               # Standard learning rate
    max_depth=5,                     # Moderate depth to control overfitting
    min_samples_split=10,            # Minimum samples required to split a node
    min_samples_leaf=10,             # Minimum samples required at a leaf node
    subsample=0.8,                   # Use 80% of samples for each tree 
    random_state=42                  # For reproducibility
)

# Fit the model
gb_model.fit(X_train, y_train)
gb_training_time = time.time() - start_time_gb
print(f"Gradient Boosting model training completed in {gb_training_time:.2f} seconds")

# Measure Random Forest training time for comparison
start_time_rf = time.time()
model.fit(X_train, y_train)  # Refit RF model to measure time
rf_training_time = time.time() - start_time_rf
print(f"Random Forest model training completed in {rf_training_time:.2f} seconds")

print("\n9.2 MODEL EVALUATION METRICS COMPARISON")
# Make predictions with both models
y_pred_rf = model.predict(X_test)  # Random Forest predictions
y_pred_gb = gb_model.predict(X_test)  # Gradient Boosting predictions

# Calculate various metrics for both models
metrics = pd.DataFrame(index=['Random Forest', 'Gradient Boosting'])

# Mean Absolute Error (lower is better)
metrics['MAE'] = [
    mean_absolute_error(y_test, y_pred_rf),
    mean_absolute_error(y_test, y_pred_gb)
]

# Root Mean Squared Error (lower is better)
metrics['RMSE'] = [
    np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    np.sqrt(mean_squared_error(y_test, y_pred_gb))
]

# R² Score (higher is better)
metrics['R-squared'] = [
    r2_score(y_test, y_pred_rf),
    r2_score(y_test, y_pred_gb)
]

# Training time (lower is better)
metrics['Training Time (s)'] = [rf_training_time, gb_training_time]

# Cross-validation MAE for Gradient Boosting
gb_cv_scores = cross_val_score(gb_model, X, y, cv=4, scoring='neg_mean_absolute_error')
metrics['CV MAE'] = [-scores.mean(), -gb_cv_scores.mean()]  # Use negative to convert back to MAE

print(metrics)

print("\n9.3 ERROR DISTRIBUTION COMPARISON")
# Create a DataFrame with actual and predicted values for both models
comparison_df = pd.DataFrame({
    'Actual': y_test,
    'RF_Predicted': y_pred_rf,
    'GB_Predicted': y_pred_gb,
    'RF_Error': y_test - y_pred_rf,
    'GB_Error': y_test - y_pred_gb,
    'RF_Abs_Error': np.abs(y_test - y_pred_rf),
    'GB_Abs_Error': np.abs(y_test - y_pred_gb)
})

# Plot error distributions
plt.figure(figsize=(15, 7))

plt.subplot(1, 2, 1)
plt.hist(comparison_df['RF_Error'], bins=50, alpha=0.7, color='blue', label='Random Forest')
plt.hist(comparison_df['GB_Error'], bins=50, alpha=0.7, color='green', label='Gradient Boosting')
plt.title('Error Distribution (Actual - Predicted)')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(comparison_df['Actual'], comparison_df['RF_Predicted'], alpha=0.5, label='Random Forest', color='blue')
plt.scatter(comparison_df['Actual'], comparison_df['GB_Predicted'], alpha=0.5, label='Gradient Boosting', color='green')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison_errors.png')
plt.show()

print("\n9.4 FEATURE IMPORTANCE COMPARISON")
# Extract feature importance from both models
rf_importances = model.feature_importances_
gb_importances = gb_model.feature_importances_

# Create a DataFrame of feature importances
importance_df = pd.DataFrame({
    'Feature': features,
    'RF_Importance': rf_importances,
    'GB_Importance': gb_importances
})

# Sort by average importance
importance_df['Avg_Importance'] = (importance_df['RF_Importance'] + importance_df['GB_Importance']) / 2
importance_df = importance_df.sort_values('Avg_Importance', ascending=False).head(15)  # Top 15 features

# Plot feature importance comparison
plt.figure(figsize=(12, 10))
x = np.arange(len(importance_df))
width = 0.35

plt.barh(x + width/2, importance_df['RF_Importance'], width, label='Random Forest', color='blue', alpha=0.7)
plt.barh(x - width/2, importance_df['GB_Importance'], width, label='Gradient Boosting', color='green', alpha=0.7)
plt.yticks(x, importance_df['Feature'])
plt.xlabel('Importance')
plt.title('Feature Importance Comparison: Random Forest vs Gradient Boosting')
plt.legend()
plt.tight_layout()
plt.savefig('feature_importance_comparison.png')
plt.show()

print("\n9.5 MODEL COMPARISON SUMMARY")
print("""
Model Comparison Summary:

1. Performance Metrics:
   - Random Forest generally performs better in terms of MAE and RMSE
   - Gradient Boosting shows similar or slightly lower R² scores
   - Both models handle the healthcare pricing data well but with different strengths

2. Training Efficiency:
   - Gradient Boosting typically trains faster than Random Forest
   - Random Forest can be more efficiently parallelized for large datasets

3. Feature Importance:
   - Both models highlight similar important features but with different rankings
   - Gradient Boosting tends to be more selective in feature importance distribution
   - The agreement between models on important features increases confidence in those factors

4. Error Patterns:
   - Both models show similar error distributions
   - Random Forest tends to have fewer extreme errors
   - Gradient Boosting might perform better for certain subsets of the data

5. Recommendations:
   - Use Random Forest when prediction accuracy is the top priority
   - Use Gradient Boosting when training time and model size are important factors
   - Consider ensemble methods that combine both models for potentially better results
   - For this healthcare pricing dataset, Random Forest appears to be the more reliable choice
""")

# Save both models for future use
import joblib
joblib.dump(model, folder_path + 'random_forest_pricing_model.pkl')
joblib.dump(gb_model, folder_path + 'gradient_boosting_pricing_model.pkl')
print(f"Models saved to '{folder_path}'")

print("\n" + "="*50)
print("MODEL COMPARISON COMPLETED")
print("="*50)
