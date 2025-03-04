import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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
from sklearn.model_selection import cross_val_score
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

print("\n" + "="*50)
print("DATA INTEGRATION AND ANALYSIS COMPLETE")
print("="*50)
