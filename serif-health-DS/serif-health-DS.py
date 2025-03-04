import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np

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
print("Hospital dataset columns:", df_hospital.columns)
print("Payer dataset columns:", df_payer.columns)
print("Hospital dataset sample:")
pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
print(df_hospital.head())
print("Payer dataset sample:")
print(df_payer.head())
print("Hospital dataset info:")
print(df_hospital.info())
print("Payer dataset info:")
print(df_payer.info())
print("Hospital dataset description:")
print(df_hospital.describe())
print("Payer dataset description:")
print(df_payer.describe())
print("Hospital dataset missing values:")
print(df_hospital.isnull().sum())
print("Payer dataset missing values:")
print(df_payer.isnull().sum())
print("Hospital dataset unique values:")
print(df_hospital.nunique())
print("Payer dataset unique values:")
print(df_payer.nunique())
print("Hospital dataset value counts:")
print(df_hospital.value_counts())
print("Payer dataset value counts:")
print(df_payer.value_counts())

# Print all unique values in 'payer_name' and 'payer' columns
# Print unique values in 'payer_name' and 'payer' columns
print("Unique 'payer_name' values in hospital dataset:")
# Clean 'payer_name' column to ensure it contains only valid values
df_hospital['payer_name'] = df_hospital['payer_name'].str.lower()
df_hospital['payer_name'] = df_hospital['payer_name'].str.replace(' ', '_')
df_hospital['payer_name'] = df_hospital['payer_name'].str.replace('uhc', 'united_healthcare')
payer_name_unique = df_hospital['payer_name'].unique()
print(payer_name_unique)

# Clean 'payer' column to ensure it contains only valid values
print("Unique 'payer' values in payer dataset:")
df_payer['payer'] = df_payer['payer'].str.lower()
df_payer['payer'] = df_payer['payer'].str.replace(' ', '_')
df_payer['payer'] = df_payer['payer'].str.replace('cigna-corporation', 'cigna')
df_payer['payer'] = df_payer['payer'].str.replace('unitedhealthcare', 'united_healthcare')

payer_unique = df_payer['payer'].unique()
print(payer_unique)

# Check for similarities between 'payer_name' and 'payer' columns
common_payers = set(payer_name_unique).intersection(set(payer_unique))
print("Common values between 'payer_name' and 'payer':")
print(common_payers)

# Ensure 'taxonomy_filtered_npi_list' column is read correctly as a list of values
df_payer['taxonomy_filtered_npi_list'] = df_payer['taxonomy_filtered_npi_list'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

# Extract 'ein' from 'source_file_name' column in hospital data
df_hospital['ein'] = df_hospital['source_file_name'].str.replace(r'[^a-zA-Z0-9]', '', regex=True)
df_hospital['ein'] = df_hospital['ein'].str.extract(r'(\d{9})').astype(float).astype('Int64')

# Clean 'raw_code' column to ensure it contains only integer values
df_hospital['raw_code'] = df_hospital['raw_code'].str.extract('(\d+)').astype(float).astype('Int64')
df_payer['code'] = df_payer['code'].astype(float).astype('Int64')

# Merge the data samples into a single, unified schema using 'code', 'payer' (payer), 'raw_code', 'payer_name' (hospital), and 'ein' columns
merged_df = pd.merge(df_payer, df_hospital, left_on=['code', 'payer', 'ein'], right_on=['raw_code', 'payer_name', 'ein'], how='outer', suffixes=('_payer', '_hospital'))

print("Merged dataset saved to 'merged_dataset.csv'")
print("Merged dataset shape:", merged_df.shape)
print("Merged dataset columns:", merged_df.columns)
print("Merged dataset sample:")
# Calculate the absolute difference between 'rangre' and 'standard_charge_negotiated_dollar' columns
# and store it in a new column 'delta'
merged_df['delta'] = np.where(merged_df[['rate', 'standard_charge_negotiated_dollar']].notnull().all(axis=1),
                              (merged_df['rate'] - merged_df['standard_charge_negotiated_dollar']).abs(), np.nan)
merged_df['delta_perc'] = np.where(merged_df[['rate', 'standard_charge_negotiated_dollar']].notnull().all(axis=1),
                                   (merged_df['delta'] / merged_df['standard_charge_negotiated_dollar']).abs() * 100, np.nan)
# Explanation:
# The 'delta' column represents the absolute difference between the 'rate' and 'standard_charge_negotiated_dollar' columns.
# This can help identify discrepancies or variations between the expected range and the negotiated dollar amount.
print(merged_df.head())

# Identify inconsistencies or discrepancies in the combined data
# Check for missing values in the merged dataset
print("Merged dataset missing values:")
print(merged_df.isnull().sum())

# Convert list-type columns to strings before checking for duplicates
list_columns = ['taxonomy_filtered_npi_list']
for col in list_columns:
    merged_df[col] = merged_df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

# Check for duplicate rows in the merged dataset
print("Merged dataset duplicate rows:")
print(merged_df.duplicated().sum())

# Check for inconsistent data types
print("Merged dataset data types:")
print(merged_df.dtypes)

# Summary statistics for the merged dataset
print("Merged dataset description:")
print(merged_df.describe())

# Check how often the number 1246.73 is used in the 'standard_charge_negotiated_dollar' column
count_1246_73 = merged_df['description'].eq('UPPER GI ENDOSCOPY BIOPSY').sum()
total_non_null = merged_df['description'].notnull().sum()
percentage_1246_73 = (count_1246_73 / total_non_null) * 100
print(f"The number 1246.73 appears {count_1246_73} times in the 'standard_charge_negotiated_dollar' column.")
print(f"This is {percentage_1246_73:.2f}% of all non-null values in this column.")
print(f"The number 1246.73 appears {count_1246_73} times in the 'standard_charge_negotiated_dollar' column.")

# Select only numeric columns for correlation matrix
numeric_cols = merged_df.select_dtypes(include=[float, int]).columns
print("Numeric columns for correlation matrix:", numeric_cols)

# Visualize missing values
plt.figure(figsize=(15, 10))
sns.heatmap(merged_df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values in Merged Dataset')
plt.savefig('merged_missing_values.png')
# plt.show()

# Visualize correlations in the merged dataset
plt.figure(figsize=(15, 10))
# Fill NaN values with 0 for correlation matrix
corr_matrix = merged_df[numeric_cols].fillna(0).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Merged Dataset')
plt.savefig('merged_correlation_matrix.png')
#plt.show()

# Define threshold using standard deviation method
delta_threshold = merged_df['delta'].mean() + 2 * merged_df['delta'].std()
delta_perc_threshold = merged_df['delta_perc'].mean() + 2 * merged_df['delta_perc'].std()

# Alternatively, use percentile approach
delta_threshold = np.percentile(merged_df['delta'].dropna(), 95)
delta_perc_threshold = np.percentile(merged_df['delta_perc'].dropna(), 95)

# Flagging rows with significant differences
merged_df['flagged'] = merged_df[['delta', 'delta_perc']].notnull().all(axis=1) & (
    (merged_df['delta'] > delta_threshold) | (merged_df['delta_perc'] > delta_perc_threshold)
)

print(merged_df[merged_df['flagged']].isnull().sum())

# Print flagged rows
print("Significant discrepancies found:")
print(merged_df[merged_df['flagged']])

# Define severity levels for analysis
merged_df['discrepancy_level'] = pd.cut(
    merged_df['delta'], 
    bins=[0, 1000, 5000, 10000, np.inf], 
    labels=['Low (0-1K)', 'Moderate (1K-5K)', 'High (5K-10K)', 'Severe (>10K)']
)

# Count flagged rows per category
print(merged_df[merged_df['flagged']]['discrepancy_level'].value_counts())

# Non-flagged rows: Use for analysis/modeling
# Flagged rows: Investigate, categorize, and either correct or exclude


# Save the merged dataframe to a CSV file
merged_df.to_csv(folder_path + 'merged_dataset.csv', index=False)
