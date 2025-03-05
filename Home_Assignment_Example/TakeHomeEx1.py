import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File paths
folder_path = "/Users/shlomtzi/PycharmProjects/" # Change to your actual folder path
user_file = folder_path+"USER_TAKEHOME.csv"
transaction_file = folder_path+"TRANSACTION_TAKEHOME.csv"
product_file = folder_path+"PRODUCTS_TAKEHOME.csv"

# Load the data
print("Loading datasets...")
df_users = pd.read_csv(user_file).drop_duplicates()
df_transactions = pd.read_csv(transaction_file)  # Keeping duplicates for transactions
df_products = pd.read_csv(product_file).drop_duplicates()

# Check for duplicate rows
duplicate_users = df_users.duplicated().sum()
duplicate_transactions = df_transactions.duplicated().sum()
duplicate_products = df_products.duplicated().sum()
len_duplicate_users = len(df_users)
len_duplicate_transactions = len(df_transactions)
len_duplicate_products = len(df_products)

# Remove duplicates, assuming that there might be valid multiple entries for transactions
df_users = df_users.drop_duplicates()
df_products = df_products.drop_duplicates()

# Display basic dataset information
df_users.info()
df_transactions.info()
df_products.info()

# Check for missing values
missing_values_users = df_users.isnull().sum()
missing_values_transactions = df_transactions.isnull().sum()
missing_values_products = df_products.isnull().sum()

# Calculate percentage of missing values
missing_percentage_users = (missing_values_users / len(df_users)) * 100
missing_percentage_transactions = (missing_values_transactions / len(df_transactions)) * 100
missing_percentage_products = (missing_values_products / len(df_products)) * 100

# Create DataFrames for plotting
missing_summary_users = pd.DataFrame({
    'Present': 100 - missing_percentage_users,
    'Missing': missing_percentage_users
})

missing_summary_transactions = pd.DataFrame({
    'Present': 100 - missing_percentage_transactions,
    'Missing': missing_percentage_transactions
})

missing_summary_products = pd.DataFrame({
    'Present': 100 - missing_percentage_products,
    'Missing': missing_percentage_products
})

missing_summary = {
    "Users": missing_values_users,
    "Transactions": missing_values_transactions,
    "Products": missing_values_products
}
print("Missing Values Summary:", missing_summary)

# Plot 100% stacked bar charts for missing values and save figures
plt.figure(figsize=(10, 6))
missing_summary_users.plot(kind='barh', stacked=True, color=["#619CFF", "#F8766D"])
plt.title("Percentage of Missing Values in Users Data")
plt.xlabel("Percentage")
plt.ylabel("Variable")
plt.legend(loc="best")
plt.savefig(folder_path + 'missing_values_users.png')
plt.show()

plt.figure(figsize=(10, 6))
missing_summary_transactions.plot(kind='barh', stacked=True, color=["#619CFF", "#F8766D"])
plt.title("Percentage of Missing Values in Transactions Data")
plt.xlabel("Percentage")
plt.ylabel("Variable")
plt.legend(loc="best")
plt.savefig(folder_path + 'missing_values_transactions.png')
plt.show()

plt.figure(figsize=(10, 6))
missing_summary_products.plot(kind='barh', stacked=True, color=["#619CFF", "#F8766D"])
plt.title("Percentage of Missing Values in Products Data")
plt.xlabel("Percentage")
plt.ylabel("Variable")
plt.legend(loc="best")
plt.savefig(folder_path + 'missing_values_products.png')
plt.show()

# Summarize duplicate records per column
duplicate_values_users = df_users.duplicated(subset=df_users.columns).sum()
duplicate_values_transactions = df_transactions.duplicated(subset=df_transactions.columns).sum()
duplicate_values_products = df_products.duplicated(subset=df_products.columns).sum()

# Print duplicate records summary
print("Duplicate Records Summary:")
print(f"Users: {duplicate_values_users} duplicates")
print(f"Transactions: {duplicate_values_transactions} duplicates")
print(f"Products: {duplicate_values_products} duplicates")

# Preprocess date columns to remove time component and 'Z' character
df_users['BIRTH_DATE'] = pd.to_datetime(df_users['BIRTH_DATE']).dt.date.astype(str).str.replace('Z', '')
df_users['CREATED_DATE'] = pd.to_datetime(df_users['CREATED_DATE']).dt.date.astype(str).str.replace('Z', '')
df_transactions['PURCHASE_DATE'] = pd.to_datetime(df_transactions['PURCHASE_DATE']).dt.date.astype(str).str.replace('Z', '')
df_transactions['SCAN_DATE'] = pd.to_datetime(df_transactions['SCAN_DATE']).dt.date.astype(str).str.replace('Z', '')

# Ensure BARCODE columns are integers
df_transactions['BARCODE'] = pd.to_numeric(df_transactions['BARCODE'], errors='coerce').fillna(0).astype(int)
df_products['BARCODE'] = pd.to_numeric(df_products['BARCODE'], errors='coerce').fillna(0).astype(int)

# Convert FINAL_QUANTITY and FINAL_SALE to numeric
df_transactions['FINAL_QUANTITY'] = df_transactions['FINAL_QUANTITY'].replace({'zero': 0}).astype(str)
df_transactions['FINAL_QUANTITY'] = pd.to_numeric(df_transactions['FINAL_QUANTITY'], errors='coerce')
df_transactions['FINAL_SALE'] = pd.to_numeric(df_transactions['FINAL_SALE'], errors='coerce')

# Identify unique values in challenging fields
unique_values_language = df_users['LANGUAGE'].unique()
unique_values_final_quantity = df_transactions['FINAL_QUANTITY'].unique()
unique_values_barcode_transactions = df_transactions['BARCODE'].unique()
unique_values_barcode_products = df_products['BARCODE'].unique()

# Display dataset statistics
print("Users Summary:\n", df_users.describe(include='all'))
print("Transactions Summary:\n", df_transactions.describe(include='all'))
print("Products Summary:\n", df_products.describe(include='all'))

# Additional Visualizations

# Distribution of Users by Age
df_users['AGE'] = pd.to_datetime('today').year - pd.to_datetime(df_users['BIRTH_DATE']).dt.year
plt.figure(figsize=(10, 6))
sns.histplot(df_users['AGE'], bins=30, kde=True)
plt.title("Distribution of Users by Age")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig(folder_path + 'distribution_users_age.png')
plt.show()

# Distribution of Users by Location (Top 10)
plt.figure(figsize=(10, 6))
top_states = df_users['STATE'].value_counts().nlargest(10)
sns.barplot(x=top_states.values, y=top_states.index, palette="viridis")
plt.title("Top 10 States in Users Data")
plt.xlabel("Count")
plt.ylabel("State")
plt.savefig(folder_path + 'distribution_users_location.png')
plt.show()

# Distribution of Transactions by Date
df_transactions['PURCHASE_DATE'] = pd.to_datetime(df_transactions['PURCHASE_DATE'])
plt.figure(figsize=(10, 6))
df_transactions['PURCHASE_DATE'].value_counts().sort_index().plot()
plt.title("Distribution of Transactions by Date")
plt.xlabel("Date")
plt.ylabel("Number of Transactions")
plt.savefig(folder_path + 'distribution_transactions_date.png')
plt.show()

# Distribution of Products by Category (Top 10)
plt.figure(figsize=(10, 6))
top_categories = df_products['CATEGORY_1'].value_counts().nlargest(10)
sns.barplot(x=top_categories.values, y=top_categories.index, palette="viridis")
plt.title("Top 10 Categories in Products Data")
plt.xlabel("Count")
plt.ylabel("Category")
plt.savefig(folder_path + 'distribution_products_category.png')
plt.show()

# Conclusions
print("\n### Data Quality Issues Identified ###")
print("\nUSER_TAKEHOME:")
print("- BIRTH_DATE and CREATED_DATE contained timestamps, now cleaned.")
print("- LANGUAGE column contains 'es-419' (Latin American Spanish), consider standardization.")
print("- Missing values in STATE, LANGUAGE, GENDER, and BIRTH_DATE.")

print("\nTRANSACTION_TAKEHOME:")
print("- FINAL_QUANTITY had mixed numeric formats, now standardized including converting 'zero' to 0.")
print("- FINAL_SALE had missing values where FINAL_QUANTITY > 0, requires further investigation.")
print("- PURCHASE_DATE and SCAN_DATE formatted differently, now standardized.")
print("- BARCODE has missing values; need to determine if -1 indicates invalid data.")

print("\nPRODUCTS_TAKEHOME:")
print("- Significant missing values in CATEGORY_4 and other attributes.")
print("- BARCODE values in products and transactions may not align perfectly.")

print("\n### Next Steps ###")
print("- Clarify the meaning of -1 in BARCODE fields.")
print("- Standardize LANGUAGE column if necessary.")
print("- Investigate FINAL_SALE inconsistencies in transactions.")
print("- Consider handling missing CATEGORY_4 values in products.")
