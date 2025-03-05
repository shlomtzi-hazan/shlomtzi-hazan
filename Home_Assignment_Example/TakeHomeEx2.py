import pandas as pd
import sqlite3

# File paths
folder_path = "/Users/shlomtzi/PycharmProjects/" # Change to your actual folder path
user_file = folder_path + "USER_TAKEHOME.csv"
transaction_file = folder_path + "TRANSACTION_TAKEHOME.csv"
product_file = folder_path + "PRODUCTS_TAKEHOME.csv"

# Load the data into pandas DataFrames and remove duplicates
df_users = pd.read_csv(user_file).drop_duplicates()
df_transactions = pd.read_csv(transaction_file) # we don't drop duplicates here as we want to keep all transactions
df_products = pd.read_csv(product_file).drop_duplicates() # we don't drop duplicates here as we want to keep all transactions

# Preprocess date columns to remove time component and 'Z' character
df_users['BIRTH_DATE'] = pd.to_datetime(df_users['BIRTH_DATE']).dt.date.astype(str).str.replace('Z', '')
df_users['CREATED_DATE'] = pd.to_datetime(df_users['CREATED_DATE']).dt.date.astype(str).str.replace('Z', '')
df_transactions['PURCHASE_DATE'] = pd.to_datetime(df_transactions['PURCHASE_DATE']).dt.date.astype(str).str.replace('Z', '')
df_transactions['SCAN_DATE'] = pd.to_datetime(df_transactions['SCAN_DATE']).dt.date.astype(str).str.replace('Z', '')

# Ensure BARCODE columns are integers
df_transactions['BARCODE'] = pd.to_numeric(df_transactions['BARCODE'], errors='coerce').fillna(0).astype(int)
df_products['BARCODE'] = pd.to_numeric(df_products['BARCODE'], errors='coerce').fillna(0).astype(int)

# Create an in-memory SQLite database
conn = sqlite3.connect(':memory:')

# Load the DataFrames into the SQLite database
df_users.to_sql('users', conn, index=False, if_exists='replace')
df_transactions.to_sql('transactions', conn, index=False, if_exists='replace')
df_products.to_sql('products', conn, index=False, if_exists='replace')

# Function to run a SQL query and return the result as a pandas DataFrame
def run_query(query):
    return pd.read_sql_query(query, conn)

# What are the top 5 brands *by receipts* scanned among users 21 and over?

# Assumptions:
# 1. A "receipt scanned" means a unique receipt ID appears in the transactions table.
# 2. Users must be at least 21 years old, calculated using birth date.
# 3. Each receipt may contain multiple products from different brands.
# 4. A brand is considered present on a receipt if at least one of its products is listed in the transaction.
# 5. We are ranking brands by the number of distinct receipts they appear on, not by total product scans.

# Methodology:
# - Join the users, transactions, and products tables to link users with their purchases.
# - Filter users to include only those aged 21+ using birth date calculation.
# - Use COUNT(DISTINCT t.RECEIPT_ID) to ensure each receipt is counted only once per brand.
# - Exclude NULL brands to focus on valid data.
# - Group by brand to aggregate the total number of receipts they appear on.
# - Order results in descending order to get the most frequently appearing brands.
# - Limit the output to the top 5 brands.

query1 = """
SELECT p.BRAND, COUNT(DISTINCT t.RECEIPT_ID) AS TOTAL_RECEIPTS
FROM users u 
LEFT JOIN transactions t 
ON u.ID = t.USER_ID 
LEFT JOIN products p 
ON t.BARCODE = p.BARCODE 
WHERE u.BIRTH_DATE < DATE('now', '-21 years')
AND p.BRAND IS NOT NULL
GROUP BY p.BRAND
ORDER BY TOTAL_RECEIPTS DESC
LIMIT 5;
"""
# Expected Output: 
"""
             BRAND  TOTAL_RECEIPTS
0      NERDS CANDY              14
1             DOVE              14
2  SOUR PATCH KIDS              13
3        HERSHEY'S              13
4        COCA-COLA              13
"""
# How this answers the question:
# - This query accurately identifies the top brands by the number of receipts they appear on.
# - It ensures receipts are counted uniquely per brand to avoid overestimation.
# - The applied filters ensure only relevant users and products are included.
# - The ranking gives a clear view of which brands are most frequently scanned by eligible users.

# This approach provides a reliable metric for brand popularity based on receipts scanned.


# What are the top 5 brands by sales among users that have had their account for at least six months?
# Assumptions:
# 1. "Sales" is interpreted as the number of **individual product purchases**, meaning each row in `transactions` represents a sale.
# 2. Users must have had their account for **at least six months**, determined using `CREATED_DATE`.
# 3. A product sale contributes to the brand's total sales count.
# 4. NULL brands are excluded to focus on valid data.
# 5. The top brands are ranked based on **total product sales**, not distinct receipts or revenue.

# Methodology:
# - Join `users`, `transactions`, and `products` to link users with their purchases.
# - Filter users whose accounts were created at least six months ago.
# - Use COUNT(*) to count total product sales per brand.
# - Exclude NULL brands.
# - Group by brand to aggregate sales numbers.
# - Order results in descending order to get the brands with the most sales.
# - Limit the output to the top 5 brands.

query2 = """
SELECT p.BRAND, COUNT(*) AS TOTAL_SALES
FROM users u 
LEFT JOIN transactions t 
ON u.ID = t.USER_ID 
LEFT JOIN products p 
ON t.BARCODE = p.BARCODE 
WHERE u.CREATED_DATE < DATE('now', '-6 months')
AND p.BRAND IS NOT NULL  
GROUP BY p.BRAND 
ORDER BY TOTAL_SALES DESC
LIMIT 5; 
"""

# Expected Output:
"""
                       BRAND  TOTAL_SALES
0  ANNIE'S HOMEGROWN GROCERY          576
1                       DOVE          558
2                   BAREFOOT          552
3                      ORIBE          504
4              SHEA MOISTURE          480
"""

# How this answers the question:
# - This query effectively identifies the top brands by total product sales among users with accounts older than six months.
# - Note, this query **does not consider monetary sales value**, just sales count. 
#    - If the question intended "sales" as total revenue, SUM(t.FINAL_QUANTITY * t.FINAL_SALE) would be needed instead of COUNT(*).
#    - However, both columns are problematic as FINAL_SALE column is lack of values for about 1/5 of the data, and FINAL_QUANTITY has the value "zero" along with numeric values.

# Which is the leading brand in the Dips & Salsa category?

# Assumptions:
# 1. "Leading brand" is determined by both **total product sales** and **how frequently a brand appears on receipts**.
# 2. The "Dips & Salsa" category is identified using the `CATEGORY_2` field.
# 3. A product sale contributes to the brand's **total sales count**.
# 4. A brand is considered present on a receipt if at least one of its products appears.
# 5. NULL brands are excluded to ensure meaningful results.
# 6. A **composite score** is used to balance both metrics:
#    - **60% weight on TOTAL_SALES** (indicating product popularity)
#    - **40% weight on TOTAL_RECEIPTS** (indicating brand presence across receipts)

# Methodology:
# - Use **two common table expressions (CTEs)**: 
#   1. `sales`: Counts total product sales per brand.
#   2. `receipts`: Counts distinct receipts where a brand appears.
# - Join both tables on `BRAND` to combine the metrics.
# - Compute `COMPOSITE_SCORE` as a weighted sum of sales and receipts.
# - Rank brands in descending order by `COMPOSITE_SCORE`.
# - Limit results to the **top 5 brands**.

# SQL Query:
query3 = """
WITH sales AS (
    SELECT p.BRAND, COUNT(*) AS TOTAL_SALES
    FROM users u 
    LEFT JOIN transactions t ON u.ID = t.USER_ID 
    LEFT JOIN products p ON t.BARCODE = p.BARCODE
    WHERE p.CATEGORY_2 LIKE 'Dips & Salsa' 
    AND p.BRAND IS NOT NULL 
    GROUP BY p.BRAND
),
receipts AS (
    SELECT p.BRAND, COUNT(DISTINCT t.RECEIPT_ID) AS TOTAL_RECEIPTS
    FROM users u 
    LEFT JOIN transactions t ON u.ID = t.USER_ID 
    LEFT JOIN products p ON t.BARCODE = p.BARCODE
    WHERE p.CATEGORY_2 LIKE 'Dips & Salsa'
    AND p.BRAND IS NOT NULL
    GROUP BY p.BRAND
)
SELECT s.BRAND, 
       s.TOTAL_SALES, 
       r.TOTAL_RECEIPTS,
       (s.TOTAL_SALES * 0.6 + r.TOTAL_RECEIPTS * 0.4) AS COMPOSITE_SCORE  -- Weighted score calculation
FROM sales s
JOIN receipts r ON s.BRAND = r.BRAND  
ORDER BY COMPOSITE_SCORE DESC 
LIMIT 5; 
"""

# Expected Output:
"""
            BRAND  TOTAL_SALES  TOTAL_RECEIPTS  COMPOSITE_SCORE
0        TOSTITOS          120              11             76.4
1  FRESH CRAVINGS           96              11             62.0
2          FRITOS           96              11             62.0
3     OLD EL PASO           96              11             62.0
4         ATHENOS           72              11             47.6
"""

# How does this answer the question?
# This query effectively identifies the **leading brand in the Dips & Salsa category**.
# It ranks brands **not just by sales volume but also by receipt presence**, giving a more holistic view of brand dominance.
# The **composite score balances** both product popularity and overall brand presence.
# The approach ensures **brands with strong sales and widespread presence** are ranked fairly.

# Next Steps:
# - If needed, adjust the **weighting** of TOTAL_SALES vs. TOTAL_RECEIPTS for different business priorities.
# - If available, incorporate **revenue (SUM(price))** to further refine brand ranking.

# Run the queries and display the results
df_query1 = run_query(query1)
df_query2 = run_query(query2)
df_query3 = run_query(query3)

print("Query 1 Result:")
print(df_query1)

print("Query 2 Result:")
print(df_query2)

print("Query 3 Result:")
print(df_query3)

# Close the database connection
conn.close()
