# 📊 Data Quality Report & Business Insights

## **🔍 Overview**
This report summarizes key data quality findings and insights from my analysis of Users, Transactions, and Products datasets. 
Below, I outline data integrity concerns, highlight an interesting trend, and provide next steps for resolution.

## **🚨 Key Data Quality Issues**
### **User Data**
- Missing values in `STATE`, `LANGUAGE`, `GENDER`, and `BIRTH_DATE`.
- `LANGUAGE` column includes regional codes (e.g., `es-419`), which may require standardization.

### **Transaction Data**
- `FINAL_QUANTITY` contained mixed numeric and text formats (e.g., "zero" instead of `0`), which has now been standardized.
- `FINAL_SALE` has missing values where `FINAL_QUANTITY > 0`, requiring clarification.
- `PURCHASE_DATE` and `SCAN_DATE` were in different formats, now corrected.
- Some transactions have `BARCODE = -1`, which requires clarification.

### **Product Data**
- Significant missing values in `CATEGORY_4`, manufacturer, and brand details.
- `BARCODE` values in the Products and Transactions datasets do not align perfectly, potentially causing issues in joins.

## **📈 Data Insights**
### **1️⃣ Top Categories by Product Sales**
The **Health & Wellness** category dominates sales volume, significantly outperforming other categories like Snacks and Beverages. 
This suggests a strong consumer preference in this segment.

![Top Categories](images/distribution_products_category.png)

### **2️⃣ Transaction Trends Over Time**
Transactions exhibit notable fluctuations, with clear periodic peaks and declines over time. This may indicate promotional periods or seasonal shopping behaviors.

![Transaction Trends](images/distribution_transactions_date.png)

### **3️⃣ User Age Distribution**
A large concentration of users falls in the 25-45 age range, with distinct peaks at key life stages. This could provide insights into targeted marketing strategies.

![User Age Distribution](images/distribution_users_age.png)

### **4️⃣ User Geographic Distribution**
The highest number of users are concentrated in Texas, Florida, and California, making these critical markets for engagement.

![User Location Distribution](images/distribution_users_location.png)

### **5️⃣ Gender Distribution**
The dataset shows a significantly higher number of female users compared to male users.

![Gender Distribution](images/gender_distribution_users.png)

### **6️⃣ Store Transactions**
Walmart accounts for the vast majority of scanned receipts, indicating a dominant retail partner.

![Store Transactions](images/store_name_distribution_transactions.png)

### **7️⃣ Top Brands**
Certain brands, such as **REM BRAND**, dominate in total transactions.

![Top Brands](images/brand_distribution_products.png)

## **❓ Outstanding Questions & Next Steps**
### **Questions for Stakeholders**
- What does `BARCODE = -1` signify? Should it be treated as missing or invalid data?
- Should we standardize `LANGUAGE` values for consistency across reports?
- Can we clarify why some `FINAL_SALE` values are missing even when products were purchased?
- Should missing `CATEGORY_4` data be inferred, or is it safe to ignore this column?

### **Actionable Next Steps**
- Clarify the meaning of `-1` in `BARCODE` fields.
- Decide whether `LANGUAGE` should be standardized.
- Investigate **FINAL_SALE inconsistencies** in transactions.
- Address **CATEGORY_4 missing values** in product data.

## **📌 How to Use This Repository**
- The dataset files are stored in CSV format.
- Python scripts are available for data cleaning, visualization, and SQL-based analysis.
- The `notebooks/` directory contains Jupyter Notebooks with exploratory data analysis.

📩 **Feedback & Questions**
If you have any questions or need further analysis, feel free to open an issue or contact us.
