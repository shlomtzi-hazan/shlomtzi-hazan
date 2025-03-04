import pandas as pd
import numpy as np
import os

# File paths
folder_path = "/Users/shlomtzi/PycharmProjects/"
merged_dataset = folder_path + "merged_dataset.csv"

# Load the merged dataset
print("Loading merged dataset...")
merged_df = pd.read_csv(merged_dataset)
print(f"Merged dataset loaded: {merged_df.shape} rows and columns")

# Define the specific case we're investigating
PROCEDURE_CODE = "43239"
HOSPITAL_NAME = "Montefiore Medical Center"
PAYER_NAME = "Aetna"
PRICE_POINT = 1246.73

# 1. Check if this specific case exists in the dataset
print("\n" + "="*50)
print("CASE INVESTIGATION: Endoscopy Biopsy Procedure (43239)")
print("="*50)

# First, let's check the hospital dataset portion
hospital_matches = merged_df[
    (merged_df['raw_code'] == PROCEDURE_CODE) & 
    (merged_df['hospital_name'].str.contains(HOSPITAL_NAME, case=False, na=False)) &
    (merged_df['payer_name'].str.contains(PAYER_NAME, case=False, na=False))
]

print(f"\n1. Hospital dataset matches: {len(hospital_matches)} rows")
if not hospital_matches.empty:
    print("\nExample hospital entries:")
    print(hospital_matches[['hospital_name', 'raw_code', 'payer_name', 'standard_charge_negotiated_dollar']].head())
    
    # Check for the specific price point
    price_match = hospital_matches[
        (hospital_matches['standard_charge_negotiated_dollar'] == PRICE_POINT)
    ]
    
    if not price_match.empty:
        print(f"\nFound exact price match ({PRICE_POINT}):")
        print(price_match[['hospital_name', 'raw_code', 'payer_name', 'standard_charge_negotiated_dollar', 'description']].head())
    else:
        print(f"\nNo exact match for price point {PRICE_POINT}, closest values:")
        sorted_by_price = hospital_matches.sort_values(by='standard_charge_negotiated_dollar')
        print(sorted_by_price[['hospital_name', 'raw_code', 'payer_name', 'standard_charge_negotiated_dollar']].head())

# Now check the payer dataset portion
payer_matches = merged_df[
    (merged_df['code'] == PROCEDURE_CODE) & 
    (merged_df['payer'].str.contains(PAYER_NAME, case=False, na=False))
]

print(f"\n2. Payer dataset matches for code {PROCEDURE_CODE} with {PAYER_NAME}: {len(payer_matches)} rows")
if not payer_matches.empty:
    print("\nExample payer entries:")
    print(payer_matches[['payer', 'code', 'rate']].head())
    
    # Filter for matches with the Montefiore EIN
    if 'ein' in payer_matches.columns:
        montefiore_ein = hospital_matches['ein'].iloc[0] if not hospital_matches.empty else None
        if montefiore_ein is not None:
            ein_matches = payer_matches[payer_matches['ein'] == montefiore_ein]
            print(f"\nPayer entries for Montefiore (EIN={montefiore_ein}): {len(ein_matches)} rows")
            if not ein_matches.empty:
                print(ein_matches[['payer', 'code', 'rate', 'ein']].head())

# 3. Check if this case is among the flagged data points
print("\n" + "="*50)
print("FLAGGED STATUS CHECK")
print("="*50)

# Find case in the merged dataset
case_in_merged = merged_df[
    ((merged_df['raw_code'] == PROCEDURE_CODE) | (merged_df['code'] == PROCEDURE_CODE)) & 
    ((merged_df['hospital_name'].str.contains(HOSPITAL_NAME, case=False, na=False)) | 
     (merged_df['standard_charge_negotiated_dollar'] == PRICE_POINT)) &
    ((merged_df['payer_name'].str.contains(PAYER_NAME, case=False, na=False)) | 
     (merged_df['payer'].str.contains(PAYER_NAME, case=False, na=False)))
]

print(f"\nFound {len(case_in_merged)} potential matches in merged dataset")

if not case_in_merged.empty:
    flagged_cases = case_in_merged[case_in_merged['flagged'] == True]
    print(f"\nOf these, {len(flagged_cases)} are flagged as discrepancies")
    
    if not flagged_cases.empty:
        print("\nDetails of flagged discrepancies:")
        for idx, row in flagged_cases.iterrows():
            print(f"\n--- Row {idx} ---")
            print(f"Hospital: {row.get('hospital_name', 'N/A')}")
            print(f"Payer: {row.get('payer', 'N/A')} / {row.get('payer_name', 'N/A')}")
            print(f"Procedure: {row.get('code', 'N/A')} / {row.get('raw_code', 'N/A')} - {row.get('description', 'N/A')}")
            print(f"Hospital price: ${row.get('standard_charge_negotiated_dollar', 'N/A')}")
            print(f"Payer rate: ${row.get('rate', 'N/A')}")
            print(f"Absolute difference: ${row.get('delta', 'N/A')}")
            print(f"Percent difference: {row.get('delta_perc', 'N/A')}%")
            print(f"Discrepancy level: {row.get('discrepancy_level', 'N/A')}")

# 4. Provide guidance on rate selection and alignment
print("\n" + "="*50)
print("ANALYSIS & RECOMMENDATIONS")
print("="*50)

# Gather all rates for comparison
if not payer_matches.empty:
    all_rates = payer_matches['rate'].dropna().unique()
    rate_count = len(all_rates)
    
    print(f"\nFound {rate_count} different rates in the payer dataset for code {PROCEDURE_CODE} with {PAYER_NAME}:")
    
    if rate_count > 0:
        all_rates_sorted = sorted(all_rates)
        print(f"  - Min: ${min(all_rates_sorted)}")
        print(f"  - Max: ${max(all_rates_sorted)}")
        print(f"  - Median: ${np.median(all_rates_sorted)}")
        print(f"  - Mean: ${np.mean(all_rates_sorted)}")
        print(f"  - Standard deviation: ${np.std(all_rates_sorted)}")
        
        if PRICE_POINT in all_rates_sorted:
            print(f"\nThe hospital price point (${PRICE_POINT}) MATCHES one of the payer rates.")
        else:
            closest_rate = min(all_rates_sorted, key=lambda x: abs(x - PRICE_POINT))
            percent_diff = ((closest_rate - PRICE_POINT) / PRICE_POINT) * 100
            print(f"\nThe hospital price point (${PRICE_POINT}) DOES NOT MATCH any payer rate exactly.")
            print(f"Closest payer rate is ${closest_rate} (difference of {percent_diff:.2f}%)")

# 5. Recommendations for alignment
print("\n" + "="*50)
print("RECOMMENDATIONS FOR ALIGNMENT")
print("="*50)

print("""
When comparing hospital and payer rates for the same procedure, consider the following strategies:

1. Filtering Criteria:
   - Network specificity: Filter for the specific insurance network (e.g., Aetna Commercial)
   - Region/location: Compare only rates applicable to the same geographic region
   - Provider type: Ensure hospital vs. provider designations are properly matched
   - Place of Service: Match in-patient vs. outpatient vs. ambulatory settings

2. Recommended Match Approaches:
   - Use EIN as the primary facility identifier rather than name matching
   - Consider rate modifiers (many payer rates have specific modifiers)
   - Look for rate patterns and clusters instead of exact matches
   - Apply negotiation type and billing code filters

3. For this specific case (CPT 43239 at Montefiore):
   - Investigate whether different modifiers apply to this procedure
   - Check if the $1246.73 rate applies to a specific setting/network combination
   - Consider the possibility that the rate agreement has changed between data collection periods
""")

print("\nAnalysis complete.")
