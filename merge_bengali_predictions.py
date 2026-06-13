import pandas as pd
import json
import os

def merge_submissions(original_csv="submission_extended.csv", 
                      bengali_csv="submission_bengali_only.csv", 
                      output_csv="final_submission.csv"):
    """
    Merges the original submission with the improved Bengali predictions.
    This script assumes that both CSVs have a common identifier column like 'page_id'.
    """
    print(f"Attempting to merge '{original_csv}' and '{bengali_csv}'...")
    
    if not os.path.exists(original_csv):
        print(f"Error: Original submission file '{original_csv}' not found.")
        return
    if not os.path.exists(bengali_csv):
        print(f"Error: Bengali predictions file '{bengali_csv}' not found.")
        return

    try:
        orig_df = pd.read_csv(original_csv)
        beng_df = pd.read_csv(bengali_csv)
        
        id_col = None
        # Determine the identifier column
        if 'page_id' in orig_df.columns and 'page_id' in beng_df.columns:
            id_col = 'page_id'
        elif 'image_id' in orig_df.columns and 'image_id' in beng_df.columns:
            id_col = 'image_id'
        elif 'id' in orig_df.columns and 'id' in beng_df.columns:
            id_col = 'id'
            
        if id_col:
            orig_df.set_index(id_col, inplace=True)
            beng_df.set_index(id_col, inplace=True)
            
            # Update original dataframe with bengali rows
            # Only updates non-NA values from beng_df
            orig_df.update(beng_df)
            
            orig_df.reset_index(inplace=True)
            orig_df.to_csv(output_csv, index=False)
            print(f"Successfully merged. Final output saved to '{output_csv}'")
        else:
            print("Could not find a common identifier column ('page_id', 'image_id', or 'id').")
            print("Please adjust the script based on your actual column names.")
            
    except Exception as e:
        print(f"An error occurred during merging: {e}")

if __name__ == "__main__":
    merge_submissions()
