import pandas as pd

# File paths
input_file = "/content/uniprotkb_mouse_AND_model_organism_1009_2025_04_08.xlsx"  # Replace with your Excel file path
output_file = "Dataset/Yeast/GO_annotation_yeast.csv"  # Specify the desired output CSV file path

# Read the Excel file into a DataFrame
df = pd.read_excel(input_file)

# Specify the columns to process
columns_to_count = [
    "Gene Ontology (biological process)",
    "Gene Ontology (molecular function)",
    "Gene Ontology (cellular component)",
]

# Function to count semicolons and add 1
def count_items(cell):
    if pd.isna(cell):  # Handle missing values
        return 0
    return cell.count(";") + 1

# Apply the function to each specified column
for column in columns_to_count:
    if column in df.columns:
        df[f"{column} Count"] = df[column].apply(count_items)
    else:
        print(f"Warning: Column '{column}' not found in the Excel file.")

# Drop the original columns
df = df.drop(columns=columns_to_count, errors="ignore")
df = df.drop(columns='Entry', errors="ignore")
df = df.rename(columns={"Entry Name": "Protein"})
# Save the updated DataFrame to a CSV file
df.to_csv(output_file, index=False)

print(f"Processing complete. Updated file saved to {output_file}")