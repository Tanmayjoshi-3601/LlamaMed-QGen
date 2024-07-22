import pandas as pd

df1 = pd.read_csv("mcqs_1_200.csv")
df2 = pd.read_csv("mcqs_201_400.csv")
df3 = pd.read_csv("mcqs_401_700.csv")


# Combine the DataFrames
combined_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv('mcq_dataset_final.csv', index=False)