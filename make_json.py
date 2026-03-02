import pandas as pd
import json

# CSV load karo
df = pd.read_csv("data.csv")

# Kisi specific row ka number yaha change kar sakte ho
row_number = 45   # 0 = first row, 35 = second row, etc.

# Row select karo
row = df.iloc[row_number]

# Drop target column (Churn Value), kyunki predict karna hai
if "Churn Value" in row.index:
    row = row.drop("Churn Value")

# Row ko dictionary me convert karo
row_dict = row.to_dict()

# JSON format me pretty print karo
json_output = json.dumps(row_dict, indent=4)

print(json_output)