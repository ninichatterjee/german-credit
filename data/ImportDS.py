from ucimlrepo import fetch_ucirepo 
import pandas as pd
  
# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
  
# data (as pandas dataframes) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 
  
# metadata 
print(statlog_german_credit_data.metadata) 
  
# variable information 
print(statlog_german_credit_data.variables)

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save to CSV
df.to_csv('raw/german_credit.csv', index=False)
print("\nDataset saved to raw/german_credit.csv") 
