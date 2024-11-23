import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../WA_Fn-UseC_-Telco-Customer-Churn.csv')

df_all = df.copy()
df = df[[i for i in df.columns if i not in ('customerID','Churn')]]
df = df.drop(columns=["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"])
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df_n = pd.get_dummies(df, drop_first=True)
corr_matrix = df_n.corr()

# Correlation Matrix Heatmap
df_numeric = (corr_matrix.corr())
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric, annot=False, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix Heatmap')
# plt.savefig('correlation_matrix_heatmap.png')
plt.show()