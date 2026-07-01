import kagglehub
import pandas as pd

# Download latest version
path = kagglehub.dataset_download("nathanlauga/nba-games")

print("Path to dataset files:", path)

df = pd.read_csv(f"{path}/games.csv")
df.info()
print(df.head())

