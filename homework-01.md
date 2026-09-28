# Homework 1

```python
import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"
df = pd.read_csv(url)

print(pd.__version__)
print(len(df))
print(df["fuel_type"].nunique())
print(df.isna().any().sum())

asia = df[df["origin"] == "Asia"]
print(asia["fuel_efficiency_mpg"].max())

before = df["horsepower"].median()
mode = df["horsepower"].mode()[0]
after = df["horsepower"].fillna(mode).median()
print(before, after)

X = asia[["vehicle_weight", "model_year"]].head(7).to_numpy()
y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])
w = np.linalg.inv(X.T @ X) @ X.T @ y
print(w.sum())
```

Answers: **2.3.3, 10000, 3, 2, 41.2, Yes it decreased, 0.369.**
