import pandas as pd
import os

df = pd.read_parquet(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "gold",
        "dim_servicos.parquet",
    )
)
print(df.head(20))
