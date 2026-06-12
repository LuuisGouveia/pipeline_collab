import pandas as pd
import os
import fastparquet as fp

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "gold",
    "dim_clientes.parquet",
)

df = pd.read_parquet(
    path,
    engine="fastparquet",
)
print(df.tail(40))
