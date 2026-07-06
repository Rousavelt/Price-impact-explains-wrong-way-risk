import pandas as pd
import numpy as np
import datetime as datetime

df = pd.read_excel("Archegos.xlsx", sheet_name = "DataActual6m")
df["Date"] = pd.to_datetime(df["Date"], format = "%d/%m/%Y")
df = df.set_index("Date")

print(df.head())

logreturns = np.log((df / df.shift(-1)).dropna())

logreturns.corr().to_pickle("correlations.pkl")

n = logreturns.shape[0] # Get number of dates

quadratic_covariation_matrix = logreturns.T @ logreturns * 260/n

print(quadratic_covariation_matrix)

vols = np.sqrt(quadratic_covariation_matrix.values.diagonal())

pd.DataFrame(vols, index = df.columns).to_pickle("vols.pkl")

correlations_2 = quadratic_covariation_matrix / np.einsum('i, j -> ij', vols, vols)

correlations_2.to_pickle("correlations.pkl")

means = np.log(df.iloc[0] / df.iloc[-1])  * 260 / n + 0.5 * vols * vols

pd.DataFrame(means).to_pickle("means.pkl")