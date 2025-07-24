import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("OnlineRetail.xlsx")

print(df.info())
print(df.head())
print(df.describe())

df = df.dropna(subset=["InvoiceNo","Quantity","UnitPrice"])
df = df[df["Quantity"] > 0]
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
df["Sales"] = df["Quantity"] * df["UnitPrice"]
df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
