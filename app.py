import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("Zimbabwe Aircraft Movements Forecast Dashboard")

df = pd.read_csv("Monthly_Expanded_WholeNumbers.csv")
df['Date'] = pd.to_datetime(df[['Year','Month']].assign(DAY=1))
df = df.sort_values('Date')
df['TimeIndex'] = np.arange(len(df))

target = 'Total'
X = df[['TimeIndex']]
y = df[target]

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X, y)

future_index = np.arange(len(df), len(df)+120)
future_pred = model.predict(future_index.reshape(-1,1))
future_dates = pd.date_range(df['Date'].iloc[-1], periods=121, freq='M')[1:]

fig, ax = plt.subplots(figsize=(12,5))
ax.plot(df['Date'], df['Total'], label="Historical")
ax.plot(future_dates, future_pred, label="Forecast")
ax.legend()
ax.set_title("10-Year Aircraft Movements Forecast")

st.pyplot(fig)

st.write("### Forecast Data Table")
forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": future_pred})
st.dataframe(forecast_df)