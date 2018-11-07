
import os
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import median_absolute_error
import math



file_name = os.path.dirname(__file__)
data_file_name = os.path.join(file_name, "../data", "DN_ML_TS_TRIPS.txt")

show_plots = True

df = pd.read_csv(data_file_name, ",")

# limit by BER fleet
df = df[df["FLEET_ID"] == "BER"]

# drop unneeded columns
df = df.drop(["FLEET_ID", "COMPANY_NO"], axis=1)

# convert day_id to index column
df.index = pd.to_datetime(df["DAY_ID"], format="%d/%m/%Y")
df = df.drop(["DAY_ID"], axis=1)

# leave only CNT_DRIVES for analysis
df = df[["CNT_DRIVES"]]

# calculate percent changed and differences just to check
df["CNT_DRIVES_CHANGE_PCT"] = df["CNT_DRIVES"].pct_change()
df["CNT_DRIVES_CHANGE_DIFF"] = df["CNT_DRIVES"].diff()

if show_plots:
    df[["CNT_DRIVES_CHANGE_PCT"]].plot()
    plt.show()

dfa = df[["CNT_DRIVES"]]
dfa = dfa.dropna()

if show_plots:
    plot_acf(dfa, lags=50, alpha=0.05)
    plt.show()

# out of interest only
# adfuller_result = adfuller(df["CNT_DRIVES"])
# print(adfuller_result[1])

'''
# Not needed anymore
mod = ARMA(dfa, order=(7, 0))
res = mod.fit()

y_true = dfa.loc['2018-10-08':'2018-10-14']["CNT_DRIVES"].values
y_pred = res.predict(start="2018-10-08", end="2018-10-14").values

print("Prediction by day error:" + str(round(median_absolute_error(y_true, y_pred)/y_true.mean() * 100, 2)) + "%")

if show_plots:
    res.plot_predict(start="2018-10-08", end="2018-10-14")
    plt.title("Prediction by day")
    plt.legend(title="Prediction by day error:" + str(round(median_absolute_error(y_true, y_pred)/y_true.mean() * 100, 2)) + "%")
    plt.show()

'''

# align dataframe by weeks
dfa = dfa.iloc[len(dfa) - math.trunc(len(dfa) / 7) * 7:]

# split to train and test
dfa_train = dfa[:-7]
dfa_test = dfa[-7:]

# resample by week
dfar = []
y_pred = []
for shift in range(7):
    dfar.append(dfa_train[shift:].resample('7D').first())
    mod = ARMA(dfar[shift], order=(7, 0))
    res = mod.fit()
    y_pred.append(res.predict(start=dfar[shift].index.max())[0])

dfa_result = pd.DataFrame(y_pred)
dfa_result.columns = ["forecast"]
dfa_result["CNT_DRIVES"] = dfa_test["CNT_DRIVES"].values
dfa_result.index = dfa_test.index

if show_plots:
    dfa_result.plot()
    plt.title("Prediction by week")
    plt.legend(title="Prediction by week error:" + str(round(median_absolute_error(dfa_result["CNT_DRIVES"], dfa_result["forecast"])/dfa_result["CNT_DRIVES"].mean() * 100, 2)) + "%")
    plt.show()

print("Prediction by week error:" + str(round(median_absolute_error(dfa_result["CNT_DRIVES"], dfa_result["forecast"])/dfa_result["CNT_DRIVES"].mean() * 100, 2)) + "%")

# prediction by day
mod = ARMA(dfa_train, order=(7, 2))
res = mod.fit()

y_pred = res.predict(start="2018-10-08", end="2018-10-14").values

dfa_result = pd.DataFrame(y_pred)
dfa_result.columns = ["forecast"]
dfa_result["CNT_DRIVES"] = dfa_test["CNT_DRIVES"].values
dfa_result.index = dfa_test.index

if show_plots:
    dfa_result.plot()
    plt.title("Prediction by day")
    plt.legend(title="Prediction by day error:" + str(round(median_absolute_error(dfa_result["CNT_DRIVES"], dfa_result["forecast"])/dfa_result["CNT_DRIVES"].mean() * 100, 2)) + "%")
    plt.show()

'''
if show_plots:
    res.plot_predict(start="2018-10-06", end="2018-10-14")
    plt.title("Prediction by day via plot_predict")
    plt.show()
'''
