
from utility import ResultComposer, DataFramePreprocessor, ARMAOrderTuner
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
import sys

# plots show configuration
show_plots = True
# leave only CNT_DRIVES for analysis
FACT_FIELD_NAME = "CNT_DRIVES"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
dfa_train, dfa_test, dfa = dfp.get_all_data()

dfd_train = dfa_train.diff().dropna()

if show_plots:
    plot_acf(dfd_train, lags=50, alpha=0.05)
    plt.title("ADF P-value:" + str(adfuller(dfd_train[FACT_FIELD_NAME].values)[1]))
    plt.show()

mod = ARMA(dfd_train, order=(14, 1))
# mod = ARIMA(dfd_train, order=(7, 0, 3))
res = mod.fit()
y_pred_d = res.predict(start="2018-10-08", end="2018-10-14").values

y_last = dfa_train[dfa_train.columns[0]].values[-1]
y_pred = []
for i in range(0, 7):
    y_current = y_last + y_pred_d[i]
    y_pred.append(y_current)
    y_last = y_current

day_diff_result = ResultComposer(y_pred, dfa_test[FACT_FIELD_NAME].values, dfa_test.index)
if show_plots:
    day_diff_result.plot_data_result("Prediction by day differences")
    plt.show()

# prediction by day tuning parameters
tuner = ARMAOrderTuner(dfd_train)
tuner.tune_bic(range(15), range(3))
tuner.plot_bic()
