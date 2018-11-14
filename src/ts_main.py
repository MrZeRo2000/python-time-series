
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller

from utility import ResultComposer, DataFramePreprocessor, ARMAOrderTuner

# plots show configuration
show_plots = True
tune_parameters = False
# possible facts for analysis: "CNT_DRIVES","CNT_VEH_USED","SUM_INCOME_NETTO","SUM_KILOMETERS","SUM_MINUTES"
FACT_FIELD_NAME = "SUM_INCOME_NETTO"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
dfa_train, dfa_test, dfa = dfp.get_all_data()

if show_plots:
    plot_acf(dfa, lags=50, alpha=0.05)
    plt.title("ADF P-value:" + str(adfuller(dfa_train[FACT_FIELD_NAME].values)[1]))
    plt.show()
    plot_pacf(dfa, lags=50, alpha=0.05)
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

# prediction by week
dfar = []
y_pred = []
for shift in range(7):
    dfar.append(dfa_train[shift:].resample('7D').first())
    mod = ARMA(dfar[shift], order=(2, 0))
    res = mod.fit()
    y_pred.append(res.forecast()[0])

pred_week_result = ResultComposer(y_pred, dfa_test[FACT_FIELD_NAME].values, dfa_test.index)
if show_plots:
    pred_week_result.plot_data_result("Prediction by week")

# previous periods prediction
y_pred = dfa_train[-7:]

pred_prev_result = ResultComposer(y_pred, dfa_test[FACT_FIELD_NAME].values, dfa_test.index)
if show_plots:
    pred_prev_result.plot_data_result("Prediction by previous period")

# prediction by day
mod = ARMA(dfa_train, order=(7, 2))
res = mod.fit()
y_pred = res.predict(start="2018-10-08", end="2018-10-14").values

pred_day_result = ResultComposer(y_pred, dfa_test[FACT_FIELD_NAME].values, dfa_test.index)
if show_plots:
    pred_day_result.plot_data_result("Prediction by day")
    plt.show()
"""
    res.plot_predict(start="2018-10-06", end="2018-10-14")
    plt.title("Prediction by day via plot_predict")
    plt.show()
"""

# prediction by day using differences
dfd_train = dfp.get_diff_train_data()

if show_plots:
    plot_acf(dfd_train, lags=50, alpha=0.05)
    plt.title("ADF P-value:" + str(adfuller(dfd_train[FACT_FIELD_NAME].values)[1]))
    plt.show()
    plot_pacf(dfa, lags=50, alpha=0.05)
    plt.show()

mod = ARMA(dfd_train, order=(14, 1))
res = mod.fit()
y_pred_d = res.predict(start="2018-10-08", end="2018-10-14").values

y_pred = dfp.get_pred_from_diff(y_pred_d)

day_diff_result = ResultComposer(y_pred, dfa_test[FACT_FIELD_NAME].values, dfa_test.index)
if show_plots:
    day_diff_result.plot_data_result("Prediction by day differences")
    plt.show()


# prediction by week using differences
dfdar = []
y_pred_d = []
for shift in range(7):
    dfdar.append(dfd_train[shift:].resample('7D').first())
    mod = ARMA(dfdar[shift], order=(2, 0))
    res = mod.fit()
    y_pred_d.append(res.forecast()[0])

y_pred = dfp.get_pred_from_diff(y_pred_d)

pred_diff_week_result = ResultComposer(y_pred, dfa_test[FACT_FIELD_NAME].values, dfa_test.index)
if show_plots:
    pred_diff_week_result.plot_data_result("Prediction by differences by week")


# prediction by day tuning parameters

# prediction by day tuning parameters
if tune_parameters:
    tuner = ARMAOrderTuner(dfa_train)
    tuner.tune_bic(range(12), range(3))
    tuner.plot_bic()


"""
df_bic = pd.DataFrame()

for ma in range(3):
    bic = []
    for ar in range(1, 12):
        mod = ARMA(dfa_train, order=(ar, ma))
        res = mod.fit()
        bic.append(res.bic)
    column_name = "MA=" + str(ma)
    df_bic[column_name] = bic

df_bic.index += 1

if show_plots:
    df_bic.plot()
    plt.title("BIC depending on AR and MA parameters")
    plt.xlabel("AR")
    plt.ylabel("BIC")
    plt.show()
"""