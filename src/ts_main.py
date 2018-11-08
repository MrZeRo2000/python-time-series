
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA

from utility import ResultComposer, DataFramePreprocessor


# plots show configuration
show_plots = True
# leave only CNT_DRIVES for analysis
FACT_FIELD_NAME = "CNT_DRIVES"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
dfa_train, dfa_test, dfa = dfp.get_all_data()

if show_plots:
    plot_acf(dfa, lags=50, alpha=0.05)
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


def plot_prediction(data_result, title):
    data_result.get_data_result().plot()
    plt.title(title)
    plt.legend(title=title + ":" + str(str(pred_week_result.get_error_percent())) + "%")
    plt.show()

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

    res.plot_predict(start="2018-10-06", end="2018-10-14")
    plt.title("Prediction by day via plot_predict")
    plt.show()

# prediction by day tuning parameters
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