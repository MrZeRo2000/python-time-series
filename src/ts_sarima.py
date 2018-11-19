

import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
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

# prediction seasonal
mod = SARIMAX(dfa_train, order=(14, 0, 2), seasonal_order=(3, 0, 1, 7))
res = mod.fit()
y_pred = res.predict(start=dfa_test.index.min(), end=dfa_test.index.max()).values

result = ResultComposer(y_pred, dfa_test[FACT_FIELD_NAME].values, dfa_test.index)
if show_plots:
    result.plot_data_result("Prediction Seasonal")
    plt.show()
