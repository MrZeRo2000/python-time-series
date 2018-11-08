
from utility import ResultComposer, DataFramePreprocessor
from statsmodels.tsa.arima_model import ARMA
import matplotlib.pyplot as plt
import pandas as pd

FACT_FIELD_NAME = "CNT_DRIVES"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
dfa_train, dfa_test, dfa = dfp.get_all_data()

# prediction by day tuning parameters
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

df_bic.plot()
plt.title("BIC depending on AR and MA parameters")
plt.xlabel("AR")
plt.ylabel("BIC")
plt.show()
