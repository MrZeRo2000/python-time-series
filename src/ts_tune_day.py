
from utility import ResultComposer, DataFramePreprocessor, ARMAOrderTuner
from statsmodels.tsa.arima_model import ARMA
import matplotlib.pyplot as plt
import pandas as pd

FACT_FIELD_NAME = "CNT_DRIVES"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
dfa_train, dfa_test, dfa = dfp.get_all_data()

# prediction by day tuning parameters
tuner = ARMAOrderTuner(dfa_train)
tuner.tune_bic(range(12), range(3))
tuner.plot_bic()
