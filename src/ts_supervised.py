
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime
from datetime import timedelta
import numpy as np
import xgboost


from utility import ResultComposer, DataFramePreprocessor, ARMAOrderTuner

# plots show configuration
show_plots = True
show_analysis = False
tune_parameters = False
# possible facts for analysis: "CNT_DRIVES","CNT_VEH_USED","SUM_INCOME_NETTO","SUM_KILOMETERS","SUM_MINUTES"
FACT_FIELD_NAME = "SUM_INCOME_NETTO"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
dfa = dfp.get_data()

dfar = dfa

# added daily shifted data

for i in range(1, 40):
    dfar["fd" + str(i)] = dfa.shift(i)[FACT_FIELD_NAME]

# added weekly shifted data
for i in range(1, 2):
    dfar["fw" + str(i)] = dfa.shift(i*7)[FACT_FIELD_NAME]    

# drop null values
dfar = dfar.dropna()

# add weekdays
dfar_dow = pd.get_dummies(dfar.index.dayofweek, prefix="weekday", drop_first=True)
dfar_dow.index = dfar.index

dfar_1 = pd.concat([dfar_dow, dfar], axis=1)

# columns and labels
features = [s for s in dfar_1.columns if s != FACT_FIELD_NAME]
labels = [FACT_FIELD_NAME]

x_train = dfar_1[:-7][features]
y_train = dfar_1[:-7][labels]

x_test = dfar_1[-7:][features]
y_test = dfar_1[-7:][labels]

predictor = xgboost.XGBRegressor(n_estimators=1000, max_depth=3)

predictor.fit(x_train, y_train.values.ravel())
y_pred = predictor.predict(x_test)

pred_result = ResultComposer(y_pred, y_test.values, y_test.index)
if show_plots:
    pred_result.plot_data_result("Prediction via " + str(type(predictor).__name__))
    plt.show()


y_pred_2 = np.empty((0,))

for d in range(0, 7):

    predictor.fit(x_train, y_train.values.ravel())

    x_test_2 = x_train[-1:][["weekday_" + str(s) for s in range(1, 7)]]
    x_test_2.index += timedelta(days=1)

    for i in range(0, 39):
        x_test_2["fd" + str(i+1)] = y_train.shift(i)[-1:][FACT_FIELD_NAME][0]

    for i in range(0, 1):
        x_test_2["fw" + str(i+1)] = y_train.shift(i*7)[FACT_FIELD_NAME][0]


    y_pred_result = predictor.predict(x_test_2)

    x_train = x_train.append(x_test_2)

    y_train_new = y_train[-1:]
    y_train_new.iloc[0][0] = y_pred_result.ravel()
    y_train_new.index += timedelta(days=1)

    y_train = y_train.append(y_train_new)

    y_pred_2 = np.append(y_pred_2, y_pred_result.ravel())

pred_result = ResultComposer(y_pred_2, y_test.values, y_test.index)

if show_plots:
    pred_result.plot_data_result("Prediction via " + str(type(predictor).__name__))
    plt.show()

if show_analysis:
    data_dmatrix = xgboost.DMatrix(data=x_train, label=y_train)
    params = {'learning_rate': 0.1, 'max_depth': 3}
    xg_reg = xgboost.train(params=params, dtrain=data_dmatrix, num_boost_round=10000)
    xgboost.plot_importance(xg_reg, max_num_features=20, title="Feature importance for " + str(type(predictor).__name__))
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
