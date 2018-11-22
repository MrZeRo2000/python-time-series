
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
import sys


from utility import ResultComposer, DataFramePreprocessor, ARMAOrderTuner, FeatureProcessor

# plots show configuration
show_plots = True
show_analysis = False
tune_parameters = False
# possible facts for analysis: "CNT_DRIVES","CNT_VEH_USED","SUM_INCOME_NETTO","SUM_KILOMETERS","SUM_MINUTES"
FACT_FIELD_NAME = "SUM_INCOME_NETTO"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
df_train, df_test = dfp.get_train_test_data()

XGBOOST_PARAM_ESTIMATORS = 10000
XGBOOST_PARAM_DEPTH = 3
XGBOOST_PARAM_LEARNING_RATE = 0.1

predictor = xgboost.XGBRegressor(n_estimators=XGBOOST_PARAM_ESTIMATORS, max_depth=XGBOOST_PARAM_DEPTH, learning_rate=XGBOOST_PARAM_LEARNING_RATE)


def show_result(x_train, y_train, y_pred, y_test, title):
    pred_result = ResultComposer(y_pred, y_test.values, y_test.index)
    if show_plots:
        pred_result.plot_data_result(title)
        plt.show()

    if show_analysis:
        data_dmatrix = xgboost.DMatrix(data=x_train, label=y_train)
        params = {'learning_rate': XGBOOST_PARAM_LEARNING_RATE, 'max_depth': XGBOOST_PARAM_DEPTH}
        xg_reg = xgboost.train(params=params, dtrain=data_dmatrix, num_boost_round=XGBOOST_PARAM_ESTIMATORS)
        xgboost.plot_importance(xg_reg, max_num_features=20, title=title)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()


def predict_incremental(df_train, df_test, features, labels):
    y_pred = np.empty((0,))

    x_train = df_train[features]
    y_train = df_train[labels]

    y_test = df_test[labels]

    for d in range(0, 7):
        predictor.fit(x_train, y_train.values.ravel())

        x_test = x_train[-1:][["weekday_" + str(s) for s in range(1, 7)]]
        x_test.index += timedelta(days=1)

        for i in range(0, 39):
            x_test["fd" + str(i+1)] = y_train.shift(i)[-1:][FACT_FIELD_NAME][0]

        for i in range(0, 1):
            x_test["fw" + str(i+1)] = y_train.shift(i*7)[FACT_FIELD_NAME][0]


        y_pred_result = predictor.predict(x_test)

        x_train = x_train.append(x_test)

        y_train_new = y_train[-1:]
        y_train_new.iloc[0][0] = y_pred_result.ravel()
        y_train_new.index += timedelta(days=1)

        y_train = y_train.append(y_train_new)

        y_pred = np.append(y_pred, y_pred_result.ravel())

    show_result(x_train, y_train, y_pred, y_test, "Incremental prediction via " + str(type(predictor).__name__))


def predict_weekday(df_train, features, labels):
    x_train = df_train[features]
    y_train = df_train[labels]

    x_test = df_train[-7:][features]
    y_test = df_test[labels]

    predictor.fit(x_train, y_train.values.ravel())
    y_pred = predictor.predict(x_test)

    show_result(x_train, y_train, y_pred, y_test, "Weekdays prediction via " + str(type(predictor).__name__))


fp = FeatureProcessor(df_train)

df_shifted, features_shifted, labels_shifted = fp.get_shifted_data(FACT_FIELD_NAME)
df_weekdays, features_weekdays, labels_weekdays = fp.get_weekdays_data(FACT_FIELD_NAME)

predict_incremental(df_shifted, df_test, features_shifted, labels_shifted)
predict_weekday(df_weekdays, features_weekdays, labels_weekdays)