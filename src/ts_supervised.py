
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost


from utility import ResultComposer, DataFramePreprocessor, ARMAOrderTuner

# plots show configuration
show_plots = True
tune_parameters = False
# possible facts for analysis: "CNT_DRIVES","CNT_VEH_USED","SUM_INCOME_NETTO","SUM_KILOMETERS","SUM_MINUTES"
FACT_FIELD_NAME = "SUM_INCOME_NETTO"

dfp = DataFramePreprocessor(fact_field_name=FACT_FIELD_NAME)
dfa = dfp.get_data()

# added daily shifted data
dfar = dfa
for i in range(1, 40):
    dfar["fd" + str(i)] = dfa.shift(i)[FACT_FIELD_NAME]

# added weekly shifted data
for i in range(1, 3):
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

predictor = xgboost.XGBRegressor(n_estimators=10000, max_depth=3)

predictor.fit(x_train, y_train.values.ravel())
y_pred = predictor.predict(x_test)

pred_result = ResultComposer(y_pred, y_test.values, y_test.index)
if show_plots:
    pred_result.plot_data_result("Prediction via " + str(type(predictor).__name__))
    plt.show()

if show_plots:
    data_dmatrix = xgboost.DMatrix(data=x_train, label=y_train)
    params = {'learning_rate': 0.1, 'max_depth': 3}
    xg_reg = xgboost.train(params=params, dtrain=data_dmatrix, num_boost_round=10000)
    xgboost.plot_importance(xg_reg, max_num_features=20, title="Feature importance for " + str(type(predictor).__name__))
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()
"""

xgboost.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()


"""