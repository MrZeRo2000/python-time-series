
"""
Utility

Purpose:
    Utility classes

"""
import os
import pandas as pd
import math
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA


class ResultComposer:
    """Class to compose result data"""

    FORECAST_FIELD_NAME = "forecast"
    FACT_FIELD_NAME = "fact"

    def __init__(self, data_pred, data_fact, data_index, auto_calc=True):
        self.data_pred = data_pred
        self.data_fact = data_fact
        self.data_index = data_index
        self.data_result = None

        if auto_calc:
            self.calc()

    def calc(self):
        self.data_result = pd.DataFrame(self.data_pred)
        self.data_result.columns = [self.FORECAST_FIELD_NAME]
        self.data_result[self.FACT_FIELD_NAME] = self.data_fact
        self.data_result.index = self.data_index

    def get_error_percent(self):
        """Returns error in percent"""
        error_value = median_absolute_error(self.data_result[self.FACT_FIELD_NAME], self.data_result[self.FORECAST_FIELD_NAME])
        mean_value = self.data_result[self.FACT_FIELD_NAME].mean()
        return round(error_value*100/mean_value, 2)

    def get_data_result(self):
        return self.data_result

    def plot_data_result(self, title):
        self.data_result.plot()
        plt.title(title)
        plt.legend(title=title + ":" + str(str(self.get_error_percent())) + "%")
        plt.show()


class DataFramePreprocessor:
    def __init__(self, fact_field_name):
        self.fact_field_name = fact_field_name
        self.df_train = None
        self.df_test = None
        self.dfa = None

    def process(self):
        file_name = os.path.dirname(__file__)
        data_file_name = os.path.join(file_name, "../data", "DN_ML_TS_TRIPS.txt")

        df = pd.read_csv(data_file_name, ",")

        # limit by BER fleet
        df = df[df["FLEET_ID"] == "BER"]

        # drop unneeded columns
        df = df.drop(["FLEET_ID", "COMPANY_NO"], axis=1)

        # convert day_id to index column
        df.index = pd.to_datetime(df["DAY_ID"], format="%d/%m/%Y")
        df = df.drop(["DAY_ID"], axis=1)

        # leave only CNT_DRIVES for analysis
        df = df[[self.fact_field_name]]

        # align dataframe by weeks
        self.dfa = df.iloc[len(df) - math.trunc(len(df) / 7) * 7:]

        # split to train and test
        self.df_train = self.dfa[:-7]
        self.df_test = self.dfa[-7:]

    def get_train_data(self):
        if self.df_train is None:
            self.process()
        return self.df_train

    def get_test_data(self):
        if self.df_test is None:
            self.process()
        return self.df_test

    def get_data(self):
        if self.dfa is None:
            self.process()
        return self.dfa

    def get_all_data(self):
        if self.dfa is None:
            self.process()
        return self.df_train, self.df_test, self.dfa


class ARMAOrderTuner:
    def __init__(self, df):
        self.df = df
        self.df_bic = None

    def tune_bic(self, list_ar, list_ma):
        self.df_bic = pd.DataFrame()

        for ma in list_ma:
            bic = []
            for ar in list_ar:
                # skip ar <= 0
                if ar <= 0:
                    continue
                mod = ARMA(self.df, order=(ar, ma))
                try:
                    res = mod.fit()
                    bic.append(res.bic)
                except Exception as e:
                    bic.append(None)
                    print("*** ERROR ***")
                    print("Failed to calc for (ar,ma)=(" + str(ar) + "," + str(ma) + ")")
                    print("***  ***")
            column_name = "MA=" + str(ma)
            self.df_bic[column_name] = bic

        self.df_bic.index += min(i for i in list_ar if i > 0)

    def plot_bic(self):
        self.df_bic.plot()
        plt.title("BIC depending on AR and MA parameters")
        plt.xlabel("AR")
        plt.ylabel("BIC")
        plt.show()
