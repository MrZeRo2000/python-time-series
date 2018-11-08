
"""
Utility

Purpose:
    Utility classes

"""

import pandas as pd
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt

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
