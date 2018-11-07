
import os
import pandas as pd
from unittest import TestCase

class TestResample(TestCase):
    def setUp(self):
        self.script_dir = os.path.dirname(__file__)
        self.data_file_name  = os.path.join(self.script_dir, "../data", "DN_ML_TS_TRIPS.txt")
        self.df = pd.read_csv(self.data_file_name, ",")

        # limit by BER fleet
        self.df = self.df[self.df["FLEET_ID"] == "BER"]

        # drop unneeded columns
        self.df = self.df.drop(["FLEET_ID", "COMPANY_NO"], axis=1)

        # convert day_id to index column
        self.df.index = pd.to_datetime(self.df["DAY_ID"], format="%d/%m/%Y")
        self.df = self.df.drop(["DAY_ID"], axis=1)

        # leave only CNT_DRIVES for analysis
        self.df = self.df[["CNT_DRIVES"]]

    def test1(self):
        print(self.df.head(14))

        dfr = self.df["CNT_DRIVES"].resample('W').last()
        print(dfr)

