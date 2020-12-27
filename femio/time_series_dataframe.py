
import numpy as np
import pandas as pd


class TimeSeriesDataFrame():

    def __init__(self, list_data_frame):
        self._data_frame = list_data_frame
        return

    @property
    def data_frame(self):
        return self._data_frame

    @property
    def index(self):
        return pd.Series(self._data_frame[0].index)

    @property
    def iloc(self):
        return TimeSeriesArray(self, method='iloc')

    @property
    def loc(self):
        return TimeSeriesArray(self, method='loc')


class TimeSeriesArray():

    def __init__(self, time_series_data_frame, method):
        self._data_frame = time_series_data_frame.data_frame
        self.method = method
        return

    def __getitem__(self, key):
        return np.stack([
            getattr(d, self.method)[key] for d in self._data_frame])
