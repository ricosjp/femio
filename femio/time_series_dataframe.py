
import numpy as np
import pandas as pd


class TimeSeriesDataFrame():

    def __init__(self, list_data_frame):
        if not isinstance(list_data_frame[0], (pd.DataFrame, pd.Series)):
            self._data_frame = [pd.DataFrame(f) for f in list_data_frame]
        else:
            self._data_frame = list_data_frame
        return

    @property
    def data_frame(self):
        return self._data_frame

    @property
    def index(self):
        return pd.Series(self._data_frame[0].index)

    @property
    def ndim(self):
        return self._data_frame[0].ndim

    @property
    def iloc(self):
        return TimeSeriesArray(self, method='iloc')

    @property
    def loc(self):
        return TimeSeriesArray(self, method='loc')

    @property
    def values(self):
        return np.stack([f.values for f in self._data_frame])

    @property
    def space_length(self):
        return len(self._data_frame[0])

    def __getitem__(self, key):
        return self._data_frame[key]


class TimeSeriesArray():

    def __init__(self, time_series_data_frame, method):
        self._data_frame = time_series_data_frame.data_frame
        self.method = method
        return

    def __getitem__(self, key):
        return TimeSeriesDataFrame([
            getattr(d, self.method)[key] for d in self._data_frame])
