# Author: Manas Saraptwar
# Date: 19/05/2021

from collections import namedtuple
import pandas as pd
import numpy as np
import config

# Lookback(past=[7, 14, 28], future=[7, 14, 28], years=2)
class Lookback:
    def __init__(self, past=[], future=[], years=0):
        if isinstance(past, int):
            past = list(range(1, past+1))
        if isinstance(future, int):
            future = list(range(1, future+1))

        self.past = past
        self.future = future
        self.years = years

    def __str__(self):
        return f"{self.length()}-{self.years}"

    def length(self):
         return (len(self.past)+1)+(len(self.past)+len(self.future)+1)*self.years

    def min_target_date(self, min_date, start_year, start_month, horizon):
        """
        min_target_date returns the minimum target date that is possible for the provided start year, month and horizon.

        :param min_date: pd.Timestamp minimum date of the train dataset
        :param start_year: Start year of the train dataset
        :param start_month: Start month of the train dataset
        :param horizon: Int number of days which it is predicting ahead, eg. 1, 14, 28
        :return: pd.Timestamp min date
        """

        if isinstance(min_date, str):
            min_date = pd.Timestamp(min_date)

        past = self.past[-1] if len(self.past) > 0 else 0
        min_date = min_date+pd.DateOffset(years=self.years, days=(past+horizon))
        target_date = max(min_date, pd.Timestamp(f"{start_year.year+self.years}-{start_month}-01"))

        return target_date
    
    def split_years(self, split, horizon):
        """
        split_years returns the train, valdidation and target years as a python slice, slice and array of pd.Timestamps respectively

        :param min_date: pd.Timestamp minimum date of the train dataset
        :param start_year: Start year of the train dataset
        :param start_month: Start month of the train dataset
        :param horizon: Int number of days which it is predicting ahead, eg. 1, 14, 28
        :return: pd.Timestamp min date
        """

        assert len(split) == 4, "Invalid split provided"
        years = list(map(str, split))
        years = list(map(pd.Timestamp, years))
        train_years = slice(years[0], years[1]-pd.DateOffset(days=horizon))
        validation_years = slice(years[1], years[2]-pd.DateOffset(days=horizon))
        test_years = pd.date_range(years[2], years[3], freq='W')
        return train_years, validation_years, test_years

    def dates(self, date, horizon):
        """
        dates returns the slice of min, max date for the given past, future and year values as well as
        the 1D mask over the temporal dimension

        :param date: Target date
        :param horizon: Int number of days which it is predicting ahead, eg. 1, 14, 28
        :return: Tuple (slice, mask) 
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)
        end = date-pd.DateOffset(days=horizon)
        dates = [end-pd.DateOffset(days=days) for days in reversed(self.past)]+[end]
        for year in range(self.years):
            start = end-pd.DateOffset(years=year+1)
            past = [start-pd.DateOffset(days=days) for days in reversed(self.past)]
            future = [start+pd.DateOffset(days=days) for days in self.future]
            dates = past+[start]+future+dates

        mask = pd.date_range(start=dates[0], end=dates[-1], freq='D').isin(dates)
        return slice(dates[0], dates[-1]), mask

def inverse(value, mean, std):
    return value*std+mean

def get_image(values, grid_shape, mask):
    """
    get_iamge returns the 2D image from the 1D predictions of the target locations
    the 1D mask over the temporal dimension

    :param values: 1D np.array Predictions/Target location values
    :param grid_shape: Tuple shape to reshape the values into
    :param mask: Boolean 1D mask to fill the 2D image with provided values
    :return: 2D np.array 
    """
    image = np.empty(grid_shape).flatten()
    image[:] = np.nan
    image[mask] = values
    image = np.flipud(image.reshape(*grid_shape))

    return image

def visualise_prediction(prediction, ground_truth, grid_shape, mask):
    return get_image(prediction, grid_shape, mask), get_image(ground_truth, grid_shape, mask)

def normalize(df, vmin=-1, vmax=1):
    return (vmax-vmin)*((df-df.min())/(df.max()-df.min()))+vmin

def get_label(target_feature):
    labels = {'precip': 'Rainfall anomaly in mm', 'tmp2m': 'Temperature anomaly in ° Celcius'}
    return labels[target_feature]

def get_feature_name(feature):
    labels = {'precip': 'rainfall', 'tmp2m': 'temperature'}
    return labels[feature]

def dataset_data(dataset):
    if dataset.upper() == 'IMDAA':
        return config.IMDAA
    elif dataset.upper() == 'RODEO':
        return config.RODEO
    return None