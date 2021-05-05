from collections import namedtuple
import pandas as pd
import numpy as np

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

    def extra_year(self, year, target_month, horizon):
        if not self.past:
            return False

        days = pd.date_range(str(year), f"{year}-{target_month}-01", freq='D', closed='left')
        return days.size-horizon-self.past[-1] < 0

    def min_target_date(self, min_date, start_year, start_month, horizon):
        if isinstance(min_date, str):
            min_date = pd.Timestamp(min_date)

        past = self.past[-1] if len(self.past) > 0 else 0
        min_date = min_date+pd.DateOffset(years=self.years, days=(past+horizon))
        target_date = max(min_date, pd.Timestamp(f"{int(start_year)+self.years}-{start_month}-01"))

        return target_date
    
    def split_years(self, split):
        split = list(map(str, split))
        train_years = slice(split[0], split[1])
        validation_years = slice(split[1], split[2])
        test_years = slice(split[2], split[3])

        return train_years, validation_years, test_years

    def dates(self, date, horizon):
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
        # return slice(dates[0], dates[-1]), mask
        return dates, mask

def inverse(value, mean, std):
    return value*std+mean

def get_image(values, grid_shape, mask):
    image = np.empty(grid_shape).flatten()
    image[:] = np.nan
    image[mask] = values
    image = np.flipud(image.reshape(*grid_shape))

    return image

def visualise_prediction(prediction, ground_truth, grid_shape, mask):
    return get_image(prediction, grid_shape, mask), get_image(ground_truth, grid_shape, mask)

def normalize(df, vmin=-1, vmax=1):
    return (vmax-vmin)*((df-df.min())/(df.max()-df.min()))+vmin

def get_labelY(target_feature):
    labels = {'rf': 'Rainfall in mm', 'temp': 'Temperature in ° Celcius', 'tmp2m': 'Temperature in ° Celcius'}
    return labels[target_feature]

def get_feature_name(feature):
    labels = {'rf': 'rainfall', 'temp': 'temperature', 'tmp2m': 'temperature'}
    return labels[feature]