import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
import numpy as np
from datetime import timedelta

# Feature engineer/clean data
def add_column(df, col):
    pass

def isHoliday(df: pd.DataFrame)->pd.DataFrame:
    r"""
    Returns new dataframe with new column indicating if dates were
    on holidays

    args:
        df(pd.DataFrame): Original pandas Dataframe with column names: year, month, date

    returns:
        df_cleaned(pd.Dataframe): Dataframe with if holiday is included
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
    holiday_range = pd.date_range(start='2017-01-01', end='2020-12-31')
    dt = pd.to_datetime(dict(year=df.year, month=df.month, day=df.date), errors='coerce')
    cal = calendar()
    holidays = cal.holidays(start=holiday_range.min(), end=holiday_range.max())

    # Holiday checker
    df_cleaned = pd.concat([df, dt], axis=1) # Concats 
    df_cleaned.columns = [*df_cleaned.columns[:-1], 'y-m-d'] # Renames newly created column
    df_cleaned['is_holiday'] = df_cleaned['y-m-d'].isin(holidays)
    df_cleaned['is_holiday'] = df_cleaned['is_holiday'].astype(int)
    df_cleaned = df_cleaned.drop(['y-m-d'], axis=1)

    return df_cleaned

def clean_s3e7(df: pd.DataFrame)->pd.DataFrame:
    r"""
    Data preprocessor for Kaggle s3e7. 

    args:
        df(pd.DataFrame): Data frame for the s3e7 dataset

    returns:
        df_cleaned: Clean data frame
    """
    # Renaming columns to be more readable
    # Current unknowns: market_segment_type, 
    df.columns = ["id", 'num_adults', 'num_children', 'num_weekend_nights', 'num_week_nights', 'meal_plan', 
                'parking', 'room_type', 'lead_time', 'year', 'month', 'date', 'market_segment_type', 'repeated_customer',
                'num_prev_cancellations', 'num_prev_not_cancelled', 'avg_price_per_room', 'num_special_requests',
                'booking_status']
                
    # Holiday checker
    df = isHoliday(df)

    # Drops
    df = df.drop(['id'], axis=1) # id
    df = df.drop(['meal_plan'], axis=1)

    # Additions
    df['cancel_ratio'] = df['num_prev_cancellations']/(df['num_prev_cancellations']+df['num_prev_not_cancelled'])

    # Normalizing
    df['num_adults'] = df["num_adults"]/max(df['num_adults'])
    df['num_children'] = df["num_children"]/max(df['num_children'])
    df['num_weekend_nights'] = df['num_weekend_nights']/max(df['num_weekend_nights'])
    df['num_week_nights'] = df['num_week_nights']/max(df['num_week_nights'])
    df['lead_time'] = df['lead_time']/max(df['lead_time'])
    df['num_prev_cancellations'] = df['num_prev_cancellations']/max(df['num_prev_cancellations'])
    df['num_prev_not_cancelled'] = df['num_prev_not_cancelled']/max(df['num_prev_not_cancelled'])
    df['avg_price_per_room'] = df['avg_price_per_room']/max(df['avg_price_per_room'])
    df['num_special_requests'] = df['num_special_requests']/max(df['num_special_requests'])

    # Modifications
    df['cancel_ratio'] = df['cancel_ratio'].fillna(0)
    target_col = df.pop('booking_status')
    df.insert(df.shape[1], 'booking_status', target_col)

    return df

