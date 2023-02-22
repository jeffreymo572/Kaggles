import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

# Feature engineer/clean data
def add_column(df, col):
    pass

def clean_s3e7(df: pd.DataFrame)-> pd.DataFrame:
    r"""
    Data preprocessor for Kaggle s3e7. 

    args:
        df(pd.DataFrame): Data frame for the s3e7 dataset

    returns:
        df_cleaned: Clean data frame
    """
    # Holiday checker
    dt = pd.to_datetime(dict(year=df.year, month=df.month, day=df.date), errors='coerce')
    dt.columns = ['is_holiday']
    df_cleaned = pd.concat([df, dt], axis=1)
    holiday_range = pd.date_range(start='2015-07-01', end='2015-07-31')
    df['is_holiday'] = 

    # Drops
    df_cleaned = df.drop(['id'], axis=1) # id

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

    return df_cleaned

