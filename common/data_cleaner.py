import pandas as pd
import holidays

# Feature engineer/clean data
def add_column(df, col):
    pass

def is_holiday(month:str, day:str, year:str)->bool:
    us_holidays = holidays.UnitedStates()


def clean_s3e7(df: pd.DataFrame)-> pd.DataFrame:
    r"""
    Data preprocessor for Kaggle s3e7. 

    args:
        df(pd.DataFrame): Data frame for the s3e7 dataset

    returns:
        df_cleaned: Clean data frame
    """
    # Holiday checker
    month = df['month']
    year = df['year']
    day = df['date']
    us_holidays = holidays.UnitedStates()

    # Drops
    df_cleaned = df.drop(['id'], axis=1) # id

    # Additions
    df['cancel_ratio'] = df['num_prev_cancellations']/(df['num_prev_cancellations']+df['num_prev_not_cancelled'])
    df['is_holiday'] = f"{year}-{month}-{day}" in us_holidays # TODO

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

