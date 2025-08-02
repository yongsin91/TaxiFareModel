import pandas as pd

FILE_PATH = "raw_data/train.csv"
TEST_FILE_PATH = "raw_data/test.csv"

def get_data(nrows=10_000, test=False):
    """Return a DataFrame with ``nrows`` rows.

    Parameters
    ----------
    nrows : int, default=10_000
        Number of rows of file to read. This value is passed to
        :func:`pandas.read_csv`.
    test : bool, default=False
        Whether to load the test dataset instead of the training dataset.
    """

    file_path = TEST_FILE_PATH if test else FILE_PATH
    return pd.read_csv(file_path, nrows=nrows)


def clean_data(df, test=False):
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count > 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


if __name__ == '__main__':
    df = get_data()
