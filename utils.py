import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


def gap_check(df, column_name, start=None, end=None, freq='H'):
    """Fill the df with indexes of expected freq and return the dataframe with NaN values (gaps)
    """
    start = start or df.index.min()
    end = end or df.index.max()
    expected_index = pd.date_range(start=start, end=end, freq=freq)
    df_extended = df.reindex(expected_index, copy=False)
    # print(df_extended.loc[df_extended[column_name].isna()])
    # print(f'length of gaps: {len(df_extended.loc[df_extended[column_name].isna()])}')
    return df_extended

def create_train_val_test_split(dataframe, train_start=None, train_end=None, test_start=None, test_end=None):
    """
    Ignore the validation set because we will do cross validation on the entire training set to aovid overfiting the validation set
    """
    train_end = train_end or int(0.85 * len(dataframe))
    train_start = train_start or 0
    test_end = test_end or -1
    test_start = test_start or train_end

    train_df = dataframe.iloc[train_start:train_end]
    test_df = dataframe.iloc[test_start:test_end]
    return train_df, test_df


def evaluate_forecast(y_true, y_pred):
    """
    y_true (np.array): m * n matrix, m = # of samples, n = # of targets, e.g. forecast 3 days ahead = 72 hours, n=72.
    y_pred (np.array):
    metric (str): 'RMSE' = 'root_mean_squared_error' or 'MAPE' = mean_absolute_percentage_error
    """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    rmse_raw = mean_squared_error(y_true, y_pred, squared=False, multioutput='raw_values')
    mape_raw = mean_absolute_percentage_error(y_true, y_pred, multioutput='raw_values')
    return {'rmse': rmse, 'mape': mape, 'rmse_raw': rmse_raw, 'mape_raw': mape_raw}