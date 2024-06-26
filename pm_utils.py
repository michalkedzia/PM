import pandas as pd
import numpy as np
import ta
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

def create_sliding_windows_with_labels(data, labels, window_size):
    n_rows = data.shape[0]
    data_windows = []
    label_windows = []
    for i in range(n_rows - window_size + 1):
        end_ix = i + window_size
        if end_ix > n_rows:
            break
        seq_x = data[i:end_ix, :]
        seq_y = labels[end_ix - 1]
        data_windows.append(seq_x)
        label_windows.append(seq_y)

    return np.array(data_windows), np.array(label_windows)


def get_data(path, base_m=False, trade_metrics=False, trade_metrics_m=False, google=False, t=0, t_shift=0):
    data_columns = ['open', 'high', 'low', 'close', 'volume', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol']
    if trade_metrics:
        data_columns += ['sum_open_interest', 'sum_open_interest_value', 'count_long_short_ratio']

    if google:
        data_columns += ['Scale_extracted_value']

    df = pd.read_csv(path, usecols=data_columns)

    return_col = 'return'
    prev_close = 'prev_close'
    close = 'close'
    fillna = False
    high = 'high'
    volume = 'volume'
    low = 'low'
    sum_open_interest_value = 'sum_open_interest_value'

    df[prev_close] = df['close'].shift(-t_shift)
    df[return_col] = (df[prev_close] - df['open'])

    if base_m:
        data_columns += ['macd', 'sma', 'ema', 'wma', 'trix', 'adx', 'obv', 'adi', 'fi', 'mfi']
        df['macd'] = ta.trend.macd(close=df[close], window_slow=t, fillna=fillna)
        df['sma'] = ta.trend.sma_indicator(close=df[close], window=t, fillna=fillna)
        df['ema'] = ta.trend.ema_indicator(close=df[close], window=t, fillna=fillna)
        df['wma'] = ta.trend.wma_indicator(close=df[close], window=t, fillna=fillna)
        df['trix'] = ta.trend.trix(close=df[close], window=t, fillna=fillna)
        df['adx'] = ta.trend.adx(high=df[high], low=df[low], close=df[close], window=t, fillna=t)
        df['obv'] = ta.volume.on_balance_volume(close=df[close], volume=df[volume], fillna=t)
        df['adi'] = ta.volume.acc_dist_index(high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=t)
        df['fi'] = ta.volume.force_index(close=df[high], volume=df[volume], window=t, fillna=t)
        df['mfi'] = ta.volume.money_flow_index(high=df[high], low=df[low], close=df[close], volume=df[volume], window=t,
                                               fillna=fillna)
        df['atr'] = ta.volatility.average_true_range(high=df[high], low=df[low], close=df[close], window=t,
                                                     fillna=fillna)
    if trade_metrics_m:
        data_columns += ['tm_obv', 'tm_adi', 'tm_fi', 'tm_mfi']
        df['tm_obv'] = ta.volume.on_balance_volume(close=df[close], volume=df[sum_open_interest_value],
                                                   fillna=fillna)
        df['tm_adi'] = ta.volume.acc_dist_index(high=df[high], low=df[low], close=df[close],
                                                volume=df[sum_open_interest_value], fillna=fillna)
        df['tm_fi'] = ta.volume.force_index(close=df[high], volume=df[sum_open_interest_value], window=t, fillna=fillna)
        df['tm_mfi'] = ta.volume.money_flow_index(high=df[high], low=df[low], close=df[close],
                                                  volume=df[sum_open_interest_value], window=t, fillna=fillna)

    df.dropna(inplace=True)
    features = df[data_columns].values
    labels = df[return_col].values
    return data_columns, features, labels


class NDStandardScaler(TransformerMixin):
    def __init__(self, **kwargs):
        self._scaler = MinMaxScaler(copy=True, **kwargs)
        self._orig_shape = None

    def fit(self, X, y=None, **kwargs):  # Dodajemy y=None, aby zgodzić się z API scikit-learn
        X = np.array(X)
        # Save the original shape to reshape the flattened X later back to its original shape
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X, **kwargs)
        return self

    def transform(self, X, **kwargs):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X, **kwargs)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        # Reshape X to <= 2 dimensions
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        # Reshape X back to its original shape
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X
