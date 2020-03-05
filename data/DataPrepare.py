# 衍生规则
import time
import pandas as pd
import os
import numpy as np
import talib

class DataPrepare(object):

    # 初始化
    def __init__(self, instruments,
                 starting_cash=1e7,
                 train_start_date=None,
                 train_end_date='2017-01-01',
                 data_local_path=r'D:\Repos\xgb_tsQuant\data\DataTest.xlsx',
                 commission_fee=0.001,
                 ):
        print(os.getcwd())
        start = time.clock()
        self.backtest_type_list = ['open_to_open', 'open_to_close', 'vwap_to_vwap']
        self.instruments = instruments
        self.starting_cash = starting_cash
        self.train_end_date = train_end_date
        self.comminssion_fee = commission_fee
        self.index_dict = self._init_market_data(data_local_path)
        self.train_df, self.target_df, self.test_dict, self.target_dict = self._init_train_data()
        print('数据初始化成功，耗时：{}'.format(time.clock() - start))

    def _init_market_data(self, data_local_path):
        index_dict = {}
        for index in self.instruments:
            index_dict[index] = pd.read_excel(data_local_path, sheet_name=index)
        for index in index_dict.keys():
            index_dict[index]['date'] = pd.to_datetime(index_dict[index]['Unnamed: 0'])
            del index_dict[index]['Unnamed: 0']
            index_dict[index].set_index('date', inplace=True)
            index_dict[index]['open_to_open'] = index_dict[index]['open'].shift(-2) / index_dict[index]['open'].shift(
                -1) - 1
            index_dict[index]['open_to_close'] = index_dict[index]['close'].shift(-1) / index_dict[index]['open'].shift(
                -1) - 1
            index_dict[index] = index_dict[index].dropna()
        return index_dict

        # 生成train_df,target_df,test_dict,target_dict

    def _init_train_data(self):
        base_var = ['close', 'high', 'low', 'open', 'volume']
        test_dict = {}
        target_dict = {}
        train_df = pd.DataFrame()
        for index in self.instruments:
            train_df_temp = self.index_dict[index][base_var]
            result_df = self.create_fea(train_df_temp)
            result_df = pd.concat([self.index_dict[index], result_df], axis=1)
            result_df = result_df.T.drop_duplicates().T
            factor_list = [factor for factor in list(result_df.columns) if factor not in self.backtest_type_list]
            # train_df设置: 大表result_df中不带backtest相关的部分
            if len(train_df) == 0:
                train_df = result_df[factor_list].loc[:self.train_end_date]
                target_df = result_df[self.backtest_type_list].loc[:self.train_end_date]
            else:
                train_df = pd.concat([train_df, result_df[factor_list].loc[:self.train_end_date]], axis=0)
                target_df = pd.concat([target_df, result_df[self.backtest_type_list].loc[:self.train_end_date]], axis=0)

            # test_dict设置
            test_dict[index] = result_df[factor_list].loc[self.train_end_date:]

            # test_dict和target_dict设置
            target_dict[index] = result_df[['close', 'open', 'high', 'low', 'volume'] + self.backtest_type_list].loc[
                                 self.train_end_date:]
            target_dict[index]['open_shift'] = target_dict[index]['open'].shift(-1)
            target_dict[index]['high_shift'] = target_dict[index]['high'].shift(-1)
            target_dict[index]['low_shift'] = target_dict[index]['low'].shift(-1)
            target_dict[index]['atr'] = talib.ATR(target_dict[index]['high'], target_dict[index]['low'],
                                                  target_dict[index]['close'], 20)

        return train_df, target_df, test_dict, target_dict

    def _init_target_data(self):
        target_dict = {}
        for index in self.instruments:
            target_dict[index] = self.test_dict[index][
                ['close', 'open', 'high', 'low', 'volume'] + self.backtest_type_list]
            target_dict[index]['open_shift'] = target_dict[index]['open'].shift(-1)
            target_dict[index]['high_shift'] = target_dict[index]['high'].shift(-1)
            target_dict[index]['low_shift'] = target_dict[index]['low'].shift(-1)
            target_dict[index]['atr'] = talib.ATR(target_dict[index]['high'], target_dict[index]['low'],
                                                  target_dict[index]['close'], 20)
        return target_dict

    # 加入外部数据：比如vwap_to_vwap，比如不能用目前手段生成的因子
    def external_data(self, external_data_path):
        pass

    def ts_rank(self,x):
        return pd.Series(x).rank().tail(1)

    def ts_rankeq10(self,x):
        res = (x == 10) * 1
        return res.sum()

    def create_fea(self, price_df):
        price = price_df
        open, close, high, volume, low = price_df['open'], price_df['close'], price_df['high'], price_df['volume'], \
                                         price_df['low']
        cols = price.pipe(self.devfea, high, low, close, open, volume).columns
        df = (price.pipe(self.devfea, high, low, close, open, volume)
                          .pipe(self.devfea_roll,cols,5)
                          .pipe(self.devfea_roll,cols,10)
                          .pipe(self.devfea_roll,cols,20)
                          .pipe(self.devfea_roll,cols,30)
                          .pipe(self.devfea_roll,cols,60)
                          .pipe(self.devfea_diff,cols,1)
                          .pipe(self.devfea_diff,cols,5)
                          .pipe(self.devfea_diff,cols,10)
                          .pipe(self.devfea_diff,cols,20)
                          .pipe(self.devfea_diff,cols,30)
                          .pipe(self.devfea_diff,cols,60)
                          .pipe(self.devfea_lag,cols,1)
                          .pipe(self.devfea_lag,cols,2)
                          .pipe(self.devfea_lag,cols,3)
                          .pipe(self.devfea_lag,cols,5)
                          .pipe(self.devfea_diff2,cols)
                          )

        df['long_MA5_flag'] = (df['close'] > df['MA5']) * 1
        df['long_MA10_flag'] = (df['close'] > df['MA10']) * 1
        df['long_MA20_flag'] = (df['close'] > df['MA20']) * 1
        df['long_MA5_MA10'] = (df['MA5'] > df['MA10']) * 1
        df['long_MA5_MA20'] = (df['MA5'] > df['MA20']) * 1
        # alpha001 量价协方差，量价背离#
        df['aphla001'] = df['close'].rolling(10).corr(df['volume'])
        # Alpha002 开盘缺口
        df['alpha002'] = df['open'] / df['close']
        # Alpha003 异常交易量
        df['alpha003'] = -1 * df['volume'] / df['volume'].rolling(20).mean()
        # Alpha004 量幅背离
        df['alpha004'] = (df['high'] / df['close']).rolling(10).corr(df['volume'])
        return df

    def devfea(self, df, high, low, close, open, volume):
        # 技术指标
        df['AD'] = talib.AD(high, low, close, volume)
        df['CCI'] = talib.CCI(high, low, close)
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df['ATR'] = talib.ATR(high, low, close, timeperiod=14)
        df['ADOSC'] = talib.ADOSC(high, low, close, volume)
        df['ADX'] = talib.ADX(high, low, close)
        df['BBANDS_upper'], df['BBANDS_mid'], df['BBANDS_lower'] = talib.BBANDS(close)
        df['RSI'] = talib.RSI(close)
        df['MA5'] = talib.MA(close, 5)
        df['MA10'] = talib.MA(close, 10)
        df['MA20'] = talib.MA(close, 20)
        df['OBV'] = talib.OBV(close, volume)
        df['SAR'] = talib.SAR(high, low)
        df['lgvol'] = np.log(volume)
        df['upshadow'] = np.abs(high - ((open + close) + (np.abs(open - close))) / 2)
        df['downshadow'] = np.abs(low - ((open + close) - (np.abs(open - close))) / 2)
        return df

    def devfea_roll(self, df, cols, ndays):
        fea = df[cols].rolling(ndays).agg(['mean', 'max', 'min', 'std', 'var', 'median'])
        fea.columns = ["_".join(col) for col in fea.columns]
        fea.columns = fea.columns + '_rl_' + str(ndays) + 'D'
        res = pd.merge(df, fea, left_index=True, right_index=True, how='inner')
        return res

    def devfea_diff(self, df, cols, ndays):
        fea = df[cols].diff(ndays)
        fea.columns = fea.columns + '_diff_' + str(ndays) + 'D'
        res = pd.merge(df, fea, left_index=True, right_index=True, how='inner')
        return res

    def devfea_diff2(self, df, cols):
        fea = df[cols].diff(1).diff(1)
        fea.columns = fea.columns + '_diff2'
        res = pd.merge(df, fea, left_index=True, right_index=True, how='inner')
        return res

    def devfea_lag(self, df, cols, ndays):
        fea = df[cols].shift(ndays)
        fea.columns = fea.columns + '_lag_' + str(ndays) + 'D'
        res = pd.merge(df, fea, left_index=True, right_index=True, how='inner')
        return res

