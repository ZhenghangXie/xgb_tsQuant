
class Evaluation(object):

    def __init__(self, backtest_type='open_to_open', commission_fee=0.001):
        self.backtest_type = backtest_type
        pass

        # 模型预测

    def predict(self, test_dict, target_dict, model):
        #
        self.target_dict = target_dict.copy()
        # 逐行运行
        for index in test_dict.keys():
            yvalid_list = []
            print('processing', index)
            x_valid = test_dict[index]
            print(x_valid)
            for m in model:
                dvalid = xgb.DMatrix(x_valid)
                yvalid_list.append(m.predict(dvalid))
                self.target_dict[index]['yvalid'] = np.array([i for i in yvalid_list]).mean(axis=0)
        return self.target_dict[index]['yvalid']

    # 模型预测
    def predict_by_date(self, index, test_dict, start_date, end_date, model):
        '''
        输入模型，得到一列yvalid
        '''
        # 逐行运行
        yvalid_list = []
        print('processing', index)
        x_valid = test_dict[index].loc[start_date:end_date]
        for m in model:
            dvalid = xgb.DMatrix(x_valid)
            yvalid_list.append(m.predict(dvalid))
            yvalid = list(np.array([i for i in yvalid_list]).mean(axis=0))
        return yvalid

    # 根据预测结果生成交易信号
    def gen_trading_signal(self, index, target_dict):
        '''
        输入带有模型预测值的target_dict，输出信号列表
        '''
        #
        yvalid = target_dict[index]['yvalid']
        target_dict = target_dict.copy()

        # 纯数分类，
        def get_long_num_sig(self, posi_line):
            signal_list = (yvalid > posi_line) * 1
            return signal_list

        def get_long_short_num_sig(self, posi_line, nega_line):
            signal_list = (yvalid > posi_line) * 1 + (yvalid < posi_line) * -1
            return signal_list

        # 纯数+下穿
        def get_long_num_band_sig(self, posi_line, nega_line):
            position = pd.DataFrame()
            position['open'] = (yvalid >= posi_line) * 1
            position['close'] = (yvalid <= nega_line) * 1
            signal_lst = []
            t = 0
            for index, row in position.iterrows():
                if row['open'] == 1:
                    signal_lst.append(1)
                    t = 1
                elif row['close'] == 1:
                    signal_lst.append(0)
                    t = 0
                elif row['open'] != 1 and row['close'] != 1:
                    signal_lst.append(t)
            return signal_lst

        def get_long_short_num_band_sig(self, posi_line, nega_line):
            position = pd.DataFrame()
            position['open'] = (yvalid >= posi_line) * 1
            position['close'] = (yvalid <= nega_line) * 1
            signal_lst = []
            t = 0
            for index, row in position.iterrows():
                if row['open'] == 1:
                    signal_lst.append(1)
                    t = 1
                elif row['close'] == 1:
                    signal_lst.append(0)
                    t = -1
                elif row['open'] != 1 and row['close'] != 1:
                    signal_lst.append(t)
            return signal_lst

        # 布林带异常捕捉类信号
        def get_long_boll_band_sig(self, window, weight):
            boll_temp_df = pd.DataFrame()
            boll_temp_df['yvalid'] = yvalid
            boll_temp_df['MA'] = boll_temp_df['yvalid'].rolling(window).mean()
            boll_temp_df['std'] = boll_temp_df['yvalid'].rolling(window).std()
            boll_temp_df['p_boll'] = (boll_temp_df['yvalid'] > (boll_temp_df['MA'] + boll_temp_df['std'] * weight)) * 1
            signal_list = boll_temp_df['p_boll']
            return signal_list

        def get_long_short_boll_band_sig(self, window, weight):
            boll_temp_df = pd.DataFrame()
            boll_temp_df['yvalid'] = yvalid
            boll_temp_df['MA'] = boll_temp_df['yvalid'].rolling(window).mean()
            boll_temp_df['std'] = boll_temp_df['yvalid'].rolling(window).std()
            boll_temp_df['p_boll'] = (boll_temp_df['yvalid'] > (boll_temp_df['MA'] + boll_temp_df['std'] * weight)) * 1
            boll_temp_df['n_boll'] = (boll_temp_df['yvalid'] < (boll_temp_df['MA'] - boll_temp_df['std'] * weight)) * 1
            signal_list = boll_temp_df['p_boll'] - boll_temp_df['n_boll']
            return signal_list

        # 正常布林带
        def get_normal_long_boll_sig(self, window, weight):
            boll_temp_df = pd.DataFrame()
            signal_list = []
            t = 0
            boll_temp_df['yvalid'] = yvalid
            boll_temp_df['MA'] = boll_temp_df['yvalid'].rolling(window).mean()
            boll_temp_df['std'] = boll_temp_df['yvalid'].rolling(window).std()
            boll_temp_df['p_boll'] = (boll_temp_df['yvalid'] > (boll_temp_df['MA'] + boll_temp_df['std'] * weight)) * 1
            for index, row in boll_temp_df.iterrows():
                if row['p_boll'] == 1:
                    t = 1
                    signal_list.append(t)
                elif row['yvalid'] < row['MA']:
                    t = 0
                    signal_list.append(t)
                else:
                    signal_list.append(t)
            return signal_list

        def get_normal_long_short_boll_sig(sig, window, weight):
            boll_temp_df = pd.DataFrame()
            signal_list = []
            t = 0
            boll_temp_df['yvalid'] = yvalid
            boll_temp_df['MA'] = boll_temp_df['yvalid'].rolling(window).mean()
            boll_temp_df['std'] = boll_temp_df['yvalid'].rolling(window).std()
            boll_temp_df['p_boll'] = (boll_temp_df['yvalid'] > (boll_temp_df['MA'] + boll_temp_df['std'] * weight)) * 1
            boll_temp_df['n_boll'] = (boll_temp_df['yvalid'] < (boll_temp_df['MA'] - boll_temp_df['std'] * weight)) * 1
            for index, row in boll_temp_df.iterrows():
                if row['p_boll'] == 1:
                    t = 1
                    signal_list.append(t)
                elif row['n_boll'] == 1:
                    t = -1
                    signal_list.append(t)
                else:
                    signal_list.append(t)
            return signal_list

            # 唐奇安通道

        def get_long_dc_sig(self, window):
            dc_temp_df = pd.DataFrame()
            signal_list = [0]
            t = 0
            dc_temp_df['yvalid'] = yvalid
            dc_temp_df['max'] = dc_temp_df['yvalid'].rolling(window).max()
            dc_temp_df['MA'] = dc_temp_df['yvalid'].rolling(window).mean()
            for i in range(len(dc_temp_df) - 1):
                if dc_temp_df['yvalid'][i] > dc_temp_df['max'][i - 1]:
                    t = 1
                    signal_list.append(t)
                elif dc_temp_df['yvalid'][i] < dc_temp_df['MA'][i]:
                    t = 0
                    signal_list.append(t)
                else:
                    signal_list.append(t)
            return signal_list

        def get_long_short_dc_sig(self, window):
            dc_temp_df = pd.DataFrame()
            signal_list = [0]
            t = 0
            dc_temp_df['yvalid'] = yvalid
            dc_temp_df['max'] = dc_temp_df['yvalid'].rolling(window).max()
            dc_temp_df['min'] = dc_temp_df['yvalid'].rolling(window).min()
            for i in range(len(dc_temp_df) - 1):
                if dc_temp_df['yvalid'][i] > dc_temp_df['max'][i - 1]:
                    t = 1
                    signal_list.append(t)
                elif dc_temp_df['yvalid'][i] < dc_temp_df['MA'][i - 1]:
                    t = -1
                    signal_list.append(t)
                else:
                    signal_list.append(t)
            return signal_list

        target_dict[index]['signal_1'] = get_long_num_sig(self, posi_line=0)
        target_dict[index]['signal_2'] = get_long_num_sig(self, posi_line=0.001)
        target_dict[index]['signal_3'] = get_long_num_sig(self, posi_line=0.002)
        target_dict[index]['signal_4'] = get_long_num_band_sig(self, posi_line=0.001, nega_line=-0.001)
        target_dict[index]['signal_5'] = get_long_num_band_sig(self, posi_line=0.002, nega_line=-0.002)
        target_dict[index]['signal_6'] = get_long_boll_band_sig(self, window=20, weight=1)
        target_dict[index]['signal_7'] = get_long_boll_band_sig(self, window=60, weight=1)
        target_dict[index]['signal_8'] = get_normal_long_boll_sig(self, window=20, weight=1)
        target_dict[index]['signal_9'] = get_normal_long_boll_sig(self, window=60, weight=1)
        target_dict[index]['signal_10'] = get_long_dc_sig(self, window=20)
        target_dict[index]['signal_11'] = get_long_dc_sig(self, window=60)

        return target_dict[index]

    # 生成收益率序列
    def get_return_series(self, signal_list, target_series, commission_fee=0.0005, stop_method=None, stop_param=None):
        '''
        输入：信号列表，目标函数列表，手续费率，止损参数
        输出：供分析的收益率序列，每次交易的盈亏序列
        '''
        return_series = []  # 收益率序列
        return_series_by_trade = []  # 单笔交易盈亏序列
        trade_tmp = 1
        for i in range(len(signal_list)):
            # 第一笔交易
            if i == 0:
                if signal_list[i] != 0:
                    return_series.append(target_series[i] - signal_list[i] * commission_fee)
                    trade_tmp = return_series[-1]
                else:
                    return_series.append(0)
            else:
                # 仓位不变
                if signal_list[i - 1] == signal_list[i]:
                    trade_tmp = trade_tmp * target_series[i] * signal_list[i]
                    return_series.append(signal_list[i] * target_series[i])
                else:
                    # 空仓变持仓
                    if signal_list[i - 1] == 0 and signal_list[i] != 0:
                        return_series.append(target_series[i] - signal_list[i] * commission_fee)
                        trade_tmp = return_series[-1]
                    # 持仓变空仓
                    elif signal_list[i - 1] != 0 and signal_list[i] == 0:
                        return_series.append(target_series[i] - signal_list[i] * commission_fee)
                        return_series_by_trade.append(trade_tmp)
                        trade_tmp = 1
                    # 反转多空
                    elif abs(signal_list[i - 1]) + abs(signal_list[i]) == 2:
                        return_series.append(target_series[i] - 2 * signal_list[i] * commission_fee)
                        return_series_by_trade.append(trade_tmp)
                        trade_tmp = 1
                    else:
                        print('error')
        #                 print('len')
        return np.array(return_series), np.array(return_series_by_trade)

    # 记录绩效
    def write_file(self, signal_No, params, path, data_list):
        try:
            df = pd.read_csv(path)
            del df['Unnamed: 0']
            index = '{}_{}_{}_{}_{}'.format(signal_No, str(params['max_depth']), str(params['eta']),
                                            str(params['subsample']), str(params['min_child_weight']))
            df.loc[index] = [index] + data_list
            df.to_csv(path)
            print('输出绩效')
        except:
            try:
                os.mkdir('result')
            except:
                pass
            df = pd.DataFrame(dict.fromkeys(['name', 'signal', 'return', 'sharpe', 'maxDown', 'tradeTimes'], []))
            index = '{}_{}_{}_{}_{}'.format(signal_No, str(params['max_depth']), str(params['eta']),
                                            str(params['subsample']), str(params['min_child_weight']))
            df.loc[index] = [index] + data_list
            df.to_csv(path)
            print('输出绩效')

    # 单个资产绩效评估
    def evaluate_by_asset(self, signal_No, return_series):
        pass

    # 组合绩效评估
    def evaluate_by_portfolio(self, signal_No, return_series):
        pass

    # 硬止损
    def hard_stop(self):
        pass

    # ATR止损
    def atr_stop(self):
        pass



