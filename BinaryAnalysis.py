
from bin.Evaluation import Evaluation
from bin.Modeling import Modeling
import empyrical

# %matplotlib inline
# 普通二分测试
class BinaryAnalysis(object):

    def __init__(self,
                 test_start='2017-01-01',
                 test_end='2019-12-27',
                 backtest_type='open_to_open',
                 params={
                     'booster': 'gbtree',
                     'objective': 'reg:linear',
                     'gamma': 0,
                     'max_depth': 6,
                     'subsample': 0.5,
                     'colsample_bytree': 0.5,
                     'min_child_weight': 5,
                     'silent': 1,
                     'eta': 0.05,
                     'seed': 1000,
                     'nthread': -1,
                 }
                 ):
        self.instruments = ['000001', '000016', '000300', '000905', '399001', '399006', '399101']
        self.Eva = Evaluation()
        self.backtest_type = backtest_type
        self.Model = Modeling(self.instruments, train_end_date=test_start)
        self.train_df = self.Model.data.train_df.copy()
        self.target_df = self.Model.data.target_df[self.backtest_type]
        self.target_dict = self.Model.data.target_dict.copy()
        self.test_dict = self.Model.data.test_dict.copy()
        self.test_start = test_start
        self.test_end = test_end
        self.params = params

        # 批量训练

    def start_training(self, param_list=None):
        # 输入参数为列表，则进行遍历训练
        if isinstance(param_list, list):
            for params in param_list:
                train_df = self.train_df.copy()
                target_df = self.target_df.copy()
                target_dict = self.target_dict.copy()
                test_dict = self.test_dict.copy()
                # 训练子模型
                Models = self.Model.start_train_once(train_df, target_df, params)
                self.Models = Models

                # 子模型预测
                for index in self.target_dict.keys():
                    target_dict[index]['yvalid'] = self.Eva.predict(test_dict, self.target_dict, Models)

                    # 生成技术信号
                for index in self.target_dict.keys():
                    self.target_dict[index] = self.Eva.gen_trading_signal(index, target_dict)

                # 信号列表
                self.signals = [signal for signal in list(self.target_dict[index].columns) if signal[:6] == 'signal']

                # 绩效生成
                for index in self.target_dict.keys():
                    for signal in self.signals:
                        return_series, return_series_by_trade = self.Eva.get_return_series(target_dict[index][signal],
                                                                                           target_dict[index][
                                                                                               self.backtest_type])
                        annual_return = empyrical.annual_return(return_series)
                        sharpe_ratio = empyrical.sharpe_ratio(return_series)
                        max_drawdown = empyrical.max_drawdown(return_series)
                        trade_times = len(return_series_by_trade)
                        data_list = [signal, annual_return, sharpe_ratio, max_drawdown, trade_times]
                        path = r'result/Binary_result_0225.csv'
                        # 导出绩效
                        self.Eva.write_file(signal, params, path, data_list)
                        if signal == 'signal_1':
                            self.return_series = return_series
                            self.return_series_by_trade = return_series_by_trade
                            self.target_test = target_dict[index]

        # 输入参数为字典，则进行单次训练
        if isinstance(param_list, dict):
            pass

if __name__ == '__main__':
    BinaryAnalysis = BinaryAnalysis(test_start='2016-01-01')
    param_list = BinaryAnalysis.Model.get_params()
    BinaryAnalysis.start_training(param_list=param_list)
print('pass')