from bin.Evaluation import Evaluation
from bin.Modeling import Modeling
import empyrical
import datetime,time
from BinaryAnalysis import BinaryAnalysis

class WalkForwardAnalysis(object):

    def __init__(self,
                 test_start='2017-01-01',
                 test_days=90,
                 train_days=1800,
                 test_end='2019-12-28',
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
        self.test_start = test_start
        self.test_days = test_days
        self.train_days = train_days
        self.test_end = test_end
        self.backtest_type = backtest_type
        self.params = params
        self.Eva = Evaluation()
        self.Model = Modeling(self.instruments, train_end_date=test_start)
        self.return_series = []

    def start_WFA(self, param_list=None, test_start='2017-01-01'):
        '''
        输入训练开始时间
        '''
        self.train_df = self.Model.data.train_df.copy()
        self.target_df = self.Model.data.target_df[self.backtest_type]
        self.target_dict = self.Model.data.target_dict.copy()
        self.test_dict = self.Model.data.test_dict.copy()
        if isinstance(param_list, list):
            for params in param_list:
                start = time.clock()
                train_df = self.train_df.copy()
                target_df = self.target_df.copy()
                target_dict = self.target_dict.copy()
                test_dict = self.test_dict.copy()
                self.Model = Modeling(self.instruments, train_end_date=test_start)
                self.target_dict = self.Model.data.target_dict.copy()
                print('数据初始化成功,耗时：%s' % (time.clock() - start))

                # 设置首次测试的测试日期
                sub_test_start = test_start
                sub_test_end = datetime.datetime.strptime(sub_test_start, '%Y-%m-%d') + datetime.timedelta(
                    days=self.test_days)
                sub_test_end = datetime.datetime.strftime(sub_test_end, '%Y-%m-%d')

                # 暂时储存WFA阶段得到的yvalid
                self.yvalid_dict = {}
                for index in instruments:
                    self.yvalid_dict[index] = []

                # while循环进行,如果sub_test_start早于test截止日期，则课运行
                while sub_test_start < self.test_end:

                    # 设置WFA子模型日期
                    sub_train_start = datetime.datetime.strptime(sub_test_start, '%Y-%m-%d') - datetime.timedelta(
                        days=self.train_days)
                    sub_train_start = datetime.datetime.strftime(sub_train_start, '%Y-%m-%d')
                    sub_train_end = datetime.datetime.strptime(sub_test_start, '%Y-%m-%d') - datetime.timedelta(days=1)
                    sub_train_end = datetime.datetime.strftime(sub_train_end, '%Y-%m-%d')
                    sub_train_df = self.Model.data.train_df.loc[sub_train_start:sub_train_end]
                    sub_target_df = self.Model.data.target_df.loc[sub_train_start:sub_train_end][self.backtest_type]

                    # 训练子模型
                    Models = self.Model.start_train_once(sub_train_df, sub_target_df, params)

                    # 模型预测
                    for index in instruments:
                        print('本次测试日期：{}到{}，共{}天'.format(sub_test_start, sub_test_end, len(
                            self.Model.data.test_dict[index][sub_test_start:sub_test_end])))
                        self.yvalid_dict[index].append(
                            self.Eva.predict_by_date(index, self.Model.data.test_dict, sub_test_start, sub_test_end,
                                                     Models))

                        # 日期更新
                    sub_test_start = datetime.datetime.strptime(sub_test_end, '%Y-%m-%d') + datetime.timedelta(days=1)
                    sub_test_start = datetime.datetime.strftime(sub_test_start, '%Y-%m-%d')
                    sub_test_end = datetime.datetime.strptime(sub_test_end, '%Y-%m-%d') + datetime.timedelta(
                        days=self.test_days + 1)
                    sub_test_end = datetime.datetime.strftime(sub_test_end, '%Y-%m-%d')

                # 填充训练好的yvalid
                for index in self.target_dict.keys():
                    # ValueError
                    self.target_dict[index]['yvalid'] = sum(self.yvalid_dict[index], [])

                    # 生成技术信号
                for index in self.target_dict.keys():
                    target_dict[index] = self.Eva.gen_trading_signal(index, self.target_dict)
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
                        path = r'result/WFA_result_0228.csv'
                        # 导出绩效
                        self.Eva.write_file(signal, params, path, data_list)
                        print('运行成功！')

if __name__ == '__main__':
    BinaryAnalysis = BinaryAnalysis()
    param_list = BinaryAnalysis.Model.get_params()
    print(len(param_list))
    wfa = WalkForwardAnalysis(train_days = 3600)
    wfa.start_WFA(param_list = param_list,test_start='2016-01-01')










