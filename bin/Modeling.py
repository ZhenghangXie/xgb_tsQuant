from data.DataPrepare import DataPrepare
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
from sklearn.model_selection import ShuffleSplit
import xgboost as xgb
import sklearn
import numpy as np

class Modeling(object):


    def __init__(self, instruments,
                 train_end_date='2017-01-01',
                 commission_fee=0.001,
                 backtest_type='open_to_open'
                 ):
        self.param_list = self.get_params()
        self.data = DataPrepare(instruments, train_end_date=train_end_date)
        self.X_Matrix = self.data.train_df.loc[:train_end_date]
        self.y = self.data.target_df[backtest_type]

    def get_params(self):
        import numpy as np
        params_list = []
        for max_depth in range(4, 10, 2):
            for eta in np.arange(0.01, 0.015, 0.005):
                for colsample_bytree_and_subsample in np.arange(0.5, 0.7, 0.2):
                    for min_child_weight in np.arange(4, 6, 2):
                        params = {
                            'booster': 'gbtree',
                            'objective': 'reg:linear',
                            'gamma': 0,
                            'max_depth': max_depth,
                            'subsample': colsample_bytree_and_subsample,
                            'colsample_bytree': colsample_bytree_and_subsample,
                            'min_child_weight': min_child_weight,
                            'silent': 1,
                            'eta': eta,
                            'seed': 1000,
                            'nthread': -1,
                        }
                        params_list.append(params)
        return params_list

    # 进行单组参数训练
    def start_train_once(self,
                         X_Matrix,
                         y,
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
                         }):

        # 参数设置
        n_split = 10
        rs = ShuffleSplit(n_splits=n_split, test_size=0.3, random_state=0)
        model = []
        # 三个变量记录训练收敛情况
        train_rmse = []
        var_rmse = []
        dt = {}
        X, y = X_Matrix, y
        for train_index, test_index in rs.split(X):
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)
            watchlist = [(dtrain, 'Train'), (dtest, 'Val')]
            bst = xgb.train(params, dtrain, num_boost_round=1000, evals=watchlist, evals_result=dt,
                            early_stopping_rounds=200)
            train_rmse.append(dt['Train']['rmse'][-1])
            var_rmse.append(dt['Val']['rmse'][-1])
            model.append(bst)

        return model

    # 进行批量参数训练
    def start_train(self):
        pass

    def start_wfa_train(self):
        pass



