import os
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.models import *
from sklearn import preprocessing
from scipy.stats import ttest_rel, f
from sklearn.metrics import mean_squared_error
from tcn import TCN


# F检验
def f_test(y0, y1, y, p, t):
    rss0 = mean_squared_error(y, y0) * t
    rss1 = mean_squared_error(y, y1) * t
    # print('rss0', rss0)
    # print('rss1', rss1)
    f = ((rss0 - rss1) / p) / (rss1 / (t - 2 * p - 1))
    return f


# 读取十次实验结果平均后的数值
all_result = pd.read_csv(r'D:\Research\result\all_result1.csv', index_col="date", parse_dates=True,
                         encoding='GBK')

# 共有几个需要检验的因子，包括收盘价以及收盘价预测得到的值
amount_of_factors = len(all_result.columns)
print('======', all_result)
print(amount_of_factors)

# close真实值，最后一行为空，去掉最后一行
y_ture = all_result['close'][-len(all_result):-1].values

# 仅使用close历史信息预测得到的值,并去掉最后一行，因为真实值最后一行为空，其他列的最后一行也要去掉匹配真实值数据的大小
y0_predict = all_result['close_predict'][-len(all_result):-1].values

# 时间窗即滞后长度
window = 1

# 样本容量
T = all_result.shape[0]

f_value = []
col = []
causal_col = []
cf_name = []
for i in range(amount_of_factors - 2):
    factor_name = all_result.columns[i]
    print('factor_name:', factor_name)

    # 使用不同因子和close进行预测得到的值
    y1_predict = all_result[factor_name][-len(all_result):-1].values

    # 计算F检验的值
    f_test_calculate = f_test(y0_predict, y1_predict, y_ture, window, T)
    print('f_test_calculate：', f_test_calculate)

    # 将各F值加入数组
    f_value.append(f_test_calculate)

    # 将各因子序号加入数组
    col.append(i)

    # F检验
    if f_test_calculate > 3.501:  # 与F表比较
        print('判断通过的i值：', i)
        causal_col.append(i)  # 将通过F检验的因子序号加入因果因子序号数组
        print(causal_col)

f_value = np.array(f_value).reshape(1, amount_of_factors - 2)  # 将F值形状从(11,1)转换成(1,11)
f_value = pd.DataFrame(f_value, columns=['open', 'high', 'low', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ',
                                         'psTTM', 'pcfNcfTTM'])  # 将F值转换为DataFrame格式
print('f检验的值', f_value)

cf_name = f_value.columns[causal_col]  # 读取因果因子的名称
print('因果因子的名称', cf_name)
