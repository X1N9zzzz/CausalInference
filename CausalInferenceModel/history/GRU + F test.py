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

os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed_value = 0
np.random.seed(seed_value)
tf.random.set_seed(0)
os.environ['PYTHONHASHSEED'] = str(seed_value)


# 预测模型
def cnn_gru_model(window, amount_of_features):
    model = Sequential()
    model.add(Conv1D(filters=100, kernel_size=1, padding='same', strides=1, activation='relu',
                     input_shape=(window, amount_of_features)))
    model.add(AveragePooling1D(1))
    model.add(GRU(120, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model


# 绘图 将预测后的曲线与原曲线画在同一张坐标里 左纵轴为real（红色） 右纵轴为predict（蓝色）
def graphpred(true, predict, mpl=None, font1=None):
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(111)
    true.plot(ax=ax1, label='True value', color='tab:red', linewidth=1)
    plt.xlabel('days', fontsize=15)
    ax1.set_yticks(np.arange(9, 14, 0.5))  # 设置左边纵坐标刻度
    ax1.set_ylabel('true close', fontproperties='Times New Roman', fontsize=15, color='tab:red')  # 设置左边纵坐标标签
    plt.legend(loc=2)  # 设置图例在左上方
    plt.tick_params(axis='y', colors='tab:red')
    plt.xticks(fontproperties='Times New Roman', fontsize=12)
    plt.xlabel('days', fontproperties='Times New Roman', fontsize=15)

    ax2 = ax1.twinx()
    predict.plot(ax=ax2, label='Predictive value', color='tab:green', linewidth=1)
    ax2.set_yticks(np.arange(9, 14, 0.5))  # 设置右边纵坐标刻度
    ax2.set_ylabel('predict close', fontproperties='Times New Roman', fontsize=15, color='tab:green')  # 设置右边纵坐标标签
    plt.legend(loc=1)  # 设置图例在右上方
    plt.tick_params(axis='y', colors='tab:green')
    plt.xticks(fontproperties='Times New Roman', fontsize=12)
    plt.title("Comparison of true value and predicted value")  # 给整张图命名

    # plt.savefig("IPSO-LSTM图像/ipso_lstm_close_predict_w1.png")
    plt.show()


# F检验
def f_test(y0, y1, y, p, t):
    rss0 = mean_squared_error(y, y0) * t
    rss1 = mean_squared_error(y, y1) * t
    # print('rss0', rss0)
    # print('rss1', rss1)
    f_value = ((rss0 - rss1) / p) / (rss1 / (t - 2 * p - 1))
    return f_value


# 1、读取csv数据文件
dfx = pd.read_csv(r'D:\Research\dataset\浦发银行数据.csv', index_col="date", parse_dates=True,
                  encoding='GBK')  # Date，索引、df日期格式
df1 = dfx.iloc[:, :]  # 截取所有行, 第01列 为date
# T = len(dfx)
T = dfx.shape[0]
# print('数据行数', T)

# 2、重新对特征列排序，最后是预测项
order = ['open', 'high', 'low', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM',
         'close']  # 12个基础数据

df1 = df1[order]  # 按照order次序排列
pred_str = 'close'  # 待预测的列  又如 'pctChg'
# df1['predict'] = df1[pred_str]
# print('加入预测列后的df1：')
# print(df1)

amount_of_features = len(df1.columns)
print('amount_of_features:', amount_of_features)

window = 1  # 时间窗设置， 利用前 window 天数据预测
cut = 1  # 划分训练集测试集 最后cut个样本为测试集

# 3、归一化操作
min_max_scaler = preprocessing.MinMaxScaler()  # 采用最大最小格式归一化
df2 = min_max_scaler.fit_transform(df1)  # df2 为ndarray形式，无列标签
# print('归一化后的df0：', df2)
df = pd.DataFrame(df2, columns=df1.columns)  # 按照df0列的顺序将其变为Dataframe形式，df0是dfx的归一化数据。
df.loc['xx0'] = 0  # 由于训练的数据为 window+1，所以归一化后在df的尾部增加一个空行。
df.loc['xx1'] = 0  # 因为索引不能用数字取最后一个元素，所以归一化后在df的尾部增加一个空行。
#
# # 4、将原始数据改造为LSTM网络的输入
# stock = df
# print('包含所有特征的df：', df)
# amount_of_features = len(stock.columns)  # 计算列的数量 8+1=9个、输入特征数 0-7 共8个
# print('amount_of_features:', amount_of_features)
#
# data = stock.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵
# seq_len = window
sequence_length = window + 1  # 序列长度+1

# 4、单独使用收盘价进行预测
df_close = pd.DataFrame(df[pred_str], columns=['close'])
# print(df_close)

data = df_close.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

# 4-1、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
result = []
for j in range(len(data) - sequence_length):  # 循环 数据长度-时间窗长度 次
    result.append(data[j:j + sequence_length, :])  # 第i行到i+(window+1)

result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features
# print('result:', result)

train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
x_train = train[:, :-1]  # 所有组、不包括最后1行的所有行
y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列
# y_train = train[:, -1][:, -2:]  # 所有组、最后1行，最后2列
x_test = result[-cut:, :-1]  # 最后1组、除最后一行的数据是待测数据，即dfx中归一化后最后一行的数据
# y_test = result[-cut:, -1][:, -1]

X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))  # 组 行 列:2
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# 展示下训练集测试集的形状 看有没有问题
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)

# 4-2、建立模型、训练模型过程
model = cnn_gru_model(window, 1)  # 收盘价1个特征
hist = model.fit(X_train, y_train, epochs=80, batch_size=145, validation_split=0, verbose=0)  # 训练模型epoch=80次

# 展示在训练集上的表现
plt.rc('font', family='serif')
fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
# ax_acc.set_title(dfx['code'][0])  # 图标题，位于图上方正中
ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
ax_loss.set_ylabel('loss', color='blue')
plt.xlabel('epochs')
plt.show()

# 4-3、对训练集数据预测拟合
y_train_predict = model.predict(X_train)  # 对训练集预测拟合

y_train_predict = pd.DataFrame(y_train_predict)  # 将ndarray变为dataframe格式
y_train_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
y_train_predict[0] /= min_max_scaler.scale_[-1]

# 4-4、训练集的反归一化
y_train = pd.DataFrame(y_train)  # 将ndarray变为dataframe格式
y_train[0] -= min_max_scaler.min_[-1]  # 反归一化
y_train[0] /= min_max_scaler.scale_[-1]

# 4-5、对测试集数据的预测
y_test_predict = model.predict(X_test)  # 对测试集预测拟合

y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
y_test_predict[0] /= min_max_scaler.scale_[-1]
print('\n下一日' + pred_str + '预测为： ', y_test_predict[0])  # pred_str='close'

# 将dfx中最前面的 window 个数据去除
close_predict = pd.DataFrame(dfx.iloc[window:, amount_of_features - 1], columns=['close'])

close_predict['predict'] = y_train_predict[0].values

# 增加下一日的日期（注意，nextday可能不是交易日）
nextday = close_predict.iloc[-1].name + pd.Timedelta(days=1)
close_predict.loc[nextday, 'predict'] = y_test_predict[0].values
y0_predict = close_predict['predict'][-len(close_predict):-1].values
y_ture = close_predict['close'][-len(close_predict):-1].values  # 最后一个close为空值，删除最后一个值

graphpred(close_predict[pred_str], close_predict['predict'])

# 5、使用待检验因子和收盘价对收盘价进行预测
f = []
col = []
causal_col = []
for i in range(amount_of_features - 1):
    # print('stock.iloc[:, i]:', stock.iloc[:, i])
    # print('stock.iloc[:, amount_of_features - 1]:', stock.iloc[:, amount_of_features - 1])
    # dfi = np.stack((df.iloc[:, i], df.iloc[:, amount_of_features - 1]), 1)
    dfi = df.iloc[:, [i, amount_of_features - 1]]
    # print('仅包含待检验因子的dfi：', dfi)
    print('i对应的列名：', df.columns[i])
    print('i=', i)
    # dfi = pd.DataFrame(dfi, columns=['test', 'close'])
    # print(dfi)

    data = dfi.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

    # 5-1、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
    result = []
    for j in range(len(data) - sequence_length):  # 循环 数据长度-时间窗长度 次
        result.append(data[j:j + sequence_length, :])  # 第i行到i+(window+1)

    result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features
    # print('result:', result)

    train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
    x_train = train[:, :-1]  # 所有组、不包括最后1行的所有行
    y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列
    # y_train = train[:, -1][:, -2:]  # 所有组、最后1行，最后2列
    x_test = result[-cut:, :-1]  # 最后1组、除最后一行的数据是待测数据，即dfx中归一化后最后一行的数据
    # y_test = result[-cut:, -1][:, -1]

    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 2))  # 组 行 列:2
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 2))
    # 展示下训练集测试集的形状 看有没有问题
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)

    # 5-2、建立LSTM、训练模型过程
    model = cnn_gru_model(window, 2)  # 检验因子+收盘价共两个特征
    hist = model.fit(X_train, y_train, epochs=80, batch_size=145, validation_split=0, verbose=0)  # 训练模型epoch=80次

    # 展示在训练集上的表现
    plt.rc('font', family='serif')
    fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # ax_acc.set_title(dfx['code'][0])  # 图标题，位于图上方正中
    ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
    ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
    ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
    ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
    ax_loss.set_ylabel('loss', color='blue')
    plt.xlabel('epochs')
    plt.show()

    # 5-3、对训练集数据预测拟合
    y_train_predict = model.predict(X_train)  # 对训练集预测拟合

    y_train_predict = pd.DataFrame(y_train_predict)  # 将ndarray变为dataframe格式
    y_train_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_train_predict[0] /= min_max_scaler.scale_[-1]

    # 5-4、训练集的反归一化
    y_train = pd.DataFrame(y_train)  # 将ndarray变为dataframe格式
    y_train[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_train[0] /= min_max_scaler.scale_[-1]

    # 5-5、对测试集数据的预测
    y_test_predict = model.predict(X_test)  # 对测试集预测拟合

    y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
    y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_test_predict[0] /= min_max_scaler.scale_[-1]
    print('\n下一日' + pred_str + '预测为： ', y_test_predict[0])  # pred_str='close'

    # 将dfx中最前面的 window 个数据去除
    test_predict = dfx.iloc[window:, [i, amount_of_features - 1]]
    # dfi_test = dfi_test.drop(dfi_test.tail(2).index, inplace=True)
    # print('y_train_predict[0].values', y_train_predict[0].values)
    test_predict['predict'] = y_train_predict[0].values

    # 增加下一日的日期（注意，nextday可能不是交易日）
    nextday = test_predict.iloc[-1].name + pd.Timedelta(days=1)
    test_predict.loc[nextday, 'predict'] = y_test_predict[0].values

    graphpred(test_predict['close'], test_predict['predict'])

    y1_predict = test_predict['predict'][-len(test_predict):-1].values
    # print('y0', y0_predict)
    # print('y1', y1_predict)
    # print('y', y_ture)

    f_value = f_test(y0_predict, y1_predict, y_ture, window, T)  # 计算F检验的值
    # print('f_value：', f_value)
    f.append(f_value)  # 将各F值加入数组
    col.append(i)  # 将各因子序号加入数组

    # F检验
    if f_value > 3.501:  # 与F表比较
        print('判断中的i值：', i)
        causal_col.append(i)  # 将通过F检验的因子序号加入因果因子序号数组
        print(causal_col)
f = np.array(f).reshape(1, amount_of_features - 1)  # 将F值形状从(11,1)转换成(1,11)
f = pd.DataFrame(f, columns=df1.columns[col])  # 将F值转换为DataFrame格式
print('f检验的值', f)

causal_factor = df.iloc[:, causal_col]  # 读取因果因子的数据
cf_name = causal_factor.columns  # 读取因果因子的名称
print('因果因子的名称', cf_name)

# 6、使用因果因子预测
causal_col.append(amount_of_features - 1)
df_causal = df.iloc[:, causal_col]
data = df_causal.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

# 6-1、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
result = []
for j in range(len(data) - sequence_length):  # 循环 数据长度-时间窗长度 次
    result.append(data[j:j + sequence_length, :])  # 第i行到i+(window+1)

result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features
# print('result:', result)

train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
x_train = train[:, :-1]  # 所有组、不包括最后1行的所有行
y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列
# y_train = train[:, -1][:, -2:]  # 所有组、最后1行，最后2列
x_test = result[-cut:, :-1]  # 最后1组、除最后一行的数据是待测数据，即dfx中归一化后最后一行的数据
# y_test = result[-cut:, -1][:, -1]

X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(df_causal.columns)))  # 组 行 列
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(df_causal.columns)))
# 展示下训练集测试集的形状 看有没有问题
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)

# 6-2、建立模型、训练模型过程
model = cnn_gru_model(window, len(df_causal.columns))  # len(causal_factor.columns)因果因子数特征
hist = model.fit(X_train, y_train, epochs=80, batch_size=145, validation_split=0, verbose=0)  # 训练模型epoch=80次

# 展示在训练集上的表现
plt.rc('font', family='serif')
fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
# ax_acc.set_title(dfx['code'][0])  # 图标题，位于图上方正中
ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
ax_loss.set_ylabel('loss', color='blue')
plt.xlabel('epochs')
plt.show()

# 6-3、对训练集数据预测拟合
y_train_predict = model.predict(X_train)  # 对训练集预测拟合

y_train_predict = pd.DataFrame(y_train_predict)  # 将ndarray变为dataframe格式
y_train_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
y_train_predict[0] /= min_max_scaler.scale_[-1]

# 6-4、训练集的反归一化
y_train = pd.DataFrame(y_train)  # 将ndarray变为dataframe格式
y_train[0] -= min_max_scaler.min_[-1]  # 反归一化
y_train[0] /= min_max_scaler.scale_[-1]

# 6-5、对测试集数据的预测
y_test_predict = model.predict(X_test)  # 对测试集预测拟合

y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
y_test_predict[0] /= min_max_scaler.scale_[-1]
print('\n下一日' + pred_str + '预测为： ', y_test_predict[0])  # pred_str='close'

# 将dfx中最前面的 window 个数据去除
causal_predict = dfx.iloc[window:, causal_col]
# dfi_test = dfi_test.drop(dfi_test.tail(2).index, inplace=True)
# print('y_train_predict[0].values', y_train_predict[0].values)
causal_predict['predict'] = y_train_predict[0].values

# 增加下一日的日期（注意，nextday可能不是交易日）
nextday = causal_predict.iloc[-1].name + pd.Timedelta(days=1)
causal_predict.loc[nextday, 'predict'] = y_test_predict[0].values

graphpred(close_predict['close'], causal_predict['predict'])
