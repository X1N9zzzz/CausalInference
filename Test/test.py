from keras import Model
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import layers
from keras.activations import gelu
from keras.utils import plot_model
from sklearn.preprocessing import StandardScaler
import os
import random


def stock_aug_net_c_data_process():
    # 数据处理
    df1 = pd.read_csv(r'D:\Research\000596古井贡酒6.SZ(1).csv')

    df1['trade_date'] = pd.to_datetime(df1['trade_date'])  # 转为日期格式

    df1.sort_values('trade_date', inplace=True)  # 日期排序

    df1 = df1.loc[:, ~df1.columns.str.contains('^Unnamed')]

    # 获取股票所属行业
    raw_path = 'D:/Research/白酒1/'

    # 所属行业市值前十的股票数据
    csv_list = [os.path.join(raw_path, i) for i in os.listdir(raw_path)]

    raw_df = pd.DataFrame({
        'trade_date': df1['trade_date']
    })

    csv_data_frame = []
    for k, v in enumerate(csv_list):
        temp = pd.read_csv(v)
        temp['trade_date'] = pd.to_datetime(temp['trade_date'])  # 转为日期格式
        temp.sort_values('trade_date', inplace=True)  # 日期排序
        temp = temp.loc[:, ['open', 'high', 'low', 'close', 'vol']]
        temp = pd.concat([raw_df, temp], axis=1, join='outer')  # 使用pd.concat()进行连接
        # 均值填补
        temp.fillna(temp.mean(), inplace=True)

        csv_data_frame.append(temp)

    print(csv_data_frame)

    res = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'vol'])
    print(df1)
    for k, v in enumerate(df1.index):
        time = df1.loc[v, 'trade_date']
        rc = random.choices(csv_data_frame, k=5)
        temp = pd.concat([i[i['trade_date'] == time] for i in rc]).sum()
        res.loc[k] = temp

    df2 = res
    print("df2", df2)
    x_col = df1.select_dtypes('float').columns.drop(['high', 'low', 'close'])  # 去掉非浮点数类型的列和待预测列

    std_y = StandardScaler()
    y = std_y.fit_transform(df1[['high', 'low', 'close']])
    y = pd.DataFrame(y, columns=['high', 'low', 'close'])

    std_1 = StandardScaler()
    df1_x = std_1.fit_transform(df1[x_col])
    df1_x = pd.DataFrame(df1_x, columns=x_col)
    df1_x = pd.concat([df1_x, y], axis=1)  # 将x_col列和待预测列拼接在一起
    # 将选择后的df1_x赋值给df1_x，实现了对df1_x的筛选，只保留了与df2中列名一致的列
    df1_x = df1_x[df2.columns[1:]]

    # 对df2中从第一列之后的所有数值列进行标准化处理
    std_2 = StandardScaler()
    df2_x = std_2.fit_transform(df2.iloc[:, 1:])
    df2_x = pd.DataFrame(df2_x, columns=df2.columns[1:])

    return df1, df1_x, df2_x, std_y


def create_x(s, ws):
    series_s = s.copy()
    series = s.copy()
    for i in range(ws):
        series = pd.concat([series, series_s.shift(-(i + 1))], axis=1)
    series.dropna(axis=0, inplace=True)
    return series


def data_set_split(sep, series_x, window_size):
    train = series_x.iloc[:sep - window_size + 1, :].values
    test = series_x.iloc[sep - 1 - window_size + 1:, :].values

    # 使用np.expand_dims()函数为训练集和测试集的数组增加一个维度
    train = np.expand_dims(train, 2)
    test = np.expand_dims(test, 2)
    return train, test


def split_data(df1_x, df2_x):
    data_set1 = []
    data_set2 = []
    data_label = []

    window_size = 5

    for i in df1_x.columns:
        temp = create_x(df1_x[[i]], window_size)
        if i in ['high', 'low', 'close']:
            data_label.append(temp.iloc[:, -1])
        data_set1.append(temp.iloc[:, :-1])

    for i in df2_x.columns:
        temp = create_x(df2_x[[i]], window_size)
        data_set2.append(temp.iloc[:, :-1])

    # 其值为data_set1[0]的行数乘以0.8再加1，用于之后划分训练集和测试集
    sep = int(data_set1[0].shape[0] * 0.8) + 1
    x_train, x_test, x_train1, x_test1 = [], [], [], []
    # 划分训练集和测试集
    for k, v in enumerate(data_set1):
        if k == 0:
            x_train, x_test = data_set_split(sep, v, window_size=10)
        else:
            x_train_temp, x_test_temp = data_set_split(sep, v, window_size=10)

            x_train = np.concatenate((x_train, x_train_temp), axis=-1)
            x_test = np.concatenate((x_test, x_test_temp), axis=-1)

    # 划分训练集和测试集
    for k, v in enumerate(data_set2):
        if k == 0:
            x_train1, x_test1 = data_set_split(sep, v, window_size=10)
        else:
            x_train_temp, x_test_temp = data_set_split(sep, v, window_size=10)

            x_train1 = np.concatenate((x_train1, x_train_temp), axis=-1)
            x_test1 = np.concatenate((x_test1, x_test_temp), axis=-1)

    data_label = pd.concat(data_label, axis=1)

    y_train, y_test = data_set_split(sep, pd.DataFrame(data_label), window_size=10)

    y_train = y_train.reshape(-1, 3)
    y_test = y_test.reshape(-1, 3)

    return x_train, x_train1, x_test, x_test1, y_train, y_test


def train_model(x_train, x_train1, x_test, x_test1, y_train, y_test):
    # 超参数
    n_hidden = 128

    output_dim = 10

    # 构造输入输出
    inp1 = layers.Input((x_train.shape[1], x_train.shape[2]))
    inp2 = layers.Input((x_train1.shape[1], x_train1.shape[2]))

    out1 = layers.GRU(n_hidden)(inp1)
    out2 = layers.GRU(n_hidden)(inp2)

    out1 = layers.Dense(output_dim, activation=gelu)(out1)
    out2 = layers.Dense(output_dim, activation=gelu)(out2)
    out = layers.concatenate([out1, out2], axis=1)
    out = layers.Dense(3)(out)

    model = Model([inp1, inp2], out)
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            'mae', 'mape', tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )

    model.summary()

    model.fit([x_train, x_train1], y_train, validation_data=([x_test, x_test1], y_test), epochs=20, batch_size=64)

    return model


def stock_aug_net_c_predict():
    df1, df1_x, df2_x, std_y = stock_aug_net_c_data_process()

    x_train, x_train1, x_test, x_test1, y_train, y_test = split_data(df1_x, df2_x)

    model = train_model(x_train, x_train1, x_test, x_test1, y_train, y_test)

    plot_model(model, to_file='model.png', show_shapes=True)

    pre = model.predict([x_test, x_test1])

    print(pre)


if __name__ == "__main__":
    stock_aug_net_c_predict()
