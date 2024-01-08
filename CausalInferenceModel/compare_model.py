import os

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import shap
from keras.layers import *
from keras.models import *
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# import shap

def mape(true, predict, mask):
    record = np.abs((predict - true) / true)  # 取绝对值并计算相对误差
    record *= mask  # record = record * mask
    non_zero_len = mask.sum()
    return (np.sum(record) / non_zero_len) * 100


def plot_predictions(true, causal_p, close_p, all_p, filename):
    fig, axs = plt.subplots(figsize=(12, 6))

    # train_index = "Comparison between Predicted Values and Ground Truth"
    # axs.set_title(train_index, fontproperties='Times New Roman', fontsize=19, y=1.05)  # 图标题，位于图上方正中
    axs.plot(true[:], color='#4B4453')
    axs.plot(close_p[:], color='#FF9671')
    axs.plot(all_p[:], color='#008F7A')
    axs.plot(causal_p[:], color='#D83121')
    plt.xticks(fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=16)
    axs.set_xlabel('Date(Days)', fontproperties='Times New Roman', fontsize=17)
    axs.xaxis.set_label_coords(1.06, -0.02)  # 设置横坐标标签在右下方
    axs.set_ylabel('Close Price(Yuan)', fontproperties='Times New Roman', fontsize=17, rotation=0)
    axs.yaxis.set_label_coords(0, 1.0)  # 设置纵坐标标签在左上方
    axs.legend(
        ('Ground Truth', 'Close Price Only', 'Potential Factors + Close Price', 'Causal Factors + Close Price'),
        fontsize='14')
    axs.grid(True)

    save_filename = filename + '不同方法预测结果'
    graph_name = save_filename + "不同方法预测结果"
    directory = "result_graph"  # 子目录名称
    path = os.path.join(directory, graph_name)  # 将目录和图像名合并为完整的保存路径
    plt.savefig(path)
    plt.savefig(os.path.join("svg_graph", graph_name + '.svg'), format='svg')
    plt.show()


def plot_scatters(true, causal_p, close_p, all_p, filename):
    fig, axs = plt.subplots(figsize=(12, 6))

    plot_begin, plot_end = min(min(true), min(causal_p), min(close_p), min(all_p)), max(max(true), max(causal_p),
                                                                                        max(close_p), max(all_p))
    plot_x = np.linspace(plot_begin, plot_end, 10)
    axs.plot(plot_x, plot_x, color='#0089BA')
    plt.xticks(fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=16)
    axs.set_xlabel('Ground Truth(Yuan)', fontproperties='Times New Roman', fontsize=17)
    axs.xaxis.set_label_coords(1.02, -0.06)  # 设置横坐标标签在右下方
    axs.set_ylabel('Predicted Value(Yuan)', fontproperties='Times New Roman', fontsize=17, rotation=0)
    axs.yaxis.set_label_coords(0, 1.0)  # 设置纵坐标标签在左上方
    axs.plot(true, close_p, 's', color='#FF9671', alpha=0.5)
    axs.plot(true, all_p, 'D', color='#008F7A', alpha=0.5)
    axs.plot(true, causal_p, 'o', color='#D83121', alpha=0.5)
    axs.legend(
        ('Ground Truth', 'Close Price Only', 'Potential Factors + Close Price', 'Causal Factors + Close Price'),
        fontsize='14')
    axs.grid(True)

    graph_name = filename + "不同方法预测值准确性散点图"
    directory = "result_graph"  # 子目录名称
    path = os.path.join(directory, graph_name)  # 将目录和图像名合并为完整的保存路径
    plt.savefig(path)
    plt.savefig(os.path.join("svg_graph", graph_name + '.svg'), format='svg')
    plt.show()


def calculate_metric(true, predict):
    # 输出结果
    print(' RMSE:', np.sqrt(mean_squared_error(true, predict)))
    y_mask = (1 - (true == 0))
    print(' MAE:', mean_absolute_error(true, predict))
    print('MAPE:', mape(true, predict, y_mask))
    print(' R^2:', r2_score(true, predict))
    return


# 预测模型
def gru_model(input_data):
    model = Sequential()
    model.add(Input(shape=(input_data.shape[1], input_data.shape[2])))
    # model.add(Conv1D(filters=15, kernel_size=6, padding='same', strides=1, activation='relu'))
    # model.add(MaxPooling1D(pool_size=1))  # 池化层
    model.add(GRU(128, activation='tanh', return_sequences=True))
    model.add(GRU(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def one_input_predict(filename):
    # Hyperband调参算法
    def build_gru_model(hp):
        build_model = Sequential()
        build_model.add(GRU(units=hp.Int('gru_units_1', min_value=32, max_value=256, step=32), activation='tanh',
                            input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        build_model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.2, max_value=0.5, step=0.1)))
        build_model.add(GRU(units=hp.Int('gru_units_2', min_value=32, max_value=256, step=32), activation='tanh'))
        build_model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
        build_model.add(Dense(1, activation='linear'))
        build_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return build_model

    # 1、读取csv数据文件
    path = r'D:\Research\dataset\\' + filename + '.csv'
    dfx = pd.read_csv(path, index_col="date", parse_dates=True, encoding='GBK')  # Date，索引、df日期格式
    dfx = dfx.fillna(method='pad')  # 若有空值，则用上一个值填充
    df_close = pd.DataFrame(dfx['close'], columns=['close'])

    window_size = 5  # 时间窗设置， 利用前 window_size 天数据预测
    step_size = 1  # 设定窗口滚动步长
    cut_ratio = 0.2  # 划分测试集比例，例如取20%

    # 2、归一化操作
    min_max_scaler = preprocessing.MinMaxScaler()  # 采用最大最小格式归一化
    df1 = min_max_scaler.fit_transform(df_close)  # df2 为ndarray形式，无列标签
    df_close = pd.DataFrame(df1, columns=['close'])

    # 3、单独使用收盘价进行预测
    data = df_close.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

    sequence_length = window_size + step_size  # 序列长度+step_size

    # 4、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
    result = []
    for j in range(len(data) - sequence_length + 1):  # 循环 数据长度-时间窗长度+1 次
        result.append(data[j:j + sequence_length, :])  # 第i行到i+(window+1)

    result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features

    cut = int(cut_ratio * len(result))  # 按比例划分训练集测试集 最后cut个样本为测试集

    train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
    x_train = train[:, :-1]  # 所有组、不包括最后1列的所有行，训练集特征
    y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列，训练集的目标变量

    test = result[-cut:, :]  # 最后 cut 组作为测试集
    x_test = test[:, :-1]  # 最后一列以外的所有列为测试集特征

    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))  # 组 行 列
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))
    # 展示下训练集测试集的形状 看有没有问题
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)

    # 5、建立模型、训练模型过程
    # Hyperband算法
    tuner = kt.Hyperband(
        build_gru_model,
        objective='accuracy',
        max_epochs=80,
        factor=3,
        directory='tuner_directory',
        project_name='pingan_one_input_gru_tuning'
    )
    # tuner.search(X_train, y_train, epochs=80, batch_size=128)
    # # best_model = tuner.get_best_models(num_models=1)[0]
    # best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    #
    # # 使用最佳超参数构建最终模型
    # model = tuner.hypermodel.build(best_hyperparameters)
    # hist = model.fit(X_train, y_train, epochs=80, batch_size=128)

    # 手动调参的模型
    model = gru_model(X_train)  # 只有收盘价特征
    hist = model.fit(X_train, y_train, epochs=80, batch_size=128, validation_split=0, verbose=0)  # 训练模型epoch=80次

    # 展示在训练集上的表现
    plt.rc('font', family='serif')
    fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax_acc.set_title("Close Price Only")  # 图标题，位于图上方正中
    ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
    ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
    ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
    ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
    ax_loss.set_ylabel('loss', color='blue')
    plt.xlabel('epochs')
    plt.show()

    # 对测试集数据的预测
    y_test_predict = model.predict(X_test)  # 对测试集预测拟合
    y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
    y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_test_predict[0] /= min_max_scaler.scale_[-1]

    # 将预测结果保存在 close_predict 中
    close_predict = dfx.iloc[-cut:, :]
    y_test_predict = y_test_predict[0].values
    close_predict.loc[:, 'close_predict'] = y_test_predict
    return close_predict


def all_input_predict(filename):
    # Hyperband调参算法
    def build_gru_model(hp):
        build_model = Sequential()
        build_model.add(GRU(units=hp.Int('gru_units_1', min_value=32, max_value=256, step=32), activation='tanh',
                            input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        build_model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.2, max_value=0.5, step=0.1)))
        build_model.add(GRU(units=hp.Int('gru_units_2', min_value=32, max_value=256, step=32), activation='tanh'))
        build_model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
        build_model.add(Dense(1, activation='linear'))
        build_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return build_model

    # 1、读取csv数据文件
    path = r'D:\Research\dataset\\' + filename + '.csv'
    dfx = pd.read_csv(path, index_col="date", parse_dates=True, encoding='GBK')  # Date，索引、df日期格式
    dfx = dfx.fillna(method='pad')  # 若有空值，则用上一个值填充

    order = ['open', 'high', 'low', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM',
             'index_open', 'index_high', 'index_low', 'index_close', 'index_volume', 'index_amount', 'index_pctChg',
             'close']  # 12个基础数据
    df1 = dfx[order]  # 按照order次序排列

    window_size = 5  # 时间窗设置， 利用前 window_size 天数据预测
    step_size = 1  # 设定窗口滚动步长
    cut_ratio = 0.2  # 划分测试集比例，例如取20%

    # 2、归一化操作
    min_max_scaler = preprocessing.MinMaxScaler()  # 采用最大最小格式归一化
    df2 = min_max_scaler.fit_transform(df1)  # df2 为ndarray形式，无列标签
    df = pd.DataFrame(df2, columns=df1.columns)

    # 3、使用所有因子进行预测
    data = df.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

    sequence_length = window_size + step_size  # 序列长度+step_size

    # 4、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
    result = []
    for j in range(len(data) - sequence_length + 1):  # 循环 数据长度-时间窗长度+1 次
        result.append(data[j:j + sequence_length, :])  # 第i行到i+(window+1)

    result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features

    cut = int(cut_ratio * len(result))  # 按比例划分训练集测试集 最后cut个样本为测试集

    train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
    x_train = train[:, :-1]  # 所有组、不包括最后1列的所有行，训练集特征
    y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列，训练集的目标变量

    test = result[-cut:, :]  # 最后 cut 组作为测试集
    x_test = test[:, :-1]  # 最后一列以外的所有列为测试集特征

    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))  # 组 行 列
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))
    # 展示下训练集测试集的形状 看有没有问题
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)

    # 5、建立模型、训练模型过程
    # Hyperband算法
    tuner = kt.Hyperband(
        build_gru_model,
        objective='accuracy',
        max_epochs=80,
        factor=3,
        directory='tuner_directory',
        project_name='pingan_all_input_gru_tuning'
    )
    # tuner.search(X_train, y_train, epochs=80, batch_size=128)
    # # best_model = tuner.get_best_models(num_models=1)[0]
    # best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    #
    # # 使用最佳超参数构建最终模型
    # model = tuner.hypermodel.build(best_hyperparameters)
    # model.summary
    # hist = model.fit(X_train, y_train, epochs=80, batch_size=128)

    # 手动调参的模型
    model = gru_model(X_train)  # len(df.columns)特征数
    hist = model.fit(X_train, y_train, epochs=80, batch_size=128, validation_split=0, verbose=0)  # 训练模型epoch=80次

    # 展示在训练集上的表现
    plt.rc('font', family='serif')
    fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax_acc.set_title("All Potential Factors + Close Price")  # 图标题，位于图上方正中
    ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
    ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
    ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
    ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
    ax_loss.set_ylabel('loss', color='blue')
    plt.xlabel('epochs')
    plt.show()

    # 对测试集数据的预测
    y_test_predict = model.predict(X_test)  # 对测试集预测拟合
    y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
    y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_test_predict[0] /= min_max_scaler.scale_[-1]

    # 将预测结果保存在 all_predict 中
    all_predict = dfx.iloc[-cut:, :]
    y_test_predict = y_test_predict[0].values
    all_predict.loc[:, 'all_predict'] = y_test_predict
    return all_predict


if __name__ == "__main__":
    # 由键盘输入文件名
    read_filename = input("请输入用于收盘价预测的文件名：")  # 中国平安
    causal_file = read_filename + '因果预测结果'
    causal_pre = pd.read_csv(r'D:\Research\result\\' + causal_file + '.csv', index_col="date", parse_dates=True,
                             encoding='GBK')  # Date，索引、df日期格式
    close_pre = one_input_predict(read_filename)
    all_pre = all_input_predict(read_filename)
    y_true = causal_pre['close']
    y_causal_predict = causal_pre['causal_predict']
    y_close_predict = close_pre['close_predict']
    y_all_predict = all_pre['all_predict']

    plot_predictions(y_true, y_causal_predict, y_close_predict, y_all_predict, read_filename)
    plot_scatters(y_true, y_causal_predict, y_close_predict, y_all_predict, read_filename)

    print('计算因果预测的绝对评估指标：')
    calculate_metric(y_true, y_causal_predict)
    print('计算仅使用收盘价预测的绝对评估指标：')
    calculate_metric(y_true, y_close_predict)
    print('计算使用所有待选取因子预测的绝对评估指标：')
    calculate_metric(y_true, y_all_predict)

    min_max_scaler = preprocessing.MinMaxScaler()  # 采用最大最小格式归一化
    causal_pre_scaler = min_max_scaler.fit_transform(causal_pre)  # causal_pre_scaler 为ndarray形式，无列标签
    causal_pre_scaler = pd.DataFrame(causal_pre_scaler, columns=causal_pre.columns)
    true_scaler = causal_pre_scaler['close']
    causal_predict_scaler = causal_pre_scaler['causal_predict']
    close_pre_scaler = min_max_scaler.fit_transform(close_pre)  # causal_pre_scaler 为ndarray形式，无列标签
    close_pre_scaler = pd.DataFrame(close_pre_scaler, columns=close_pre.columns)
    close_predict_scaler = close_pre_scaler['close_predict']
    all_pre_scaler = min_max_scaler.fit_transform(all_pre)  # causal_pre_scaler 为ndarray形式，无列标签
    all_pre_scaler = pd.DataFrame(all_pre_scaler, columns=all_pre.columns)
    all_predict_scaler = all_pre_scaler['all_predict']
    print('计算因果预测的相对评估指标：')
    calculate_metric(true_scaler, causal_predict_scaler)
    print('计算仅使用收盘价预测的相对评估指标：')
    calculate_metric(true_scaler, close_predict_scaler)
    print('计算使用所有待选取因子预测的相对评估指标：')
    calculate_metric(true_scaler, all_predict_scaler)

    date = causal_pre.index[:]
    close_pre.insert(0, 'date', date)
    close_filename = read_filename + '用收盘价的预测结果'
    close_path = "D:\\Research\\result\\" + close_filename + ".csv"
    close_pre.to_csv(close_path, encoding="gbk", index=False)

    all_pre.insert(0, 'date', date)
    all_filename = read_filename + '用潜在因子的预测结果'
    all_path = "D:\\Research\\result\\" + all_filename + ".csv"
    all_pre.to_csv(all_path, encoding="gbk", index=False)





