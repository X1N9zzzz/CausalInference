import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras_tuner as kt
from keras.layers import *
from keras.models import *
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


# 计算MAPE
def mape(true, predict, mask):
    record = np.abs(predict - true) / (np.abs(true) + 1)  # 取绝对值
    record *= mask  # record = record * mask
    non_zero_len = mask.sum()
    return np.sum(record) / non_zero_len


def calculate_metric(true, predict):
    # 输出结果
    print(' RMSE:', np.sqrt(mean_squared_error(true, predict)))
    y_mask = (1 - (true == 0))
    print(' MAE:', mean_absolute_error(true, predict))
    print('MAPE:', mape(true, predict, y_mask))
    print(' R^2:', r2_score(true, predict))
    return


def predict_close(filename):
    def build_gru_model(hp):
        build_model = Sequential()
        build_model.add(GRU(units=hp.Int('gru_units_1', min_value=32, max_value=256, step=32), activation='tanh',
                            input_shape=(window_size, shape), return_sequences=True))
        build_model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.2, max_value=0.5, step=0.1)))
        build_model.add(GRU(units=hp.Int('gru_units_2', min_value=32, max_value=256, step=32), activation='tanh'))
        build_model.add(Dropout(hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)))
        build_model.add(Dense(1, activation='linear'))
        build_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        return build_model

    # 1、读取csv数据文件
    # 拼接文件路径
    read_path = r'D:\Research\dataset\\' + filename + '.csv'
    dfx = pd.read_csv(read_path, index_col="date", parse_dates=True, encoding='GBK')  # Date，索引、df日期格式
    # df1 = dfx.iloc[:, :]  # 截取所有行, 第01列 为date
    # T = dfx.shape[0]
    # 检查整个 DataFrame 是否有缺失值
    missing_values = dfx.isnull().sum().sum()
    print("缺失值的数量：", missing_values)

    # 检查每列是否有缺失值
    missing_values_per_column = dfx.isnull().sum()
    print("每列的缺失值数量：")
    print(missing_values_per_column)

    dfx = dfx.fillna(method='pad')  # 若有空值，则用上一个值填充

    # 2、重新对特征列排序，最后是预测项
    order = ['open', 'high', 'low', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM',
             'index_open', 'index_high', 'index_low', 'index_close', 'index_volume', 'index_amount', 'index_pctChg',
             'close']  # 19个基础数据

    df1 = dfx[order]  # 按照order次序排列

    amount_of_features = len(df1.columns)  # 数据集中的特征数，包括close列
    print('amount_of_features:', amount_of_features)

    window_size = 5  # 时间窗设置， 利用前 window_size 天数据预测
    step_size = 1  # 设定窗口滚动步长
    cut_ratio = 0.2  # 划分测试集比例，例如取20%

    # 3、归一化操作
    min_max_scaler = preprocessing.MinMaxScaler()  # 采用最大最小格式归一化
    df2 = min_max_scaler.fit_transform(df1)  # df2 为ndarray形式，无列标签
    df = pd.DataFrame(df2, columns=df1.columns)  # 按照df0列的顺序将其变为Dataframe形式，df0是dfx的归一化数据。

    sequence_length = window_size + step_size  # 序列长度+step_size

    # 4、使用实验组数据进行预测
    data = df.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

    # 4-1、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
    result = []
    for j in range(len(data) - sequence_length + 1):  # 循环 数据长度-时间窗长度+1 次
        result.append(data[j:j + sequence_length, :])  # 第j行到j+window_size

    result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features

    cut = int(cut_ratio * len(result))  # 按比例划分训练集测试集 最后cut个样本为测试集

    train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
    x_train = train[:, :-1]  # 所有组、不包括最后1列的所有行，训练集特征
    y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列，训练集的目标变量

    test = result[-cut:, :]  # 最后 cut 组作为测试集
    x_test = test[:, :-1]  # 最后一列以外的所有列为测试集特征

    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(df.columns)))  # 组 行 列
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(df.columns)))
    # 展示下训练集测试集的形状 看有没有问题
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)

    # 4-2、建立模型、训练模型过程
    shape = 19
    # Hyperband算法
    tuner = kt.Hyperband(
        build_gru_model,
        objective='accuracy',
        max_epochs=80,
        factor=3,
        directory='tuner_directory',
        project_name='pingan_gru_tuning'
    )
    tuner.search(X_train, y_train, epochs=80, batch_size=128)
    # # best_model = tuner.get_best_models(num_models=1)[0]
    # best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 使用最佳超参数构建最终模型
    # model = tuner.hypermodel.build(best_hyperparameters)
    # hist = model.fit(X_train, y_train, epochs=80, batch_size=128)

    # 手动调参的模型
    model = gru_model(X_train)
    hist = model.fit(X_train, y_train, epochs=80, batch_size=145, validation_split=0, verbose=0)  # 训练模型epoch=80次

    # 展示在训练集上的表现
    plt.rc('font', family='serif')
    fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # ax_acc.set_title("实验组训练过程")  # 图标题，位于图上方正中
    ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
    ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
    ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
    ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
    ax_loss.set_ylabel('loss', color='blue')
    plt.xlabel('epochs')
    plt.show()

    # 4-3、对测试集数据的预测
    y_test_predict = model.predict(X_test)  # 对测试集预测拟合

    y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
    y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_test_predict[0] /= min_max_scaler.scale_[-1]

    # 4-4、获取 dfx 最后的 cut 个数据, 只保留 close 列
    all_predict = pd.DataFrame(dfx.iloc[-cut:, amount_of_features - 1], columns=['close'])

    # 4-5、将预测结果保存在 close_predict 的 predict 列中
    all_predict['predict'] = y_test_predict[0].values

    print("实验组预测的精度：")
    calculate_metric(all_predict['close'], all_predict['predict'])

    # 获取 dfx 的最后 cut 个索引
    last_cut_indices = dfx.index[-cut:]

    # 定义一个空的dataframe数据，记录所有的预测结果
    all_result = pd.DataFrame()

    # 5、使用对照组数据对收盘价进行预测
    for i in range(11):
        # dfi = df.iloc[:, [i, amount_of_features - 1]]
        dfi = df.drop(df.columns[i], axis=1)
        print('i对应的列名：', df.columns[i])
        print('i=', i)

        data = dfi.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

        # 5-1、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
        result = []
        for j in range(len(data) - sequence_length + 1):  # 循环 数据长度-时间窗长度+1 次
            result.append(data[j:j + sequence_length, :])  # 第i行到i+(window+1)

        result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features

        train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
        x_train = train[:, :-1]  # 所有组、不包括最后1行的所有行
        y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列
        x_test = result[-cut:, :-1]  # 最后1组、除最后一行的数据是待测数据，即dfx中归一化后最后一行的数据

        X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(dfi.columns)))  # 组 行 列
        X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(dfi.columns)))
        # 展示下训练集测试集的形状 看有没有问题
        print("X_train", X_train.shape)
        print("y_train", y_train.shape)
        print("X_test", X_test.shape)

        # 5-2、建立LSTM、训练模型过程
        # Hyperband算法
        shape = 18
        tuner = kt.Hyperband(
            build_gru_model,
            objective='accuracy',
            max_epochs=80,
            factor=3,
            directory='tuner_directory',
            project_name='pingan_gru_tuning' + str(i)
        )
        tuner.search(X_train, y_train, epochs=80, batch_size=128)
        # # best_model = tuner.get_best_models(num_models=1)[0]
        # best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

        # 使用最佳超参数构建最终模型
        # model = tuner.hypermodel.build(best_hyperparameters)
        # hist = model.fit(X_train, y_train, epochs=80, batch_size=128)

        # 手动调参的模型
        model = gru_model(X_train)
        hist = model.fit(X_train, y_train, epochs=80, batch_size=145, validation_split=0, verbose=0)  # 训练模型epoch=80次

        # 展示在训练集上的表现
        plt.rc('font', family='serif')
        fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
        # ax_acc.set_title(df.columns[i] + "对照组训练过程")  # 图标题，位于图上方正中
        ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
        ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
        ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
        ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
        ax_loss.set_ylabel('loss', color='blue')
        plt.xlabel('epochs')
        plt.show()

        # 5-3、对测试集数据的预测
        y_test_predict = model.predict(X_test)  # 对测试集预测拟合

        y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
        y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
        y_test_predict[0] /= min_max_scaler.scale_[-1]

        # 5-4、获取 dfx 最后的 cut 个数据
        test_predict = pd.DataFrame(dfx.iloc[-cut:, amount_of_features - 1], columns=['close'])
        test_predict['predict'] = y_test_predict[0].values

        print(df.columns[i] + "对照组预测的精度：")
        calculate_metric(test_predict['close'], test_predict['predict'])

        # 5-5、记录使用待检验因子和收盘价预测的结果
        col_name = df.columns[i] + '_predict'
        all_result[col_name] = test_predict['predict']

    # 6、上证指数对照组
    dfi = df.drop(df.columns[11:18], axis=1)
    print('对照组去掉的列名：index')

    data = dfi.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

    # 6-1、按照Windows的数值将元数据变换成 组、行x列 的3维矩阵
    result = []
    for j in range(len(data) - sequence_length + 1):  # 循环 数据长度-时间窗长度+1 次
        result.append(data[j:j + sequence_length, :])  # 第i行到i+(window+1)

    result = np.array(result)  # 得到样本，样本形式为 window * amount_of_features

    train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
    x_train = train[:, :-1]  # 所有组、不包括最后1行的所有行
    y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列
    x_test = result[-cut:, :-1]  # 最后1组、除最后一行的数据是待测数据，即dfx中归一化后最后一行的数据

    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], len(dfi.columns)))  # 组 行 列
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], len(dfi.columns)))
    # 展示下训练集测试集的形状 看有没有问题
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)

    # 6-2、建立LSTM、训练模型过程
    shape = 12
    # Hyperband算法
    tuner = kt.Hyperband(
        build_gru_model,
        objective='accuracy',
        max_epochs=80,
        factor=3,
        directory='tuner_directory',
        project_name='pingan_index_gru_tuning'
    )
    tuner.search(X_train, y_train, epochs=80, batch_size=128)
    # # best_model = tuner.get_best_models(num_models=1)[0]
    # best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 使用最佳超参数构建最终模型
    # model = tuner.hypermodel.build(best_hyperparameters)
    # hist = model.fit(X_train, y_train, epochs=80, batch_size=128)

    # 手动调参的模型
    model = gru_model(X_train)
    hist = model.fit(X_train, y_train, epochs=80, batch_size=145, validation_split=0, verbose=0)  # 训练模型epoch=80次

    # 展示在训练集上的表现
    plt.rc('font', family='serif')
    fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # ax_acc.set_title("index对照组训练过程")  # 图标题，位于图上方正中
    ax_acc.plot(hist.history['accuracy'], color='red')  # 左侧坐标 精度曲线  红色
    ax_acc.set_ylabel('accuracy', color='red')  # y坐标的标签 坐标左边正中
    ax_loss = ax_acc.twinx()  # 设置同图的右侧坐标
    ax_loss.plot(hist.history['loss'], color='blue')  # 右侧坐标 损失曲线  蓝色
    ax_loss.set_ylabel('loss', color='blue')
    plt.xlabel('epochs')
    plt.show()

    # 6-3、对测试集数据的预测
    y_test_predict = model.predict(X_test)  # 对测试集预测拟合

    y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
    y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_test_predict[0] /= min_max_scaler.scale_[-1]

    # 6-4、获取 dfx 最后的 cut 个数据
    test_predict = pd.DataFrame(dfx.iloc[-cut:, amount_of_features - 1], columns=['close'])
    test_predict['predict'] = y_test_predict[0].values

    print("index对照组预测的精度：")
    calculate_metric(test_predict['close'], test_predict['predict'])

    # 6-5、记录使用待检验因子和收盘价预测的结果
    col_name = 'index_predict'
    all_result[col_name] = test_predict['predict']

    # 7、保存预测的数据和真实值
    # 记录之前仅用收盘价进行预测的结果
    all_result['all_predict'] = all_predict['predict']

    # 记录真实值
    all_result['close'] = all_predict['close']
    return last_cut_indices, all_result


# 由键盘输入文件名
read_filename = input("请输入文件名：")  # 中国平安、浦发银行
# 用于存储预测结果的列表
predictions = []
date = []
# 进行十次预测并将预测结果添加到列表中
for _ in range(10):
    date, prediction = predict_close(read_filename)
    print("实验的次数:", _)
    # 将预测结果添加到列表中
    predictions.append(prediction)

# 创建空的平均值 DataFrame
average_df = pd.DataFrame()

# 对每个预测结果进行累加求和
for k in range(len(predictions)):
    # 将每个预测结果添加到平均值 DataFrame
    average_df = average_df.add(predictions[k], fill_value=0)

# 求平均值，并除以预测次数
average_df = average_df / len(predictions)
# 将最后 cut 个索引保存到 all_result 的 date 列
average_df.insert(0, 'date', date)

# 由键盘输入文件名
save_filename = read_filename + '对照组和实验组预测结果'
path = "D:\\Research\\result\\" + save_filename + ".csv"
average_df.to_csv(path, encoding="gbk", index=False)
