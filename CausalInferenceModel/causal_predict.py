import os

import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# import shap
from f_test import causal_inference
from keras.layers import *
from keras.models import *
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# 计算MAPE
def mape(true, predict, mask):
    record = np.abs((predict - true) / true)  # 取绝对值并计算相对误差
    record *= mask  # record = record * mask
    non_zero_len = mask.sum()
    return (np.sum(record) / non_zero_len) * 100


def plot_prediction(true, causal_p, filename):
    fig, axs = plt.subplots(figsize=(12, 6))

    # train_index = "Comparison between Predicted Values and Ground Truth"
    # axs.set_title(train_index, fontproperties='Times New Roman', fontsize=19, y=1.05)  # 图标题，位于图上方正中
    axs.plot(true[:], color='#4B4453')
    axs.plot(causal_p[:], color='#D83121')
    plt.xticks(fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=16)
    axs.set_xlabel('Date(Days)', fontproperties='Times New Roman', fontsize=17)
    axs.xaxis.set_label_coords(1.06, -0.02)  # 设置横坐标标签在右下方
    axs.set_ylabel('Close Price(Yuan)', fontproperties='Times New Roman', fontsize=17, rotation=0)
    axs.yaxis.set_label_coords(0, 1.0)  # 设置纵坐标标签在左上方
    axs.legend(('Ground Truth', 'Causal Factors + Close Price'), fontsize='14')
    axs.grid(True)

    save_filename = filename
    graph_name = save_filename + "真实值和预测值比较"
    directory = "result_graph"  # 子目录名称
    path = os.path.join(directory, graph_name)  # 将目录和图像名合并为完整的保存路径
    plt.savefig(path)
    plt.savefig(os.path.join("svg_graph", graph_name + '.svg'), format='svg')
    plt.show()


def plot_scatter(true, predict, filename):
    fig, axs = plt.subplots(figsize=(6, 6))

    plot_begin, plot_end = min(min(true), min(predict)), max(max(true), max(predict))
    plot_x = np.linspace(plot_begin, plot_end, 10)
    axs.plot(plot_x, plot_x)
    axs.set_xlabel('Ground Truth(Yuan)')
    axs.set_ylabel('Predicted Value(Yuan)')
    axs.plot(true, predict, 'o', color='y')
    axs.grid(True)

    save_filename = filename
    graph_name = save_filename + "预测值准确性散点图"
    directory = "result_graph"  # 子目录名称
    path = os.path.join(directory, graph_name)  # 将目录和图像名合并为完整的保存路径
    plt.savefig(path)
    plt.savefig(os.path.join("svg_graph", graph_name + '.svg'), format='svg')
    plt.show()


def calculate_metric(true, predict):
    # 计算相对变化
    true_relative = np.diff(true) / true[:-1]
    predict_relative = np.diff(predict) / predict[:-1]
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
    model.add(GRU(256, activation='tanh', return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(GRU(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


def cnn_gru_model(window, amount_of_features):
    model = Sequential()
    # 第一层，filters卷积核个数，kernel_size卷积核大小，padding卷积填充方式，strides卷积步长，activation激活函数，input_shape为声明网络输入的形状（第一层需要声明）
    model.add(Conv1D(filters=15, kernel_size=6, padding='same', strides=1, activation='relu',
                     input_shape=(window, amount_of_features)))
    model.add(MaxPooling1D(pool_size=1))  # 池化层
    # model.add(Conv1D(filters=35,kernel_size=3, padding='same', strides=1, activation='relu'))#第二层卷积层
    # model.add(MaxPooling1D(pool_size=1))#第二层池化层
    # gru层，units为神经元数，activation激活函数，return_sequences后面还有gru层需要设置为True
    model.add(GRU(units=128, activation='relu', return_sequences=True))
    # 第二层gru，units为神经元数，activation激活函数，return_sequences后面没有gru层需要设置为False(默认)
    model.add(GRU(units=256, activation='relu'))
    model.add(Dropout(0.2))  # 防止过拟合层
    model.add(Dense(1))  # 全连接层
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])  # 指定loss函数和优化器类型
    # 总结模型
    model.summary()
    return model


# 模型可解释性
# def explain_model(model, X_train, X_test):
#     # 创建一个解释器对象
#     explainer = shap.DeepExplainer(model, X_train)
#     # 计算每个特征的 SHAP 值
#     shap_values = explainer.shap_values(X_test)
#     # 绘制SHAP摘要图
#     shap.summary_plot(shap_values, X_test)


def causal_predict(filename):
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
    # T = dfx.shape[0]
    print(dfx)
    # 假设 df 是你的 DataFrame 对象
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

    # 热力图相关性分析，需要预测的特征列与其他列颜色越深，相关性越大
    plt.figure(figsize=(10, 8))
    sns.heatmap(df1.corr(method='spearman'), cmap='RdYlBu_r')
    plt.title('Correlation Analysis', size=15)

    heatmap_name = filename + "因子热力图相关性分析"
    directory = "result_graph"  # 子目录名称
    path = os.path.join(directory, heatmap_name)  # 将目录和图像名合并为完整的保存路径
    plt.savefig(path)
    plt.savefig(os.path.join("svg_graph", heatmap_name + '.svg'), format='svg')
    plt.show()

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

    # 6、使用因果因子预测
    f_df, causal_col = causal_inference(filename)
    # causal_col = [x for x in causal_col if x < 11]
    causal_col.append(amount_of_features - 1)
    df_causal = df.iloc[:, causal_col]
    data = df_causal.iloc[:, :].values  # pd.DataFrame(stock) 表格转化为矩阵

    # 6-1、按照window_size的数值将元数据变换成 组、行x列 的3维矩阵
    result = []
    for j in range(len(data) - sequence_length + 1):  # 循环 数据长度-时间窗长度+1 次
        result.append(data[j:j + sequence_length, :])  # 第j行到j+window_size

    result = np.array(result)  # 得到样本，样本形式为 sequence_length * amount_of_features
    # print('result:', result)

    cut = int(cut_ratio * len(result))  # 按比例划分训练集测试集 最后cut个样本为测试集

    train = result[:-cut, :]  # 取出 除最后 cut 组的数据作为训练数据
    x_train = train[:, :-1]  # 所有组、不包括最后1列的所有行，训练集特征
    y_train = train[:, -1][:, -1:]  # 所有组、最后1行，最后1列，训练集的目标变量

    test = result[-cut:, :]  # 最后 cut 组作为测试集
    x_test = test[:, :-1]  # 最后一列以外的所有列为测试集特征
    # y_test = test[:, -1][:, -1:]  # 测试集的目标变量

    X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))  # 组 行 列
    X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_train.shape[2]))
    # 展示下训练集测试集的形状 看有没有问题
    print("X_train", X_train.shape)
    print("y_train", y_train.shape)
    print("X_test", X_test.shape)

    # 6-2、建立模型、训练模型过程
    # Hyperband算法
    tuner = kt.Hyperband(
        build_gru_model,
        objective='accuracy',
        max_epochs=80,
        factor=3,
        directory='tuner_directory',
        project_name='pingan_causal_gru_tuning'
    )
    tuner.search(X_train, y_train, epochs=80, batch_size=128)
    # best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    # 使用最佳超参数构建最终模型
    # model = tuner.hypermodel.build(best_hyperparameters)
    # hist = model.fit(X_train, y_train, epochs=80, batch_size=128)

    # 手动调参的模型
    model = gru_model(X_train)  # len(df_causal.columns)因果因子数特征
    hist = model.fit(X_train, y_train, epochs=80, batch_size=128, validation_split=0, verbose=0)  # 训练模型epoch=80次

    # 展示在训练集上的表现
    plt.rc('font', family='serif')
    fig, ax_acc = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    ax_acc.set_title("Causal Factors + Close Price")  # 图标题，位于图上方正中
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
    # y_train = pd.DataFrame(y_train)  # 将ndarray变为dataframe格式
    # y_train[0] -= min_max_scaler.min_[-1]  # 反归一化
    # y_train[0] /= min_max_scaler.scale_[-1]

    # 6-5、对测试集数据的预测
    y_test_predict = model.predict(X_test)  # 对测试集预测拟合

    # 画出神经网络结构图
    plot_model(model, to_file='NetworkStructureDiagram/model.png', show_shapes=True)
    # 获得关于模型预测如何受不同特征影响的详细解释
    # explain_model(model, X_train, X_test)

    y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
    y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
    y_test_predict[0] /= min_max_scaler.scale_[-1]
    # print('\n下一日' + pred_str + '预测为： ', y_test_predict[0])  # pred_str='close'

    # 将dfx中最前面的 window_size 个以及最后面的 cut 个数据去除
    train_pre = dfx.iloc[window_size:-cut, causal_col]
    y_train_predict = y_train_predict[0].values
    train_pre.loc[:, 'predict'] = y_train_predict

    # 获取 dfx 最后的 cut 个数据
    c_pre = dfx.iloc[-cut:, causal_col]
    # 获取预测值
    y_test_predict = y_test_predict[0].values
    # 将 y_test_predict 当做值赋给 'predict' 列
    c_pre.loc[:, 'causal_predict'] = y_test_predict
    return f_df, c_pre


if __name__ == "__main__":
    # 由键盘输入文件名
    read_filename = input("请输入用于收盘价预测的文件名：")  # 中国太保
    # read_filename = '中国平安'
    f_v, causal_pre = causal_predict(read_filename)
    y_true = causal_pre['close']
    y_causal_predict = causal_pre['causal_predict']

    plot_prediction(y_true, y_causal_predict, read_filename)
    plot_scatter(y_true, y_causal_predict, read_filename)
    print('计算因果预测的绝对评估指标：')
    calculate_metric(y_true, y_causal_predict)

    min_max_scaler = preprocessing.MinMaxScaler()  # 采用最大最小格式归一化
    causal_pre_scaler = min_max_scaler.fit_transform(causal_pre)  # causal_pre_scaler 为ndarray形式，无列标签
    causal_pre_scaler = pd.DataFrame(causal_pre_scaler, columns=causal_pre.columns)
    y_true_scaler = causal_pre_scaler['close']
    y_predict_scaler = causal_pre_scaler['causal_predict']
    print('计算因果预测的相对评估指标：')
    calculate_metric(y_true_scaler, y_predict_scaler)

    # 由键盘输入文件名
    date = causal_pre.index[:]
    causal_pre.insert(0, 'date', date)
    causal_filename = read_filename + '因果预测结果'
    causal_path = "D:\\Research\\result\\" + causal_filename + ".csv"
    causal_pre.to_csv(causal_path, encoding="gbk", index=False)

    # 由键盘输入文件名
    f_filename = read_filename + 'F值'
    f_path = "D:\\Research\\result\\" + f_filename + ".csv"
    f_v.to_csv(f_path, encoding="gbk", index=False)