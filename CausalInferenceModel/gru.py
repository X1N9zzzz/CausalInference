import keras_tuner as kt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import *
from keras.models import *
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_prediction(true, predict):
    fig, axs = plt.subplots(figsize=(12, 6))

    # train_index = "Comparison between Predicted Values and Ground Truth"
    # axs.set_title(train_index, fontproperties='Times New Roman', fontsize=19, y=1.05)  # 图标题，位于图上方正中
    axs.plot(true[:], color='#4B4453')
    axs.plot(predict[:], color='#D83121')
    plt.xticks(fontproperties='Times New Roman', fontsize=16)
    plt.yticks(fontproperties='Times New Roman', fontsize=16)
    axs.set_xlabel('Date(Days)', fontproperties='Times New Roman', fontsize=17)
    axs.xaxis.set_label_coords(1.06, -0.02)  # 设置横坐标标签在右下方
    axs.set_ylabel('Close Price(Yuan)', fontproperties='Times New Roman', fontsize=17, rotation=0)
    axs.yaxis.set_label_coords(0, 1.0)  # 设置纵坐标标签在左上方
    axs.legend(('Ground Truth', 'Causal Factors + Close Price'), fontsize='14')
    axs.grid(True)
    plt.show()


# 计算MAPE
def mape(true, predict, mask):
    record = np.abs((predict - true) / true)  # 取绝对值并计算相对误差
    record *= mask  # record = record * mask
    non_zero_len = mask.sum()
    return (np.sum(record) / non_zero_len) * 100


def calculate_metric(true, predict):
    # 输出结果
    print(' RMSE:', np.sqrt(mean_squared_error(true, predict)))
    y_mask = (1 - (true == 0))
    print(' MAE:', mean_absolute_error(true, predict))
    print('MAPE:', mape(true, predict, y_mask))
    print(' R^2:', r2_score(true, predict))
    return


def gru_model(input_data):
    model = Sequential()
    model.add(Input(shape=(input_data.shape[1], input_data.shape[2])))
    # model.add(Conv1D(filters=15, kernel_size=6, padding='same', strides=1, activation='relu'))
    # model.add(MaxPooling1D(pool_size=1))  # 池化层
    model.add(GRU(256, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(GRU(256, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='uniform', activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model


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


read_filename = input("请输入用于收盘价预测的文件名：")  # 中国平安
# 1、读取csv数据文件
path = r'D:\Research\dataset\\' + read_filename + '.csv'
dfx = pd.read_csv(path, index_col="date", parse_dates=True, encoding='GBK')  # Date，索引、df日期格式
dfx = dfx.fillna(method='pad')  # 若有空值，则用上一个值填充

# 检查整个 DataFrame 是否有缺失值
missing_values = dfx.isnull().sum().sum()
print("缺失值的数量：", missing_values)

# 检查每列是否有缺失值
missing_values_per_column = dfx.isnull().sum()
print("每列的缺失值数量：")
print(missing_values_per_column)

dfx = dfx.fillna(method='pad')  # 若有空值，则用上一个值填充

order = ['open', 'high', 'low', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM',
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
    project_name='maotai_all_input_gru_tuning'
)
tuner.search(X_train, y_train, epochs=80, batch_size=128)
# best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

# 使用最佳超参数构建最终模型
# model = tuner.hypermodel.build(best_hyperparameters)
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

# 画出神经网络结构图
plot_model(model, to_file='NetworkStructureDiagram/model.png', show_shapes=True)

# 对测试集数据的预测
y_test_predict = model.predict(X_test)  # 对测试集预测拟合
y_test_predict = pd.DataFrame(y_test_predict)  # 将ndarray变为dataframe格式
y_test_predict[0] -= min_max_scaler.min_[-1]  # 反归一化
y_test_predict[0] /= min_max_scaler.scale_[-1]

# 将预测结果保存在 all_predict 中
predict_result = dfx.iloc[-cut:, :]
y_test_predict = y_test_predict[0].values
predict_result.loc[:, 'predict'] = y_test_predict

y_true = predict_result['close']
y_predict = predict_result['predict']
print('计算预测的绝对评估指标：')
calculate_metric(y_true, y_predict)
plot_prediction(y_true, y_predict)

predict_result_scaler = min_max_scaler.fit_transform(predict_result)  # df2 为ndarray形式，无列标签
predict_result_scaler = pd.DataFrame(predict_result_scaler, columns=predict_result.columns)
y_true_scaler = predict_result_scaler['close']
y_predict_scaler = predict_result_scaler['predict']
print('计算预测的相对评估指标：')
calculate_metric(y_true_scaler, y_predict_scaler)
