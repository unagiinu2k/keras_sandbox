#https://elix-tech.github.io/ja/2016/06/02/kaggle-facial-keypoints-ja.html
#http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle

FTRAIN = 'data/training.csv'
FTEST = 'data/test.csv'


def load(test=False, cols=None):
    """testがTrueの場合はFTESTからデータを読み込み、Falseの場合はFTRAINから読み込みます。
    colsにリストが渡された場合にはそのカラムに関するデータのみ返します。
    """

    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) # pandasのdataframeを使用
    if False:
        type(df)
        df.shape
        df.columns
        df.head(1)
        #imageというカラムに画像データが入っていて、ほかのカラムには鼻の位置などがx,y座標にわけて入っている
        type(df['Image'][0])

    # スペースで句切られているピクセル値をnumpy arrayに変換
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # カラムに関連するデータのみを抽出
        df = df[list(cols) + ['Image']]

    print(df.count())  # カラム毎に値が存在する行数を出力
    df = df.dropna()  # データが欠けている行は捨てる

    X = np.vstack(df['Image'].values) / 255.  # 0から1の値に変換
    X = X.astype(np.float32)

    if not test:  # ラベルが存在するのはFTRAINのみ
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # -1から1の値に変換
        X, y = shuffle(X, y, random_state=42)  # データをシャッフル
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

X, y = load()
if False:
   X[1].max()
   X[1].min()
   #Xは96x96=9216次元の濃淡データ
   #各要素は０から１の間に規格化

print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(
    X.shape, X.min(), X.max()))
print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(
    y.shape, y.min(), y.max()))


from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

model = Sequential()
model.add(Dense(100, input_dim=9216))
model.add(Activation('relu'))
model.add(Dense(30))

sgd = SGD(lr='0.01', momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)
hist = model.fit(X, y, nb_epoch=10, validation_split=0.2)#20%のデータはvalidation用に使っている

from matplotlib import pyplot
pyplot.plot(hist.history['loss'], linewidth=3, label='train')
pyplot.plot(hist.history['val_loss'], linewidth=3, label='valid')
pyplot.grid()
pyplot.legend()
pyplot.xlabel('epoch')
pyplot.ylabel('loss')
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale('log')
pyplot.show()


X_test, _ = load(test=True)
y_test = model.predict(X_test)

fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


for i in range(16):
    axis = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    plot_sample(X_test[i], y_test[i], axis)

pyplot.show()


def load2d(test=False, cols=None):
    X, y = load(test, cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y


from keras.layers import Convolution2D, MaxPooling2D, Flatten

X, y = load2d()
model2 = Sequential()

model2.add(Convolution2D(32, 3, 3, input_shape=(1, 96, 96)))#32レイヤーの94x94 latticeを生成
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2))) #32レイヤーの47x47 latticeを生成

model2.add(Convolution2D(64, 2, 2)) #64レイヤーの46x46 latticeを生成
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2))) #64レイヤーの23x23 latticeを生成

model2.add(Convolution2D(128, 2, 2))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))

model2.add(Flatten()) #128*11*11を15488次元のflatな構造に
model2.add(Dense(500))
model2.add(Activation('relu'))
model2.add(Dense(500))
model2.add(Activation('relu'))
model2.add(Dense(30))

sgd = SGD(lr='0.01', momentum=0.9, nesterov=True)
model2.compile(loss='mean_squared_error', optimizer=sgd)
hist2 = model2.fit(X, y, nb_epoch=10, validation_split=0.2)

from keras.utils.visualize_util import plot
plot(model2, to_file='model2.png', show_shapes=True)

sample1 = load(test=True)[0][6:7]
sample2 = load2d(test=True)[0][6:7]
y_pred1 = model.predict(sample1)[0]
y_pred2 = model2.predict(sample2)[0]

fig = pyplot.figure(figsize=(6, 3))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
plot_sample(sample1, y_pred1, ax)
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
plot_sample(sample2, y_pred2, ax)
pyplot.show()