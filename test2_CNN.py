import numpy 
import matplotlib.pyplot as plt
import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Conv1D, MaxPooling1D
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

from Multi_adaboost_CNN import AdaBoostClassifier as Ada_CNN  # 你的AdaBoost-CNN

# --- baseline_model 需加 n_classes 參數 ---
def baseline_model(n_features=10, n_classes=3, seed=100):
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    model = Sequential()
    model.add(Conv1D(32, 3, padding="same", input_shape=(n_features, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['accuracy'])
    print (model.summary())
    return model

def train_CNN(X_train=None, y_train=None, epochs=None, batch_size=None, X_test=None, y_test=None, n_features=10, n_classes=3, seed=100):
    numpy.random.seed(seed)
    tensorflow.random.set_seed(seed)
    model = baseline_model(n_features, n_classes, seed)
    lb = OneHotEncoder(sparse_output=False)
    y_train_b = lb.fit_transform(y_train.reshape(-1, 1))
    newshape = list(X_train.shape) + [1]
    X_train_r = numpy.reshape(X_train, newshape)
    model.fit(X_train_r, y_train_b, epochs=epochs, batch_size=batch_size, verbose=0)
    newshape = list(X_test.shape) + [1]
    X_test_r = numpy.reshape(X_test, newshape)
    y_test_b = lb.transform(y_test.reshape(-1, 1))
    print('\nSingle CNN evaluation on training data, [loss, test_accuracy]:')
    train_eval = model.evaluate(X_train_r, y_train_b, verbose=0)
    print(train_eval)
    print('\nSingle CNN evaluation on testing data, [loss, test_accuracy]:')
    test_eval = model.evaluate(X_test_r, y_test_b, verbose=0)
    print(test_eval)
    return model

def synethetic_data(n_features=10, n_classes=3):
    from sklearn.datasets import make_gaussian_quantiles
    import matplotlib as mpl

    def plot_hist(y_test, oName0):
        mpl.rc('font', family='Times New Roman')
        (n, bins, patches) = plt.hist(y_test+1, bins=[0.75, 1.25, 1.75, 2.25, 2.75, 3.25])
        plt.xlabel('Class')
        plt.ylabel('# Samples')
        oName = oName0 + '_hist.png'
        plt.savefig(oName, dpi=200)
        plt.show()
        print_t = 'The Histogram of the data is saved as: ' + oName
        print(print_t)
        print(n)
        print(n/len(y_test))
        labels = 'C1', 'C2', 'C3'
        sizes = [v for i, v in enumerate(n) if (i % 2) == 0]
        explode = (0, 0, 0.1)
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.axis('equal')
        oName = oName0 + '_pie.png'
        plt.savefig(oName, dpi=200)
        plt.show()
        print_t = 'The Histogram of the data is saved as: ' + oName
        print(print_t)

    X, y = make_gaussian_quantiles(n_samples=13000, n_features=n_features,
                                   n_classes=n_classes, random_state=1)
    n_split = int(0.8 * X.shape[0])
    X_train, X_test = X[:n_split], X[n_split:]
    y_train, y_test = y[:n_split], y[n_split:]
    N_re = [500, 2000]
    a = [index for index, v in enumerate(y_train) if v == 0]
    y_train = numpy.delete(y_train, a[0:N_re[0]])
    X_train = numpy.delete(X_train, a[0:N_re[0]], axis=0)
    a = [index for index, v in enumerate(y_train) if v == 1]
    y_train = numpy.delete(y_train, a[0:N_re[1]])
    X_train = numpy.delete(X_train, a[0:N_re[1]], axis=0)
    plot_hist(y_train, oName0='synethetic_train')
    plot_hist(y_test, oName0='synethetic_test')
    return X_train, y_train, X_test, y_test

def reshape_for_CNN(X):
    newshape = list(X.shape) + [1]
    return numpy.reshape(X, newshape)

# ------------------- 主程式 ---------------------
n_features = 10
n_classes = 3
batch_size = 10
epochs = 10
seed = 50
n_estimators = 10
epochs_boost = 1

# 資料生成
X_train, y_train, X_test, y_test = synethetic_data(n_features=n_features, n_classes=n_classes)
X_train_r = reshape_for_CNN(X_train)
X_test_r = reshape_for_CNN(X_test)

# --- AdaBoost+CNN ---
bdt_real_test_CNN = Ada_CNN(
    base_estimator=baseline_model(n_features=n_features, n_classes=n_classes, seed=seed),
    n_estimators=n_estimators,
    learning_rate=1,
    epochs=epochs_boost
)
bdt_real_test_CNN.fit(X_train_r, y_train, batch_size)
ada_train_pred = bdt_real_test_CNN.predict(X_train_r)
ada_test_pred  = bdt_real_test_CNN.predict(X_test_r)
ada_train_acc = accuracy_score(ada_train_pred, y_train)
ada_test_acc  = accuracy_score(ada_test_pred, y_test)
print('\n Training accuracy of bdt_real_test_CNN (AdaBoost+CNN):', ada_train_acc)
print('\n Testing accuracy of bdt_real_test_CNN (AdaBoost+CNN):', ada_test_acc)

# --- 單一CNN ---
cnn_model = train_CNN(
    X_train=X_train, y_train=y_train,
    epochs=epochs, batch_size=batch_size,
    X_test=X_test, y_test=y_test,
    n_features=n_features, n_classes=n_classes, seed=seed
)
cnn_train_pred = cnn_model.predict(X_train_r).argmax(axis=1)
cnn_test_pred  = cnn_model.predict(X_test_r).argmax(axis=1)
cnn_train_acc = accuracy_score(cnn_train_pred, y_train)
cnn_test_acc = accuracy_score(cnn_test_pred, y_test)

# --- 畫模型比較圖 ---
labels = ['Train', 'Test']
cnn_scores = [cnn_train_acc, cnn_test_acc]
ada_scores = [ada_train_acc, ada_test_acc]
x = numpy.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, cnn_scores, width, label='Single CNN')
rects2 = ax.bar(x + width/2, ada_scores, width, label='AdaBoost-CNN')
ax.set_ylabel('Accuracy')
ax.set_title('Model Comparison on Imbalanced Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1)
ax.legend()
for rects in [rects1, rects2]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=200)
plt.show()
