import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import scipy.io

def custom_cross_val_score(clf, X, y, k=5):
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // k
    accuracies = []

    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        clf.fit(X_train, y_train)
        accuracy = clf.score(X_test, y_test)
        accuracies.append(accuracy)

    return np.array(accuracies)


def test_dual_svm():
    np.random.seed(11)

    C = 10
    n = 100

    data = scipy.io.loadmat('datanew.mat')

    dataX_N = data['X_N']
    dataX_P = data['X_P']

    x_p = dataX_P[:n, :]
    x_n = dataX_N[:n, :]

    y_p = np.ones(n)
    y_n = -np.ones(n)

    x = np.vstack((x_p, x_n))
    y = np.hstack((y_p, y_n))

    clf = svm.SVC(C=C, kernel='rbf')

    # 五折交叉验证
    scores = custom_cross_val_score(clf, x, y)

    print(f'Accuracy for each fold: {scores}')
    print(f'Mean accuracy: {np.mean(scores)}')

    plt.scatter(x_p[:, 0], x_p[:, 1], marker='x', color='b')
    plt.scatter(x_n[:, 0], x_n[:, 1], marker='.', color='g')

    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], marker='o', color='r')

    x_min, x_max = -10, 8
    y_min, y_max = -10, 7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(xx.shape)

    # 画图
    plt.contour(xx, yy, z, colors='m')

    plt.title('SVM Decision Boundary and Support Vectors')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.legend(['Positive Class', 'Negative Class', 'Support Vectors'])

    plt.show()


test_dual_svm()