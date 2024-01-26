import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.optimize import minimize

def gaussian_kernel(x1, x2, sigma): #高斯核
    diff = x1[:, np.newaxis] - x2
    return np.exp(-np.sum(diff) ** 2 / (2 * sigma ** 2))

def linear_kernel(X,Y): # 线性核
    return np.outer(Y.T, Y) * np.dot(X.T, X)

def svm_train(X, Y, C, sigma):
    n = len(Y)
    f = -1 * np.ones(n)
    A = np.empty((0, n))
    b = np.empty((0, 1))
    Aeq = Y  # 约束条件
    beq = 0
    lb = np.zeros(n)
    ub = C * np.ones(n)
    a0 = np.zeros(n)

    # H = np.zeros((n, n))
    # for i in range(n):
    #     for j in range(n):
    #         H[i, j] = Y[i] * Y[j] * gaussian_kernel(X[:, i], X[:, j], sigma)

    H=linear_kernel(X, Y)

    def objective(a):
        return 0.5 * np.dot(a, np.dot(H, a)) + np.dot(f, a)
    constraints = [{'type': 'eq', 'fun': lambda a: np.dot(Aeq, a) - beq}]
    bounds = list(zip(lb, ub))
    result = minimize(objective, a0, method='SLSQP', bounds=bounds, constraints=constraints)
    a = result.x
    epsilon = 1e-8
    sv_label = np.where(np.abs(a) > epsilon)[0] # 支持向量

    svm = {
        'a': a[sv_label],
        'Xsv': X[:, sv_label],
        'Ysv': Y[sv_label],
        'svnum': len(sv_label),
        'sigma': sigma
    }

    return svm

def linear_svm_test(svm, Xt, Yt):
    w = np.sum((svm['a'] * svm['Ysv']) * svm['Xsv'], axis=1)
    b = np.mean(svm['Ysv'] - np.dot(w, svm['Xsv']))
    score = np.dot(w, Xt) + b
    Y = np.sign(score)

    result = {
        'score': score,
        'Y': Y,
        'accuracy': np.sum(Y == Yt) / len(Yt)
    }

    return result

def gaussian_svm_test(svm, Xt, Yt):
    w = np.sum((svm['a'] * svm['Ysv']) * np.array([gaussian_kernel(svm['Xsv'][:, i], Xt, svm['sigma']) for i in range(svm['svnum'])]), axis=0)
    b = np.mean(svm['Ysv'] - np.dot(w, svm['Xsv']))
    score = np.dot(w, Xt) + b
    Y = np.sign(score)

    result = {
        'score': score,
        'Y': Y,
        'accuracy': np.sum(Y == Yt) / len(Yt)
    }

    return result

def main():
    C = 10
    n = 80
    sigma = 2.0

    data = scipy.io.loadmat('datanew.mat')
    x1 = data['X_P'][:n, :].T
    y1 = np.ones(n)
    x2 = data['X_N'][:n, :].T
    y2 = -np.ones(n)

    X = np.hstack((x1, x2))
    Y = np.hstack((y1, y2))

    svm = svm_train(X, Y, C, sigma)

    plt.figure()
    plt.plot(x1[0, :], x1[1, :], 'bx', x2[0, :], x2[1, :], 'g.')
    plt.axis([-11, 8, -11, 8])
    plt.gca().set_prop_cycle(None)

    plt.scatter(svm['Xsv'][0, :], svm['Xsv'][1, :], c='r', marker='o')

    x1, x2 = np.meshgrid(np.arange(-11, 8, 0.05), np.arange(-11, 8, 0.05))
    Xt = np.vstack((x1.ravel(), x2.ravel()))
    Yt = np.ones(len(x1.ravel()))

    result = linear_svm_test(svm, Xt, Yt)

    Yd = result['Y'].reshape(x1.shape)
    plt.contour(x1, x2, Yd, colors='m')

    plt.show()
    print(result['accuracy'])

main()