# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import time


def hinge_loss(weight, sample, label):
    return max(0, 1 - label * np.dot(weight, sample))


def calc_obj_svm(w, xmat, y, lmd):
    obj_val = (lmd / 2.) * np.linalg.norm(w)
    obj_val += np.mean([hinge_loss(w, xmat[i], y[i]) for i in xrange(len(y))])
    return obj_val


def update_pa(weight, sample, label, pa_type, cost=1e0):
    # PA
    if pa_type == 0:
        step_size = hinge_loss(weight, sample, label) / np.dot(sample, sample)
    # PA-I
    elif pa_type == 1:
        step_size = min(cost, hinge_loss(weight, sample, label) / np.dot(sample, sample))
    # PA-II
    else:
        step_size = hinge_loss(weight, sample, label) / (np.dot(sample, sample) + 1 / cost)
    return weight + step_size * label * sample


def update_pgdc(weight, sample, label, cost=1e0, eta1=1e-3, eta2=1e-3):
    # step 1
    if label * np.dot(weight, sample) < 0:
        weight -= cost * eta1 * label * sample
    # step 2
    step_size = min(cost, hinge_loss(weight, sample, label) / (eta2 * np.dot(sample, sample)) + 1 / np.dot(sample, sample))
    return (1 / (1+eta2)) * weight + step_size * label * sample


def oppai(w, x, y, eta, lmd):
    alpha = (1. - y * np.dot(w, x) + eta * lmd) / (eta * np.linalg.norm(x))
    if alpha < 0:
        alpha = 0.
    if alpha > 1:
        alpha = 1.
    return (w + eta * alpha * y * x) / (eta * lmd + 1)


def fobos(w, x, y, eta, lmd):
    if 1 - y * np.dot(w, x) >= 0:
        return (w + eta * y * x) / (eta * lmd + 1)
    else:
        return w / (eta * lmd + 1)


def sgd(w, x, y, eta, lmd):
    if 1 - y * np.dot(w, x) >= 0:
        return w - eta * (lmd * w - y * x)
    else:
        return w - eta * lmd * w


if __name__ == '__main__':
    dataset = np.loadtxt('data/libsvm/adult/a1a.csv', delimiter=',')
    labels = dataset[:, -1]
    dmat = dataset[:, :-1]
    labels = dataset[:, 0]
    dmat = dataset[:, 1:]
    m, n = np.shape(dmat)

    itr = 10000
    lmd = 1e-3
    eta = 10.
    w_oppai = np.zeros(n)
    w_sgd = np.zeros(n)
    w_fobos = np.zeros(n)
    obj_oppai = []
    obj_fobos = []
    obj_sgd = []

    np.random.seed(0)

    start = time.time()

    for i in xrange(itr):
        ind = np.random.randint(m)
        w_oppai = oppai(w=w_oppai, x=dmat[ind%m], y=labels[ind%m], eta=eta, lmd=lmd)
        if i % 3000 == 0:
            obj_oppai.append(calc_obj_svm(w_oppai, dmat, labels, lmd))

        w_sgd = sgd(w=w_sgd, x=dmat[ind%m], y=labels[ind%m], eta=eta, lmd=lmd)
        if i % 3000 == 0:
            obj_sgd.append(calc_obj_svm(w_sgd, dmat, labels, lmd))
        #
        w_fobos = fobos(w=w_fobos, x=dmat[ind%m], y=labels[ind%m], eta=eta, lmd=lmd)
        if i % 3000 == 0:
            obj_fobos.append(calc_obj_svm(w_fobos, dmat, labels, lmd))

    end = time.time()
    print 'comp.time:', end - start

    plt.plot(obj_sgd, label='SGD')
    plt.plot(obj_fobos, label='FOBOS')
    plt.plot(obj_oppai, label='OPPAI')
    plt.legend()
    # plt.yscale('log')
    plt.grid()
    plt.show()
