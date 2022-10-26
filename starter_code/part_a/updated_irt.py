from utils import *
import matplotlib.pyplot as plt
import numpy as np

# This is an implementation of 4-pl IRT model


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, alpha, c, d):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    log_lklihood = 0.
    for i in range(len(data["user_id"])):
        theta_i = theta[data["user_id"][i]]
        beta_j = beta[data["question_id"][i]]
        a = alpha[data["question_id"][i]]
        sig = sigmoid(a * (theta_i - beta_j))
        cij = data['is_correct'][i]
        ci = c[data["user_id"][i]]
        di = d[data["user_id"][i]]
        log_lklihood += (cij * np.log(ci + (di-ci)*sig) + (1-cij) * np.log(1-(ci + (di-ci)*sig)))  # May be modified

    return -log_lklihood


def sparse_matrix(data):
    matrix = np.empty(shape=(542, 1774))
    matrix[:] = np.NaN
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        mark = data["is_correct"][i]
        matrix[u, q] = mark
    return matrix


def update_theta_beta(data, lr, theta, beta, alpha, c, d):

    theta_copy = theta.copy()
    beta_copy = beta.copy()
    alpha_copy = alpha.copy()
    c_copy = c.copy()
    d_copy = d.copy()

    train_matrix = sparse_matrix(data)

    for i in range(len(theta)):
        c_decay = []
        d_decay = []
        t = 0.
        for j, q_id in enumerate(train_matrix[i, :]):
            if np.isnan(q_id):
                continue
            else:
                k = alpha_copy[j] * (theta_copy[i] - beta_copy[j])
                diff = alpha_copy[j] * (d_copy[i] - c_copy[i])
                t += q_id * (diff * sigmoid(k) / (d_copy[i] * np.exp(k) + c_copy[i])) + (1 - q_id) * (-diff * sigmoid(k) / (c_copy[i] * np.exp(k) + d_copy[i]))
                c_decay.append(q_id * (1 / (c_copy[i] + d_copy[i] * np.exp(k))) + (1 - q_id) * (np.exp(k) / (d_copy[i] + c_copy[i] * np.exp(k))))
                d_decay.append(q_id * (-np.exp(k) / (c_copy[i] + d_copy[i] * np.exp(k))) + (1 - q_id) * (-1 / (d_copy[i] + c_copy[i] * np.exp(k))))

        theta[i] += lr * t
        c[i] += lr * np.mean(c_copy)
        d[i] += lr * np.mean(d_copy)

    for j in range(len(beta)):
        b = 0.
        alpha_grad = 0.
        for i, u_id in enumerate(train_matrix[:, j]):
            if np.isnan(u_id):
                continue
            else:
                k = alpha_copy[j] * (theta_copy[i] - beta_copy[j])
                diff = (d_copy[i] - c_copy[i]) * alpha_copy[j]
                b += u_id * (-diff * sigmoid(k) / (d_copy[i] * np.exp(k) + c_copy[i])) + (1 - u_id) * (
                            diff * sigmoid(k) / (c_copy[i] * np.exp(k) + d_copy[i]))
                alpha_grad += u_id * (-diff * sigmoid(k) * (theta_copy[i] - beta_copy[j]) / (
                            d_copy[i] * np.exp(k) + c_copy[i])) + (1 - u_id) * (
                                          diff * sigmoid(k) * (theta_copy[i] - beta_copy[j]) / (
                                              c_copy[i] * np.exp(k) + d_copy[i]))
        beta[j] += lr * b
        alpha[j] += lr * alpha_grad

    return theta, beta, alpha, c, d


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(len(data["user_id"]))
    beta = np.zeros(len(data["question_id"]))
    alpha = np.ones(len(data["question_id"]))
    c = np.zeros(len(data["user_id"])) + 0.000001
    d = np.ones(len(data["user_id"])) - 0.000001

    val_acc_lst = []
    train_lklihood = []
    val_lklihood = []

    for i in range(iterations):

        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha, c=c, d=d)
        train_lklihood.append(neg_lld)
        ngg_lld_val = neg_log_likelihood(val_data, theta, beta, alpha, c, d)
        val_lklihood.append(ngg_lld_val)

        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, c=c, d=d)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha, c, d = update_theta_beta(data, lr, theta, beta, alpha, c, d)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lklihood, val_lklihood


def evaluate(data, theta, beta, alpha, c, d):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = ((theta[u] - beta[q]) * alpha[q]).sum()
        p_a = c[u] + (d[u] - c[u]) * sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    iterations = [15]
    lr = [0.01]
    # iterations_lst = list(range(iterations))
    acc_lst = []
    vac_acc_lst = []
    train_llh_lst = []
    val_llh_lst = []
    hp_lst = []
    tb_lst = []
    for it in iterations:
        for lrt in lr:
            theta, beta, val_acclst, train_llh, val_llh = irt(train_data, val_data, lrt, it)
            vac_acc_lst.append(val_acclst)
            train_llh_lst.append(train_llh)
            val_llh_lst.append(val_llh)
            acc_lst.append(evaluate(val_data, theta, beta))
            hp_lst.append([lrt, it])
            tb_lst.append([theta, beta])
    max_index = np.argmax(acc_lst)
    max_hp = hp_lst[max_index]
    iterations_lst = list(range(max_hp[1]))
    max_train_llh = train_llh_lst[max_index]
    max_val_llh = val_llh_lst[max_index]
    max_val_acclst = vac_acc_lst[max_index]
    max_tb = tb_lst[max_index]
    print("The hyperparameters with highest validation accuracy is learning rate:{}, iterations:{}".format(max_hp[0], max_hp[1]))


if __name__ == "__main__":
    main()



