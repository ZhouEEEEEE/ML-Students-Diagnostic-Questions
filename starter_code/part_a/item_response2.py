from utils import *
import matplotlib.pyplot as plt
import numpy as np

# This is an implementation of 3-pl IRT model


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x + 0.001) / (1 + np.exp(x + 0.001))


def neg_log_likelihood(data, theta, beta, alpha, c):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    # log_lklihood = 0.
    # for i in range(len(data["user_id"])):
    #     theta_i = theta[data["user_id"][i]]
    #     beta_j = beta[data["question_id"][i]]
    #     a = alpha[data["question_id"][i]]
    #     sig = sigmoid(a * (theta_i - beta_j))
    #     cij = data['is_correct'][i]
    #     ci = c[data["user_id"][i]]
    #     # di = d[data["user_id"][i]]
    #     log_lklihood += (cij * np.log(ci + (1-ci)*sig) + (1-cij) * np.log(1-(ci + (1-ci)*sig)))  # May be modified

    # log_lklihood = 0.
    # for idx, u_id in enumerate(data["user_id"]):
    #     q_id = data["question_id"][idx]
    #
    #     ex = np.exp(alpha[q_id] * (theta[u_id] - beta[q_id]))
    #
    #     if data["is_correct"][idx] == 0:
    #         if np.isnan(-(((1 - c[u_id]) * ex) / (ex + 1)) - c[u_id] + 1):
    #             print("A")
    #         log_lklihood += np.log(-(((1 - c[u_id]) * ex) / (ex + 1)) - c[u_id] + 1)
    #
    #
    #     elif data["is_correct"][idx] == 1:
    #         if np.isnan(np.log(((1 - c[u_id]) * ex / (ex + 1)) + c[u_id])):
    #             print("A")
    #         log_lklihood += np.log(((1- c[u_id]) * ex / (ex + 1)) + c[u_id])
    #
    # return -log_lklihood
    #
    log_lklihood = 0
    for i, cr in enumerate(data['is_correct']):
        curr_theta = theta[data['user_id'][i]]
        a = alpha[data['question_id'][i]]
        curr_beta = beta[data['question_id'][i]]
        ci = c[data['question_id'][i]]
        sig = sigmoid(a * (curr_theta - curr_beta))
        log_lklihood += (cr * np.log(ci + (1 - ci) * sig) + (1 - cr) * np.log(1 - (ci + (1 - ci) * sig)))
    return -log_lklihood
    # log_lklihood = 0.
    #
    # for i in range(len(data["user_id"])):
    #   u = data["user_id"][i]
    #   q = data["question_id"][i]
    #   c = data["is_correct"][i]
    #
    #   log_lklihood += np.log(1-c) - np.log(1 + np.exp(alpha[q] * (theta[u]-beta[q]))) - c*np.log(1-c) + c*np.log(c + np.exp(alpha[q] * (theta[u]-beta[q])))

    # return -log_lklihood


# def sparse_matrix(data):
#     matrix = np.empty(shape=(542, 1774))
#     matrix[:] = np.NaN
#     for i, q in enumerate(data["question_id"]):
#         u = data["user_id"][i]
#         mark = data["is_correct"][i]
#         matrix[u, q] = mark
#     return matrix


# def update_theta_beta(data, lr, theta, beta, alpha, c, d):
#
#     theta_copy = theta.copy()
#     beta_copy = beta.copy()
#     alpha_copy = alpha.copy()
#     c_copy = c.copy()
#     d_copy = d.copy()
#
#     train_matrix = sparse_matrix(data)
#
#     for i in range(len(theta)):
#         c_decay = []
#         d_decay = []
#         t = 0.
#         for j, q_id in enumerate(train_matrix[i, :]):
#             if np.isnan(q_id):
#                 continue
#             else:
#                 k = alpha_copy[j] * (theta_copy[i] - beta_copy[j])
#                 diff = alpha_copy[j] * (d_copy[i] - c_copy[i])
#                 t += q_id * (diff * sigmoid(k) / (d_copy[i] * np.exp(k) + c_copy[i])) + (1 - q_id) * (-diff * sigmoid(k) / (c_copy[i] * np.exp(k) + d_copy[i]))
#                 c_decay.append(q_id * (1 / (c_copy[i] + d_copy[i] * np.exp(k))) + (1 - q_id) * (np.exp(k) / (d_copy[i] + c_copy[i] * np.exp(k))))
#                 d_decay.append(q_id * (-np.exp(k) / (c_copy[i] + d_copy[i] * np.exp(k))) + (1 - q_id) * (-1 / (d_copy[i] + c_copy[i] * np.exp(k))))
#
#         theta[i] += lr * t
#         c[i] += lr * np.mean(c_copy)
#         d[i] += lr * np.mean(d_copy)
#
#     for j in range(len(beta)):
#         b = 0.
#         alpha_grad = 0.
#         for i, u_id in enumerate(train_matrix[:, j]):
#             if np.isnan(u_id):
#                 continue
#             else:
#                 k = alpha_copy[j] * (theta_copy[i] - beta_copy[j])
#                 diff = (d_copy[i] - c_copy[i]) * alpha_copy[j]
#                 b += u_id * (-diff * sigmoid(k) / (d_copy[i] * np.exp(k) + c_copy[i])) + (1 - u_id) * (
#                             diff * sigmoid(k) / (c_copy[i] * np.exp(k) + d_copy[i]))
#                 alpha_grad += u_id * (-diff * sigmoid(k) * (theta_copy[i] - beta_copy[j]) / (
#                             d_copy[i] * np.exp(k) + c_copy[i])) + (1 - u_id) * (
#                                           diff * sigmoid(k) * (theta_copy[i] - beta_copy[j]) / (
#                                               c_copy[i] * np.exp(k) + d_copy[i]))
#         beta[j] += lr * b
#         alpha[j] += lr * alpha_grad
#
#     return theta, beta, alpha, c, d


def update_theta_beta(data, lr, theta, beta, a, c):
    theta_deriv = np.zeros(theta.shape[0])
    a_deriv = np.zeros(a.shape[0])
    beta_deriv = np.zeros(beta.shape[0])
    c_deriv = np.zeros(c.shape[0])

    for i, correct in enumerate(data['is_correct']):
        curr_theta = theta[data['user_id'][i]]
        curr_a = a[data['question_id'][i]]
        curr_beta = beta[data['question_id'][i]]
        curr_c = c[data['question_id'][i]]

        diff = curr_theta - curr_beta
        theta_deriv[data['user_id'][i]] += (correct * curr_a - curr_a * sigmoid(curr_a * diff))
        a_deriv[data['question_id'][i]] += (diff*np.exp(curr_a*diff)) * ((correct-1)*np.exp(curr_a*diff)-curr_c+correct)/(np.exp(curr_a*diff) + 1)*(np.exp(curr_a*diff)+curr_c)

        beta_deriv[data['question_id'][i]] += (curr_a * sigmoid(curr_a * diff) - correct * curr_a)
        # c_deriv[data['question_id'][i]] += -((correct * diff * np.exp(curr_a*diff))/(curr_c + np.exp(curr_a*diff))**2)
        # if a_deriv[data['user_id'][i]] * lr >= 1:
        #     a_deriv[data['user_id'][i]] = 0.9/lr
        # if a_deriv[data['user_id'][i]] * lr <= 0:
        #     a_deriv[data['user_id'][i]] = 0.01/lr

    theta += (lr * theta_deriv)
    a += (lr * a_deriv)
    beta += (lr * beta_deriv)
    # c = calculate_t1t2(c, c_deriv, lr)
    return theta, beta, a, c


# def calculate_t1t2(original, grads, lr):
#     for i, row in enumerate(grads):
#         row_sum = np.sum(row)
#         if original[i] - lr*row_sum < 0:
#             original[i] = 0.001
#         elif original[i] - lr*row_sum > 1:
#             original[i] = 0.999
#         else:
#             original[i] += lr*row_sum
#     return original


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
    c = np.zeros(len(data["question_id"])) + 0.25
    # d = np.ones(len(data["user_id"])) - 0.000001

    val_acc_lst = []
    train_lklihood = []
    val_lklihood = []

    for i in range(iterations):

        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, alpha=alpha, c=c)
        train_lklihood.append(neg_lld)
        ngg_lld_val = neg_log_likelihood(val_data, theta, beta, alpha, c)
        val_lklihood.append(ngg_lld_val)

        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, c=c)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, alpha, c = update_theta_beta(data, lr, theta, beta, alpha, c)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, alpha, c, val_acc_lst, train_lklihood, val_lklihood


def evaluate(data, theta, beta, alpha, c):
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
        p_a = c[u] + (1 - c[u]) * sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")


    # iterations = [20, 50, 100, 200]
    iterations = [30]
    # lr = [0.1, 0.05, 0.01]
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
            theta, beta, alpha, c, val_acclst, train_llh, val_llh = irt(train_data, val_data, lrt, it)
            vac_acc_lst.append(val_acclst)
            train_llh_lst.append(train_llh)
            val_llh_lst.append(val_llh)
            acc_lst.append(evaluate(val_data, theta, beta, alpha, c))
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

    # plt.plot(iterations_lst, max_train_llh, "r-", label="training")
    # plt.plot(iterations_lst, max_val_llh, "b-", label="validation")
    # plt.xlabel("iterations")
    # plt.ylabel("negative likelihood")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(iterations_lst, max_val_acclst)
    # plt.xlabel("iterations")
    # plt.ylabel("validation accuracy")
    # plt.show()
    #
    # # Implement part (c)
    # val_acc = evaluate(val_data, max_tb[0], max_tb[1])
    # test_acc = evaluate(test_data, max_tb[0], max_tb[1])
    # print("Validation accuracy: {}".format(val_acc))
    # print("Test accuracy: {}".format(test_acc))
    #
    # # Implement part (d)
    # theta = np.sort(max_tb[0])
    #
    # plt.plot(theta, sigmoid(theta - max_tb[1][150]), label=f"question 1")
    # plt.plot(theta, sigmoid(theta - max_tb[1][300]), label=f"question 2")
    # plt.plot(theta, sigmoid(theta - max_tb[1][450]), label=f"question 3")
    #
    # plt.title("probability of the correctly responsed the question vs theta")
    # plt.xlabel("theta")
    # plt.ylabel("probability")
    # plt.legend()
    # plt.show()




if __name__ == "__main__":
    main()
