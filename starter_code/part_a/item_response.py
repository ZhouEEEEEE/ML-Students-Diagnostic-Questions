from utils import *
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
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
        sig = sigmoid(theta_i - beta_j)
        cij = data['is_correct'][i]
        log_lklihood += (cij * np.log(sig) + (1-cij) * np.log(1-sig))
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    t = np.zeros(theta.shape)
    b = np.zeros(beta.shape)
    for i in range(len(data["user_id"])):
        theta_i = theta[data["user_id"][i]]
        beta_j = beta[data["question_id"][i]]
        cij = data['is_correct'][i]
        t[data["user_id"][i]] -= cij - (np.exp(theta_i) / (np.exp(theta_i)+np.exp(beta_j)))
        b[data["question_id"][i]] -= (np.exp(theta_i) / (np.exp(theta_i)+np.exp(beta_j))) - cij

    theta = theta - t * lr
    beta = beta - b * lr
    return theta, beta


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

    val_acc_lst = []
    train_lklihood = []
    val_lklihood = []

    for i in range(iterations):

        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        train_lklihood.append(neg_lld)
        ngg_lld_val = neg_log_likelihood(val_data, theta, beta)
        val_lklihood.append(ngg_lld_val)

        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_lklihood, val_lklihood


def evaluate(data, theta, beta):
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
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")


    iterations = [200]
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

    plt.plot(iterations_lst, max_train_llh, "r-", label="training")
    # plt.plot(iterations_lst, max_val_llh, "b-", label="validation")
    plt.xlabel("iterations")
    plt.ylabel("negative likelihood")
    plt.legend()
    plt.show()

    plt.plot(iterations_lst, max_val_acclst)
    plt.xlabel("iterations")
    plt.ylabel("validation accuracy")
    plt.show()

    # Implement part (c)
    val_acc = evaluate(val_data, max_tb[0], max_tb[1])
    test_acc = evaluate(test_data, max_tb[0], max_tb[1])
    print("Validation accuracy: {}".format(val_acc))
    print("Test accuracy: {}".format(test_acc))

    # Implement part (d)
    theta = np.sort(max_tb[0])

    plt.plot(theta, sigmoid(theta - max_tb[1][150]), label=f"question 1")
    plt.plot(theta, sigmoid(theta - max_tb[1][300]), label=f"question 2")
    plt.plot(theta, sigmoid(theta - max_tb[1][450]), label=f"question 3")

    plt.title("probability of the correctly responsed the question vs theta")
    plt.xlabel("theta")
    plt.ylabel("probability")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
