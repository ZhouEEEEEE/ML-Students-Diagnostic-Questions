from utils import *
import matplotlib.pyplot as plt
import numpy as np

# This is an implementation of 3-pl IRT model


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta, c, alpha):

    log_lklihood = 0.

    for i in range(len(data["user_id"])):
        u = data["user_id"][i]
        q = data["question_id"][i]
        cij = data["is_correct"][i]

        log_lklihood += np.log(1-c) - np.log(1 + np.exp(alpha[q] * (theta[u]-beta[q]))) - cij*np.log(1-c) + cij*np.log(c + np.exp(alpha[q] * (theta[u]-beta[q])))
    return -log_lklihood


def update(lr, theta, beta, c, alpha, answer_matrix, data_matrix):

    cur_t = theta.copy()
    cur_b = beta.copy()
    cur_c = c
    cur_a = alpha.copy()

    tb_diff = np.subtract.outer(cur_t[:, 0], cur_b[:, 0]) * np.exp((np.subtract.outer(cur_t[:, 0], cur_b[:, 0]).T * cur_a).T)
    c_diff = cur_c + np.exp((np.subtract.outer(cur_t[:, 0], cur_b[:, 0]).T * cur_a).T)
    alpha_diff = (np.exp((np.subtract.outer(cur_t[:, 0], cur_b[:, 0]).T * cur_a).T).T * cur_a).T

    d_theta = answer_matrix * (alpha_diff / c_diff) - alpha_diff / (1 + np.exp((np.subtract.outer(cur_t[:, 0], cur_b[:, 0]).T * cur_a).T))
    d_beta = np.dot((-1 * d_theta * data_matrix).T, np.ones((542, 1)))
    d_theta = np.dot(d_theta * data_matrix, np.ones((1774, 1)))
    d_alpha = np.dot(((-1 * tb_diff / (1 + np.exp((np.subtract.outer(cur_t[:, 0], cur_b[:, 0]).T * cur_a).T)) + answer_matrix * tb_diff / c_diff) * data_matrix).T,
                 np.ones((542, 1)))

    theta += lr * d_theta
    beta += lr * d_beta
    alpha = alpha + lr * d_alpha

    return theta, beta, c, alpha


def evaluate(data, theta, beta, c, alpha):
    # Plz implement evaluate, it is same as evaluate in item_response file by sim
    # ply adding two more parameter c and alpha

    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = c + (1 - c) * sigmoid(alpha[q] * x)
        pred.append(p_a >= 0.5)

    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def irt(data, val_data, lr, iterations, answer_matrix, data_matrix):

    theta = np.full((542, 1), 0.5)
    beta = np.full((1774, 1), 0.5)
    c = 0.25
    alpha = np.full((1774, 1), 1)
    train_llh = []
    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta[:, 0], beta=beta[:, 0], c=c, alpha=alpha[:, 0])
        train_llh.append(neg_lld)
        score = evaluate(data=val_data, theta=theta[:, 0], beta=beta[:, 0], c=c, alpha=alpha[:, 0])
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, c, alpha = update(lr, theta, beta, c, alpha, answer_matrix, data_matrix)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, c, alpha, val_acc_lst, train_llh


def get_matrix(data):
    users = data["user_id"]
    questions = data["question_id"]
    cij_lst = data["is_correct"]
    answer_matrix = np.zeros((542, 1774))
    data_matrix = np.zeros((542, 1774))
    for i in range(len(users)):
        u_id = users[i]
        q_id = questions[i]
        cij = cij_lst[i]
        answer_matrix[u_id, q_id] = cij
        data_matrix[u_id, q_id] = 1
    return answer_matrix, data_matrix


def main():
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    answer_matrix, data_matrix = get_matrix(train_data)
    # Using the same hyperparameter as before
    # theta, beta, c, alpha, val_acclst, train_llh = irt(train_data, val_data, 0.01, 200, answer_matrix, data_matrix)
    #
    # plt.plot(list(range(200)), train_llh, "r-", label="training")
    # plt.xlabel("iterations")
    # plt.ylabel("negative likelihood")
    # plt.legend()
    # plt.show()
    #
    # val_acc = evaluate(val_data, theta[:, 0], beta[:, 0], c, alpha[:, 0])
    # test_acc = evaluate(test_data, theta[:, 0], beta[:, 0], c, alpha[:, 0])
    # print("Validation accuracy: {}".format(val_acc))
    # print("Test Accuracy: {}".format(test_acc))

    # Doing another hyperparameter tuning for the algorith it self
    iterations = [20]
    lr = [0.01]

    acc_lst = []
    vac_acc_lst = []
    train_llh_lst = []
    hp_lst = []
    tb_lst = []
    c_lst = []
    alpha_lst = []
    for it in iterations:
        for lrt in lr:
            theta, beta, c, alpha, val_acclst, train_llh = irt(train_data, val_data, lrt, it, answer_matrix, data_matrix)
            vac_acc_lst.append(val_acclst)
            train_llh_lst.append(train_llh)
            acc_lst.append(evaluate(val_data, theta[:, 0], beta[:, 0], c, alpha[:, 0]))
            hp_lst.append([lrt, it])
            tb_lst.append([theta, beta])
            c_lst.append(c)
            alpha_lst.append(alpha)
    max_index = np.argmax(acc_lst)
    max_hp = hp_lst[max_index]
    iterations_lst = list(range(max_hp[1]))
    max_train_llh = train_llh_lst[max_index]
    max_val_acclst = vac_acc_lst[max_index]
    max_tb = tb_lst[max_index]
    max_c = c_lst[max_index]
    max_alpha = alpha_lst[max_index]
    print("The hyperparameters with highest validation accuracy is learning rate:{}, iterations:{}".format(max_hp[0], max_hp[1]))

    plt.plot(iterations_lst, max_train_llh, "r-", label="training")
    plt.xlabel("iterations")
    plt.ylabel("negative likelihood")
    plt.legend()
    plt.show()

    plt.plot(iterations_lst, max_val_acclst)
    plt.xlabel("iterations")
    plt.ylabel("validation accuracy")
    plt.show()

    val_acc = evaluate(val_data, max_tb[0][:, 0], max_tb[1][:, 0], max_c, max_alpha[:, 0])
    test_acc = evaluate(test_data, max_tb[0][:, 0], max_tb[1][:, 0], max_c, max_alpha[:, 0])
    print("Validation accuracy: {}".format(val_acc))
    print("Test Accuracy: {}".format(test_acc))

    theta = np.sort(max_tb[0])

    # x = np.polyfit(theta, sigmoid(theta - max_tb[1][150]), 1)
    # coeff1 = x[0]
    # inter1 = x[1]

    plt.plot(theta, sigmoid(theta - max_tb[1][150]), label=f"question 1")
    # plt.plot(theta, coeff1*theta + inter1)
    # plt.plot(theta, sigmoid(theta - max_tb[1][300]), label=f"question 2")
    # plt.plot(theta, sigmoid(theta - max_tb[1][450]), label=f"question 3")

    plt.title("probability of the correctly responsed the question vs theta")
    plt.xlabel("theta")
    plt.ylabel("probability")
    # plt.legend()
    plt.show()




if __name__ == "__main__":
    main()
