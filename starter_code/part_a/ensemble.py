# TODO: complete this file.
# Pick knn test on users
# Pick IRT
# Pick Neural networks
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import part_a.item_response as ir
import part_a.knn as knn
import part_a.neural_network as nn
from sklearn.utils import resample


def boot_strap(data):
    size = len(data["question_id"])
    res = {"user_id": [],  "question_id": [], "is_correct": []}
    for i in np.random.choice(size, int(np.floor(size * 1))):
        res["user_id"].append(data["user_id"][i])
        res["question_id"].append(data["question_id"][i])
        res["is_correct"].append(data["is_correct"][i])
    return res
#
#
# def knn_te(matrix, data):
#     return knn.knn_impute_by_user(matrix, data, 11)
#
#
# def knn_te1(matrix, data):
#     return knn.knn_impute_by_item(matrix, data, 11)
#


def irt_te(data, val_data):
    theta, beta, val_acc_lst, train_lklihood, val_lklihood = ir.irt(data, val_data, 0.01, 20) # 0.001, 100
    return theta, beta, val_acc_lst, train_lklihood, val_lklihood


def prediction(data, theta, beta):
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = ir.sigmoid(x)
        pred.append(1) if p_a >= 0.5 else pred.append(0)
    return pred

# def prediction(data, theta, beta):
#     prob = []
#     for i, q in enumerate(data["question_id"]):
#         u = data["user_id"][i]
#         x = (theta[u] - beta[q]).sum()
#         p_a = ir.sigmoid(x)
#         prob.append(p_a)
#     t = []
#     for i in np.arange(len(prob)):
#         t.append(1) if prob[i] >= 0.5 else t.append(0)
#     return t
#
#
# def f(lst):
#     # return lst
#     t = []
#     for i in np.arange(len(lst)):
#         t.append(1) if lst[i] >= 0.5 else t.append(0)
#     return t

def evaluation(data, lst1, lst2, lst3):
    vote = []
    for i in range(len(lst1)):
        value = (lst1[i] + lst2[i] + lst3[i]) / 3
        vote.append(value >= 0.5)

    acc = np.sum(data['is_correct'] == np.array(vote))/len(data['is_correct'])
    return acc
#
# def nn_te(num_question, train_matrix, zero_train_matrix, valid_data, test_data):
#     model = nn.AutoEncoder(num_question, 10)
#     val_acc_list, _ = nn.train(model, 0.05, 0.001, train_matrix, zero_train_matrix, valid_data, 20)
#
#     return max(val_acc_list), nn.evaluate(model, zero_train_matrix, test_data)


def main():
    # zero_train_matrix, train_matrix, _, _ = nn.load_data()  # Not resampled
    # num_question = train_matrix.shape[1]

    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Start Training 3 IRT models")
    theta1, beta1, _, _, _ = irt_te(boot_strap(train_data), val_data)
    theta2, beta2, _, _, _ = irt_te(boot_strap(train_data), val_data)
    theta3, beta3, _, _, _ = irt_te(boot_strap(train_data), val_data)

    print("Start Predicting validation set by 3 IRT models")
    pre_val1 = prediction(val_data, theta1, beta1)
    pre_val2 = prediction(val_data, theta2, beta2)
    pre_val3 = prediction(val_data, theta3, beta3)

    print("Validation accuracy is {}".format(evaluation(val_data, pre_val1, pre_val2, pre_val3)))
    print("Start Predicting testing set by 3 IRT models")
    pre_test1 = prediction(test_data, theta1, beta1)
    pre_test2 = prediction(test_data, theta2, beta2)
    pre_test3 = prediction(test_data, theta3, beta3)

    print("Validation accuracy is {}".format(evaluation(test_data, pre_test1, pre_test2, pre_test3)))

    # val_acc1 = ir.evaluate(val_data, theta, beta)
    # test_acc1 = ir.evaluate(test_data, theta, beta)
    # print("nn validation accuracy is {}".format(val_acc1))
    # print("nn testing accuracy is {}".format(test_acc1))
    print("End of Evaluating 3 IRT models")
    print("===========================================")
    #
    # print("Start Training user based KNN model")
    # val_acc2 = knn_te(resample(sparse_matrix), val_data)
    # test_acc2 = knn_te(resample(sparse_matrix), test_data)
    # print("nn validation accuracy is {}".format(val_acc2))
    # print("nn testing accuracy is {}".format(test_acc2))
    # print("End of Evaluating user based KNN model")
    # print("===========================================")
    #
    # # print("Start Training neural network")
    # # val_acc3, test_acc3 = nn_te(num_question, train_matrix, zero_train_matrix, val_data, test_data)
    # # print("nn validation accuracy is {}".format(val_acc3))
    # # print("nn testing accuracy is {}".format(test_acc3))
    # # print("End of Evaluating neural network model")
    # print("Start Training user based KNN model")
    # val_acc3 = knn_te1(resample(sparse_matrix), val_data)
    # test_acc3 = knn_te1(resample(sparse_matrix), test_data)
    # print("nn validation accuracy is {}".format(val_acc3))
    # print("nn testing accuracy is {}".format(test_acc3))
    # print("End of Evaluating KNN model")
    # print("===========================================")

    # print("Average validation accuracy is {}".format((val_acc1 + val_acc2 + val_acc3)/3))
    # print("Average testing accuracy is {}".format((test_acc1 + test_acc2 + test_acc3)/3))


if __name__ == "__main__":
    main()



