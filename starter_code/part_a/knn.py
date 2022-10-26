import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from utils import *
import numpy as np


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    print("Validation Accuracy on question: {}".format(acc))
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    lst = [1, 6, 11, 16, 21, 26]

    # Question 1 a
    acc_lst1 = []
    for k in lst:
        acc = knn_impute_by_user(sparse_matrix, val_data, k)
        acc_lst1.append(acc)
        print("The accuracy of k = {}, is {}".format(k, acc))

    # Question 1 b
    k_max = lst[np.argmax(acc_lst1)]
    print("The k value with max accuracy on users is {}".format(k_max))
    plt.scatter(lst, acc_lst1)
    plt.xlabel("k")
    plt.ylabel("accuracy on users")
    plt.show()
    test_acc = knn_impute_by_user(sparse_matrix, test_data, k_max)
    print("The accuracy of test_set on users is {}".format(test_acc))

    # Question 1 c
    acc_lst = []
    for k in lst:
        acc = knn_impute_by_item(sparse_matrix, val_data, k)
        acc_lst.append(acc)
        print("The accuracy of k = {}, is {}".format(k, acc))
    k_max = lst[np.argmax(acc_lst)]
    print("The k value with max accuracy is {}".format(k_max))
    plt.scatter(lst, acc_lst)
    plt.xlabel("k")
    plt.ylabel("accuracy on questions")
    plt.show()
    test_acc = knn_impute_by_item(sparse_matrix, test_data, k_max)
    print("The accuracy of test_set on questions is {}".format(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

if __name__ == "__main__":
    main()
