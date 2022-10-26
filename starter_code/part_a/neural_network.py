from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import matplotlib.pyplot as plt

import numpy as np
import torch


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight, 2) ** 2
        h_w_norm = torch.norm(self.h.weight, 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        input1 = self.g(inputs)
        sig_input1 = torch.sigmoid(input1)
        input2 = self.h(sig_input1)
        out = torch.sigmoid(input2)
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    train_loss_lst = []
    val_acc_lst = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)  # objective
            loss += 0.5 * lamb * (model.get_weight_norm())  # regularization

            loss.backward()

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
        val_acc_lst.append(valid_acc)
        train_loss_lst.append(train_loss)

    return val_acc_lst, train_loss_lst
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    num_question = train_matrix.shape[1]
    # Set model hyperparameters.
    k_list = [10, 50, 100, 200, 500]

    # num_student = train_matrix.shape[0]

    # Set optimization hyperparameters.
    lr_list = [0.01, 0.05, 0.1]
    num_epoch_list = [20, 40, 60]
    # lr_list = [0.01, 0.1]
    # num_epoch_list = [3, 6]
    lamb = 0
    # lamb_list = [0.001, 0.01, 0.1, 1]

    # 3c)
    result = []
    for k in k_list:
        for lr in lr_list:
            for ep in num_epoch_list:
                model = AutoEncoder(num_question, k)
                val_acc_list, _ = train(model, lr, lamb, train_matrix, zero_train_matrix, valid_data, ep)
                result.append([max(val_acc_list)*10000, ep, np.argmax(val_acc_list), lr, k])
    result_ay = np.array(result)
    max_acc_index = np.unravel_index(np.argmax(result_ay), result_ay.shape)
    max_acc = result[max_acc_index[0]][0]/10000
    max_ep = result[max_acc_index[0]][1]
    at_ep = result[max_acc_index[0]][2]
    max_lr = result[max_acc_index[0]][3]
    max_k = result[max_acc_index[0]][4]
    print("The max validation accuracy is {}, which at the {}th epoch with total {} epoch, learning rate {}, and k value {}".format(max_acc, at_ep, max_ep, max_lr, max_k))

    # 3d)
    model = AutoEncoder(num_question, max_k)
    val_acc_list, train_loss_lst = train(model, max_lr, lamb, train_matrix, zero_train_matrix, valid_data, max_ep)
    test_acc = evaluate(model, zero_train_matrix, test_data)
    plt.figure(1)
    plt.plot(list(range(max_ep)), train_loss_lst, label="training lost")
    plt.xlabel("epoch")
    plt.ylabel("training loss")

    plt.figure(2)
    plt.plot(list(range(max_ep)), val_acc_list, label="training lost")
    plt.xlabel("epoch")
    plt.ylabel("validation accuracy")

    plt.tight_layout()
    plt.show()
    print("test accuracy is {}".format(test_acc))

    # 3e)
    lamb_list = [0.001, 0.01, 0.1, 1]
    acc = []
    model_lst = []
    for lam in lamb_list:
        model = AutoEncoder(num_question, max_k)
        val_acc_list, _ = train(model, max_lr, lam, train_matrix, zero_train_matrix, valid_data, max_ep)
        acc.append(max(val_acc_list))
    max_lam = lamb_list[np.argmax(acc)]
    print("The max validation accuracy is {} exists when lambda is {}".format(max(acc), max_lam))
    test_acc1 = evaluate(model_lst[np.argmax(acc)], zero_train_matrix, test_data)
    print("test accuracy is {}".format(test_acc1))



    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
