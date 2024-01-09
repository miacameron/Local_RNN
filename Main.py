import numpy as np
from matplotlib import pyplot as plt
from RNN_class import *
from helper import *
import pickle

import argparse
import sys
import os
import shutil

"""
N = 200
T = 100
hidden_N = 200
bptt_depth = 100
print_freq = 1000
epochs = 20000
lr = 0.01
"""

parser = argparse.ArgumentParser(description='Numpy BPTT Training')
parser.add_argument('--epochs', default=40000, type=int, metavar='N',help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-n', '--n', default=200, type=int, help='Input/output size')
parser.add_argument('--hidden-n', default=200, type=int, help='Hidden dimension size')
parser.add_argument('--savename', default='', type = str, help='output saving name')
parser.add_argument('-t','--total_steps', default=100, type=int, help='Total steps per traversal')
parser.add_argument('-p', '--print_freq', default=1000, type=int,metavar='N', help='print frequency (default: 1000)')
parser.add_argument('--learning_alg', default='bptt', type=str, help='learning algorithm (default: bptt)')


def main():

    global args

    args = parser.parse_args()
    lr = args.lr
    epochs = args.epochs
    N = args.n
    hidden_N = args.hidden_n
    T = args.total_steps
    learning_alg = args.learning_alg
    print_freq = args.print_freq
    savename = args.savename

    # Creating Gaussian inputs
    inputs = create_inputs(N, T+1, T, sigma=15)
    y_mini = np.zeros((N, T))
    y_mini = inputs

    # Init. first hidden state as zeros
    h0 = np.zeros((hidden_N))

    net, loss_list, grad_list, hidden_rep, output_rep = train_partial(y_mini, h0, epochs, lr, learning_alg)

    param_dict = {  "net" : net, 
                    "loss_list" : loss_list, 
                    "grad_list" : grad_list, 
                    "hidden_rep" : hidden_rep, 
                    "output_rep" : output_rep}
    
    f = open(savename, 'ab')
    pickle.dump(param_dict, f)
    f.close()


def train_partial(Y_mini, h0, n_epochs, lr, learning_alg):
    '''
    Fix the intermediate recording of predloss (10/26/2021 Y.C.)
    Note first time-step info is not used (07/08/2021,Y.C.)
    Add stop criteria (07/11/2021, Y.C.)
        INPUT:
            Y_mini: seqN*featureN
            n_epochs: number of epoches to train
            net: pre-defined network structure
            RecordEp: the recording and printing frequency
        OUTPUT:
            net:
            loss_list:
            y_hat: SeqN*HN
    ''' 

    loss_list = []
    N,T = Y_mini.shape
    hidden_N = h0.shape[0]
    epoch = 0

    hidden_rep = []
    output_rep = []
    dLdW_list = []
    dLdV_list = []
    dLdU_list = []
    dLdb_list = []
    dLdc_list = []

    if (learning_alg == "bptt"):
        net = BPTTRNN(N,hidden_N,T)
    elif (learning_alg == "local"):
        net = LocalRNN(N,hidden_N,T)
    else:
        raise NotImplementedError

    while epoch < n_epochs:
        y = Y_mini
        # not optimized - doing forward prop twice
        h_seq, y_hat, L = net.forward_propagation(y,h0)
        dLdV, dLdW, dLdU, dLdb, dLdc = net.gradient(y,h0)
        loss_list.append(L)
        dLdW_list.append(dLdW)
        dLdV_list.append(dLdV)
        dLdU_list.append(dLdU)
        dLdb_list.append(dLdb)
        dLdc_list.append(dLdc)
        net.update_weights(dLdV, dLdW, dLdU, dLdb, dLdc, lr) # Updates the weights accordingly
        epoch += 1
        if epoch%args.print_freq == 0:
            hidden_rep.append(h_seq)
            output_rep.append(y_hat)
            print('Epoch: {}/{}.............'.format(epoch,n_epochs), end=' ')
            print("Loss: {:.4f}".format(L))

    return net, loss_list, [dLdW_list, dLdV_list, dLdU_list, dLdb_list, dLdc_list], hidden_rep, output_rep


if __name__ == '__main__':
    main()
