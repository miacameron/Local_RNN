# Set of local and non-local RNN classes defined M.C. based on work by Y.C
# No PyTorch

import numpy as np
from scipy.stats import norm
from scipy.special import softmax


class BaseRNN:
    """
    Base RNN: numpy based
    tanh and softmax nonlinearity
    """

    def __init__(self, N, hidden_N, bptt_depth):
        self.N = N
        self.hidden_N = hidden_N
        self.bptt_depth = bptt_depth  # number of time-steps
        self.U = np.random.uniform(-0.1, 0.1, size=(hidden_N, N))
        self.W = np.random.uniform(-0.1, 0.1, size=(hidden_N, hidden_N))  # recurrent weight matrix
        self.V = np.random.uniform(-0.1, 0.1, size=(N, hidden_N))  # output weight matrix
        self.b = np.random.uniform(-0.1, 0.1, size=(hidden_N))
        self.c = np.random.uniform(-0.1, 0.1, size=(N))

    def forward_propagation(self, y, h0):
        """
        X	  : np array of size (N, bptt_depth) with the input sequence
        h0	 : np array of size (hidden_N) with starting state of h
        """
        h = np.zeros((self.hidden_N, np.int(self.bptt_depth)))
        # initialize hidden state
        h[:, 0] = h0

        y_hat = np.zeros((self.N, np.int(self.bptt_depth)))
        y_hat[:, 0] = y[:, 0]  # first element is free

        for i in np.arange(1, np.int(self.bptt_depth)):  # loop from 1:latest
            u = self.W @ h[:, i - 1] + self.b + (self.U @ y[:, i-1])
            h[:, i] = np.tanh(u)
            o = self.V @ h[:, i] + self.c
            y_hat[:, i] = softmax(o)
        delta = y_hat[1::] - y[1::]  # loss vector
        L = np.sum(np.power(delta, 2))  # MSE loss function (scalar)
        return h, y_hat, L

    def softmax_jacobian(self, x):
        J = -1 * np.outer(x, x)
        J += np.diag(x)
        return J

    def gradient(self, y, h0):
        raise NotImplementedError

    def update_weights(self, dLdV, dLdW, dLdU, dLdb, dLdc, lr=0.05):
        self.V += -lr * dLdV
        self.W += -lr * dLdW
        self.U += -lr * dLdU
        self.b += -lr * dLdb
        self.c += -lr * dLdc
        return


class BPTTRNN(BaseRNN):
    """
    BPTT RNN: numpy based
    Inherits from BaseRNN
    """
    
    def __init__(self, N, hidden_N, bptt_depth):
        super().__init__(N, hidden_N, bptt_depth)

    def gradient(self, y, h0):
        h, y_hat, L = self.forward_propagation(y, h0)
        delta = y_hat - y
        L = np.sum(np.power(delta,2))

        # initialize jacobians used in error computation
        dLdU = np.zeros((self.U.shape[0], self.U.shape[1], np.int(self.bptt_depth)))
        dLdV = np.zeros((self.V.shape[0], self.V.shape[1], np.int(self.bptt_depth)))
        dLdh = np.zeros((h.shape[0], np.int(self.bptt_depth)))
        dLdW = np.zeros((self.W.shape[0], self.W.shape[1], np.int(self.bptt_depth)))
        dLdb = np.zeros((self.b.shape[0], np.int(self.bptt_depth)))
        dLdc = np.zeros((self.c.shape[0], np.int(self.bptt_depth)))

        # last element
        dLdo = self.softmax_jacobian(y_hat[:, -1]) @ delta[:, -1]
        dLdV[:, :, -1] = np.outer(dLdo, h[:, -1])
        dLdh[:, -1] = self.V.T @ dLdo
        dLdW[:, :, -1] = np.outer(np.diag(1 - np.power(h[:, -1], 2)) @ dLdh[:, -1], h[:, -2])
        dLdU[:,:,-1] = np.outer(dLdh[:,-1], y[:,-1])
        dLdb[:, -1] = np.diag(1 - np.power(h[:, -1], 2)) @ dLdh[:, -1]
        dLdc[:, -1] = dLdo

        for t in range(self.bptt_depth - 2, 1, -1):
            dLdo = self.softmax_jacobian(y_hat[:, t]) @ delta[:, t]
            dLdV[:, :, t] = np.outer(dLdo, h[:, t])
            dLdh[:, t] = self.W.T @ dLdh[:, t + 1] @ np.diag(1 - np.power(h[:, t + 1], 2)) + self.V.T @ dLdo
            dLdW[:, :, t] = np.outer((np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]), h[:, t - 1])
            dLdU[:,:,t] = np.outer(dLdh[:,t], y[:,t])
            dLdb[:, t] = np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]
            dLdc[:, t] = dLdo

        dLdV_accum = np.sum(dLdV, 2)
        dLdW_accum = np.sum(dLdW, 2)
        dLdU_accum = np.sum(dLdU, 2)
        dLdb_accum = np.sum(dLdb, 1)
        dLdc_accum = np.sum(dLdc, 1)
        return L, dLdV_accum, dLdW_accum, dLdU_accum, dLdb_accum, dLdc_accum


# Modified recirculation-trained Elman network
class LocalRNN(BaseRNN):
    """
    Modified-recircuation based RNN: numpy based
    Inherits from BaseRNN
    """
    def __init__(self, N, hidden_N, bptt_depth):
        super().__init__(N, hidden_N, bptt_depth)

    def gradient(self, y, h0):
        h, y_hat, L = self.forward_propagation(y, h0)
        delta = y_hat - y
        L = np.sum(np.power(delta,2))

        # initialize jacobians used in error computation
        dLdU = np.zeros((self.U.shape[0], self.U.shape[1], np.int(self.bptt_depth)))
        dLdV = np.zeros((self.V.shape[0], self.V.shape[1], np.int(self.bptt_depth)))
        dLdh = np.zeros((h.shape[0], np.int(self.bptt_depth)))
        dLdW = np.zeros((self.W.shape[0], self.W.shape[1], np.int(self.bptt_depth)))
        dLdb = np.zeros((self.b.shape[0], np.int(self.bptt_depth)))
        dLdc = np.zeros((self.c.shape[0], np.int(self.bptt_depth)))

        for t in range(1, self.bptt_depth):
            dLdo = self.softmax_jacobian(y_hat[:, t]) @ delta[:, t]
            dLdV[:, :, t] = np.outer(dLdo, h[:, t])
            dLdh[:, t] = self.U @ dLdo # truncated 
            dLdW[:, :, t] = np.outer((np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]), h[:, t - 1])
            dLdU[:,:,t] = np.outer(dLdh[:,t], y[:,t])
            dLdb[:, t] = np.diag(1 - np.power(h[:, t], 2)) @ dLdh[:, t]
            dLdc[:, t] = dLdo

        dLdV_accum = np.sum(dLdV, 2)
        dLdW_accum = np.sum(dLdW, 2)
        dLdU_accum = np.sum(dLdU, 2)
        dLdb_accum = np.sum(dLdb, 1)
        dLdc_accum = np.sum(dLdc, 1)
        return L, dLdV_accum, dLdW_accum, dLdU_accum, dLdb_accum, dLdc_accum
