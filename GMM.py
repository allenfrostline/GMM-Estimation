import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, root


class gmm():
    
    def __init__(self, g, gp):
        self.g = g       #  g[i](theta, x)   i=1...n_momn
        self.gp = gp     # gp[j](theta)      j=1...n_para
        self.n_momn = len(g)
        self.n_para = len(gp)
        self.theta = np.random.rand(self.n_para)
    
    def fit(self, x):
        G          = lambda theta, x: np.matrix([g_(theta, x) for g_ in self.g])
        Gm         = lambda theta, x: np.mean(G(theta, x), axis=1)
        jac        = lambda theta: np.asarray((2 * np.vstack([gp_(theta) for gp_ in self.gp]) * (self.W * Gm(theta, self.x))).T)
        objective  = lambda theta, x, W: np.asscalar(Gm(theta, x).T * W * Gm(theta, x))
        covariance = lambda theta, x: G(theta, x) * G(theta, x).T / len(x)
        self.x     = x
        self.W     = np.identity(self.n_momn)
        iterations = 20
        print_len  = 1 + 15 * self.n_para
        print(' '*((print_len-13)//2)+'GMM Iteration'+' '*((print_len-13)//2))
        print('='*print_len)
        print(' #      {}'.format('        '.join(['theta_{}'.format(i+1) for i in range(self.n_para)])))
        print('-'*print_len)
        for i in range(iterations):
            self.theta = minimize(lambda theta: objective(theta, self.x, self.W), self.theta, jac=jac, method='SLSQP').x
            print('{:>2}   {}'.format(i+1, '    '.join(['{: .4e}'.format(i) for i in self.theta])))
            self.W = np.linalg.inv(covariance(self.theta, self.x))
        print('-'*print_len)
        self.objective = objective
    
    def result(self, theta_true):
        print('{:^36}'.format('Iterated GMM Results'))
        print('='*36)
        print('coef       estimated      true value')
        print('-'*36)
        print('obj       {: .4e}    {: .4e}'.format(self.objective(self.theta, self.x, self.W), self.objective(theta_true, self.x, np.identity(self.n_momn))))
        for i in range (self.n_para):
            print('theta_{}   {: .4e}    {: .4e}'.format(i+1, self.theta[i], theta_true[i]))
        print('-'*36)



def simulate(T, theta_x, theta_e, init, disp=0):
    len_tx, len_te = len(theta_x), len(theta_e)
    T_ = int(T * 1.1) # we drop the first 1/11 observations here as burn-in
    X = np.ndarray(T_)
    X[0] = init[0]
    e = np.random.normal(0, 1, T_)
    for t in range(1, T_):
        X[t] = sum([theta_x[tx]*X[t - tx] for tx in range(len_tx)]) + e[t] + sum([theta_e[te]*e[t-te-1] for te in range(len_te)])
    X = X[T_ - T:]
    if disp: 
        plt.close()
        fig = plt.figure(figsize=(14,6))
        ax1 = fig.add_subplot(211)
        ax1.plot(X, 'k-')
        ax1.set_xlabel('t')
        ax1.set_ylabel('x')
        ax2 = fig.add_subplot(212)
        ax2.hist(X, color='#444444', bins=50)
        ax2.set_xlabel('x')
        ax2.set_ylabel('observations')
        plt.tight_layout()
        plt.show()
    return X