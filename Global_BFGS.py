import numpy as np
import argparse
import matplotlib.pyplot as plt


# BFGS Parameters

e = 5e-4
sigma = 1e-6
ro = 0.9
t0 = 1
gamma = 2


# Bateman function starting points, measurements

x_0 = np.array([0.05, 0.1, 0.4])
bateman_samples = [(15, 0.038),
                    (25, 0.085),
                    (35, 0.1),
                    (45, 0.103),
                    (55, 0.093),
                    (65, 0.095),
                    (75, 0.088),
                    (85, 0.08),
                    (105, 0.073),
                    (185, 0.05),
                    (245, 0.038),
                    (305, 0.028),
                    (365, 0.02)]


# Rosenbrock Function Methods
def ros(x):
    return 100 * (x[1] - x[0] * 2) * 2 + (1 - x[0]) ** 2

def grad_ros(x):
    return np.array([400 * x[0] * 3 - 400 * x[0] * x[1] + 2 * x[0] - 2, 200 * (x[1] - x[0] * 2)])

# Bateman Function Methods
def bateman(t, x):
    return x[2] * (np.exp(-x[0]*t) - np.exp(-x[1]*t))


# LMS Function

def LMS_bate(x):
    sum = 0
    for ti, yi in bateman_samples:
        sum += (bateman(ti, x) - yi)**2
    return sum / 2


# LMS gradient
def grad_LMS(x):
    epsilon = 1e-6
    gradient = np.zeros([x.size])
    for i in range(x.size):
        h = np.zeros([x.size])
        h[i] = epsilon
        gradient[i] = (LMS_bate(x+h) - LMS_bate(x-h)) / (2*epsilon)
    return gradient

# Helper methods

def phi(fn, x, t, d):
    return fn(x + t * d)

def grad_phi(fn_grad, x, t, d):
    return np.dot(phi(fn_grad, x, t, d), d)

def psi(fn, fn_grad, x, t, d):
    return phi(fn, x, t, d) - phi(fn, x, 0, d) - sigma * t * grad_phi(fn_grad, x, 0, d)


# BFGS method
def BFGS(fn, fn_grad, x):
    H = np.identity(len(x))
    t = t0
    iteration = 0
    while np.linalg.norm(fn_grad(x)) > e:
        d = np.matmul(-fn_grad(x).T, np.linalg.pinv(H))
        t = wolfe_powell(fn, fn_grad, x, t, d)
        x_old = x
        x = x_old + t * d
        s = x - x_old
        y = fn_grad(x) - fn_grad(x_old)
        y = y.reshape(len(x),1)
        s = s.reshape(len(x),1)
        H = H + np.dot(y, y.T) / np.dot(y.T, s) - np.dot(H,s).dot(np.dot(H,s).T) / np.dot(s.T, H).dot(s)
        iteration += 1
        print('x: ', x, 'f(x): ', fn(x),  'Iterations: ', iteration,)
    return x, fn(x), iteration



# Wolfe_Powell Rule

def wolfe_powell(fn, fn_grad, x, t, d):
    t = 1
    while(psi(fn, fn_grad, x, t, d)<0):
        if grad_phi(fn_grad, x, t, d) >= grad_phi(fn_grad, x, 0, d) * ro:
            return t
        else:
            t = gamma*t
    a = 0
    b = t

    while(True):
        t = a + (b-a)/2
        if psi(fn, fn_grad, x, t, d) >= 0:
            b = t
        elif grad_phi(fn_grad, x, t, d) >= ro * grad_phi(fn_grad, x, 0, d):
            return t
        else:
            a = t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f ', "--function", help='Rosenbrock or Bateman', default='Bateman')
    args = parser.parse_args()
    if args.function == 'Rosenbrock' or args.function == 'R':
        x, func, counter = BFGS(ros, grad_ros, np.array([-1.2, 1]).T)
    else :
        x, func, counter = BFGS(LMS_bate, grad_LMS, np.array([0.05, 0.1, 0.4]).T)

        opt_param = x;
        plt.scatter([i[0] for i in bateman_samples], [i[1] for i in bateman_samples], c='m')
        plt.plot(np.arange(0, 365, 0.01),  [bateman(i, x) for i in np.arange(0, 365, 0.01)], '--b')

        plt.show()

if __name__ == "__main__":
    main()