import numpy as np


# Starting params

beta = 0.5
sigma = 1e-4
epsilon = 1e-4

# Initial points
x_0 = [-1.2, 1]

x = x_0[0]
y = x_0[1]


def rosenbrock(x1, x2):

    r_x = 100 * (x2 - x1 ** 2)**2 + (1 - x1) ** 2
    return r_x


def grad_r(x1, x2):
    grad = [400 * np.power(x1, 3) - 400 * x1 * x2 + 2*x1 - 2, 200 * (x2 - np.power(x1, 2))]
    return np.asarray(grad)


def norm(gradient):

    left = np.power(gradient[0], 2)
    right = np.power(gradient[1], 2)
    print("Gradient", np.sqrt(left+right))
    return np.sqrt(left+right)


def compute_step_size(points, sigma, d_k):
    t_k_test = []

    for index in np.arange(0, 1, 5e-4):
        input_test = np.asarray(points) + index * d_k
        # print(t_k_test)
        rosen_test = rosenbrock(input_test[0], input_test[1])
        rosen_constraint = rosenbrock(points[0], points[1]) + index * sigma * np.matmul(grad_r(points[0], points[1]).T, d_k)
        if rosen_test <= rosen_constraint:
            t_k_test.append(index)

    return max(t_k_test)


def grad_descent_ros(initial_points, sigma, epsilon):
    x = initial_points[0]
    y = initial_points[1]
    x_k = initial_points
    min_value = rosenbrock(x, y)

    # Initialize iteration counter
    k = 0

    while norm(grad_r(x_k[0], x_k[1])) > epsilon:
        print(norm(grad_r(x_k[0], x_k[1])))
        # Direction vector

        d_k = -1 * np.asarray(grad_r(x_k[0], x_k[1]))
        t_k = compute_step_size(x_k, sigma, d_k)
        # print(t_k)
        x_k = x_k + t_k * d_k
        k = k + 1
        min_value = rosenbrock(x_k[0], x_k[1])
        print("min_value: ", min_value)
        print("Iteration: ", k)

    return min_value, k


min_value_rose, iterations = grad_descent_ros(x_0, sigma, epsilon)
print(min_value_rose, iterations)





