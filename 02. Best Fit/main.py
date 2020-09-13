from statistics import mean

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import style


style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.plot(xs, ys)
# plt.show()


def best_fit_slope(xs, ys):
    # https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    # also known as Element-wise product, Entry-wise product, Schur product
    # element_wise_product = xs * ys

    # m = (
    #         ( (mean(xs) * mean(ys)) - mean(xs * ys) )
    #         /
    #         ( (mean(xs) * mean(xs)) - mean(xs * xs) )
    # )

    numerator = (mean(xs) * mean(ys)) - mean(xs * ys)
    denominator = (mean(xs) * mean(xs)) - mean(xs * xs)

    slope = numerator/denominator

    return slope


def best_fit_intercept(xs, ys):
    intercept = mean(ys) - (best_fit_slope(xs, ys) * mean(xs))

    return intercept


def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)


def coefficient_of_determination(ys_orig, ys_line):
    # https://en.wikipedia.org/wiki/Coefficient_of_determination
    # R squared
    ys_mean_line = np.array([mean(ys_orig) for _ in ys_orig])
    squared_error_regression_line = squared_error(ys_orig, ys_line)
    squared_error_ys_mean_line = squared_error(ys_orig, ys_mean_line)
    return 1 - (squared_error_regression_line / squared_error_ys_mean_line)


m = best_fit_slope(xs, ys)
b = best_fit_intercept(xs, ys)
# print(m, b)

regression_line = np.array([(m*x)+b for x in xs])

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()
