import numpy as np
import math as mt
import matplotlib.pyplot as plt


def shifts(model):
    size = len(model)
    model_shifts = np.zeros((5, size))
    for shift in range(1, 6):
        model_shifts[shift - 1] = np.roll(model, -shift)
    return model_shifts


def discrete_derivatives(model):
    size = len(model)
    model_shift = shifts(model)
    model_derivative = np.zeros((5, size))
    model_derivative[0] = model_shift[0] - model
    model_derivative[1] = model_shift[1] - 2 * model_shift[0] + model
    model_derivative[2] = model_shift[2] - 3 * model_shift[1] + 3 * model_shift[0] - model
    model_derivative[3] = model_shift[3] - 4 * model_shift[2] + 6 * model_shift[1] - 4 * model_shift[0] + model
    model_derivative[4] = model_shift[4] - 5 * model_shift[3] + 10 * model_shift[2] - 10 * model_shift[1] + 5 * \
                          model_shift[0] - model

    for index in range(5):
        model_derivative[index][-(index + 1):] = np.nan

    return model_derivative


def numpy_statistical_characteristics(derivatives):
    numpy_derivatives_mean = np.nanmedian(derivatives, axis=1)
    numpy_derivatives_var = np.nanvar(derivatives, axis=1)
    print('numpy derivatives means:')
    print(numpy_derivatives_mean)
    print('numpy derivatives dispersions:')
    print(numpy_derivatives_var)
    return


def experimental_derivatives_mean(derivatives):
    size = len(derivatives[0])
    derivatives_mean = np.zeros(5)
    for index in range(5):
        bias = 1 / (size - (index + 1))
        derivatives_mean[index] = bias * np.nansum(derivatives[index])

    print('experimental derivatives means')
    print(derivatives_mean)
    return derivatives_mean


def experimental_derivatives_dispersion(derivatives):
    size = len(derivatives[0])
    derivatives_dispersions = np.zeros(5)
    derivatives_mean = experimental_derivatives_mean(derivatives)
    for index in range(5):
        bias = 1 / ((size - (index + 1)) - 1)
        diff = derivatives[index] - derivatives_mean[index]
        diff_2 = diff ** 2
        derivatives_dispersions[index] = bias * np.nansum(diff_2)

    return derivatives_dispersions


def delta(theor_st_dev, experimental_var):
    return np.abs(np.sqrt(experimental_var) - theor_st_dev)


def model_create(size):
    # parameters of normal errors
    error_mean = 0
    error_standard_deviation = 10

    # errors generation
    rng = np.random.default_rng()
    error = rng.normal(error_mean, error_standard_deviation, size)
    plt.hist(error)
    plt.title('The Normal Distribution of Errors')
    plt.show()

    # trend generation
    trend = np.zeros(size)
    for i in range(size):
        # linear
        # trend[i] = i

        # squared
        trend[i] = (i * i)
        # trend[i] = (0.0000005 * i * i)

        # cubical
        # trend[i] = (i * i * i)

    # model creation
    model = error + trend

    return model, 2*np.var(error)


if __name__ == '__main__':
    sample_size = 10

    model, theoretical_dispersion = model_create(sample_size)
    plt.plot(model)
    plt.title('Model = Trend + Errors')
    plt.show()
    print('theoretical dispersion of transformations in model:')
    print(theoretical_dispersion)
    print('theoretical standard deviation of transformations in model:')
    theoretical_standard_deviation = mt.sqrt(theoretical_dispersion)
    print(theoretical_standard_deviation)

    # derivatives up to 5-th power
    derivatives = discrete_derivatives(model)

    # calculation of means and dispersions of derivatives via numpy
    numpy_statistical_characteristics(derivatives)

    # calculation of means and dispersions of derivatives manually
    experimental_dispersions = experimental_derivatives_dispersion(derivatives)
    print('experimental derivatives dispersion:')
    print(experimental_dispersions)

    # deltas of dispersions between theoretical and experimental
    dispersions_delta = np.abs(experimental_dispersions - theoretical_dispersion)
    print('delta of dispersions:')
    print(dispersions_delta)

    # deltas of standard deviation between theoretical and experimental
    st_dev_delta = delta(theoretical_standard_deviation, experimental_dispersions)
    print('delta of standard deviation:')
    print(st_dev_delta)
