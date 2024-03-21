import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt


def model_create_interface(size, *random_error_generation_params):
    """
        Interface for creating a synthetic model with user-defined parameters.

        Parameters:
        - size (int): Size of the synthetic model.
        - *random_error_generation_params: Variable number of parameters for random error generation.

        Returns:
        - numpy.ndarray: Array containing the synthetic model.
    """

    model_params = dict()
    model = np.zeros(size)
    # User choice of error and trend types
    print('Choose random error generation type:\n 1. Uniform \n 2. Normal')
    model_params['error_type'] = 'uniform' if int(input('mode:')) == 1 else 'normal'
    print('Choose trend type:\n 1. Linear \n 2. Cubical')
    model_params['trend_type'] = 'linear' if int(input('mode:')) == 1 else 'cubic'

    anomalies_mode = input('Do you want to add anomalies to the data? (y/n)\n')
    if anomalies_mode == 'y':
        # Additive model with anomalies creation
        model = trend_errors_anomalies_model_compilation(size, *random_error_generation_params, **model_params)
    if anomalies_mode == 'n':
        # Additive model without anomalies creation
        model = error_plus_trend_model_compilation(size, *random_error_generation_params, **model_params)

    return model


def random_error_sampling(*params, dist_type, size):
    """
        Generate a random sample of errors from specified distribution parameters.

        Parameters:
        - *params: Variable number of distribution parameters. For 'uniform' distribution, provide (low, high),
                   and for 'normal' distribution, provide (mean, stdev).
        - dist_type (str): Type of distribution, either 'uniform' or 'normal'.
        - size (int): Size of the random error sample.

        Returns:
        - numpy.ndarray: Array containing the generated random error sample.
    """
    rng = np.random.default_rng()
    error_dist = np.zeros(size)
    mS, scvS, dS = np.zeros(3)

    if dist_type == 'uniform':
        low, high = params
        error_dist = rng.uniform(low, high, size)
    if dist_type == 'normal':
        mean, stdev = params
        error_dist = rng.normal(mean, stdev, size)

    mS = np.median(error_dist)
    dS = np.var(error_dist)
    scvS = mt.sqrt(dS)
    print('------------ Error sampling generation characteristics -------------')
    print('Mean: ', mS)
    print('Dispersion: ', dS)
    print('Standard deviation: ', scvS)
    print('-----------------------------------------------------')

    return error_dist


def anomaly_sampling(size):
    """
        Generate random indexes for introducing anomalies in a synthetic model.

        Parameters:
        - size (int): Size of the synthetic model.

        Returns:
        - numpy.ndarray: Array containing random indexes for introducing anomalies.
    """
    rng = np.random.default_rng()
    anomaly_count = round(size * 0.1)
    anomaly_indexes = rng.integers(low=1, high=size, size=anomaly_count)
    return anomaly_indexes


def trend_sampling(trend_type, size):
    """
        Generate a synthetic trend based on the specified trend type.

        Parameters:
        - trend_type (str): Type of trend, either 'linear' or 'cubic'.
        - size (int): Size of the synthetic trend sample.

        Returns:
        - numpy.ndarray: Array containing the generated synthetic trend.
    """
    trend = np.zeros(size)
    for i in range(size):
        if trend_type == 'linear':
            trend[i] = (0.005 * i)
        if trend_type == 'cubic':
            trend[i] = (0.0000000001 * i * i * i)
    return trend


def error_plus_trend_model_compilation(size, *numeric_parameters, **type_parameters):
    """
       Compile a synthetic model by combining a trend and random errors.

       Parameters:
       - size (int): Size of the synthetic model.
       - *numeric_parameters: Variable number of numeric parameters.
                             Provide the error_type and trend_type in this order.
       - **type_parameters: Variable number of keyword parameters.
                            Specify error_type and trend_type as keyword arguments.

       Returns:
       - numpy.ndarray: Array containing the compiled synthetic model.
    """
    # Samplings generations
    RE = random_error_sampling(*numeric_parameters, dist_type=type_parameters['error_type'], size=size)
    Trend = trend_sampling(trend_type=type_parameters['trend_type'], size=size)

    # Model using created samplings
    return Trend + RE


def trend_errors_anomalies_model_compilation(size, *numeric_parameters, **type_parameters):
    """
       Compile a synthetic model by combining a trend and random errors.

       Parameters:
       - size (int): Size of the synthetic model.
       - *numeric_parameters: Variable number of numeric parameters.
                             Provide the error_type and trend_type in this order.
       - **type_parameters: Variable number of keyword parameters.
                            Specify error_type and trend_type as keyword arguments.

       Returns:
       - numpy.ndarray: Array containing the compiled synthetic model.
    """
    # Samplings generations
    RE = random_error_sampling(*numeric_parameters, dist_type=type_parameters['error_type'], size=size)
    # error sampling generation characteristics
    mS = np.median(RE)
    dS = np.var(RE)
    scvS = mt.sqrt(dS)

    Trend = trend_sampling(trend_type=type_parameters['trend_type'], size=size)

    Model = Trend + RE

    rng = np.random.default_rng()
    anomalies_indexes = anomaly_sampling(size)
    for index in anomalies_indexes:
        Model[index] += rng.normal(mS, 3 * scvS, 1)

    # Model using created samplings
    return Model


if __name__ == '__main__':
    task_size = 10000
    anomalies = anomaly_sampling(task_size)
    print(anomalies)
