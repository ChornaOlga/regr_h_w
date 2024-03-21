import numpy as np

from HW_Chorna.data_transform.science_package.anomalies_detection import *
from HW_Chorna.data_transform.science_package.filter import *
from HW_Chorna.data_transform.science_package.MNK import fit_LSM


def del_trend_from_model(model):
    """
        Remove the trend component from a given time series model using the method of least squares (MNK).

        Parameters:
        - model (array-like): Input array representing the time series model.

        Returns:
        - numpy.ndarray: Output array representing the time series model with the trend component removed.
    """
    iter = len(model)
    polynome_pow, Yout = fit_LSM(model)  # визначається за МНК
    model_minus_trend = np.zeros((iter))
    for i in range(iter):
        model_minus_trend[i] = model[i] - Yout[i]

    model_MNK_dict = {'model': model_minus_trend, 'yout': Yout}

    return polynome_pow, model_MNK_dict


def del_anomaly_from_model(model):
    """
        Detect and clean anomalies from the given model using different methods.

        Parameters:
        - model (array-like): Input array representing the model.

        Returns:
        - numpy.ndarray: Array containing the model with anomalies removed.
    """
    print('Anomalies detection and cleaning:')
    print('1 - Medium method')
    print('2 - Least squares method')
    print('3 - Sliding window method')
    mode = int(input('mode:'))
    S_AV_Detect = np.zeros(len(model))

    if mode == 1:
        print('The sample was cleaned of outliers using the medium method')
        # --------- Увага!!! якість результату залежить від якості еталонного вікна -----------
        N_Wind_Av = 5  # розмір ковзного вікна для виявлення АВ
        Q = 1.6  # коефіцієнт виявлення АВ
        S_AV_Detect = Sliding_Window_AV_Detect_medium(model, N_Wind_Av, Q)

    if mode == 2:
        print('The sample was cleaned of outliers using the least squares method')
        # ------------------- Очищення від аномальних похибок МНК --------------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        Q_MNK = 7  # коефіцієнт виявлення АВ
        S_AV_Detect = Sliding_Window_AV_Detect_MNK(model, Q_MNK, n_Wind)

    if mode == 3:
        print('The sample was cleaned of outliers using the sliding window method')
        # --------------- Очищення від аномальних похибок sliding window -------------------
        n_Wind = 5  # розмір ковзного вікна для виявлення АВ
        S_AV_Detect = Sliding_Window_AV_Detect_sliding_wind(model, n_Wind)

    return S_AV_Detect


def smooth_model(model):
    """
        Apply a smoothing filter to the given model.

        Parameters:
        - model (array-like): Input array representing the model.

        Returns:
        - numpy.ndarray: Array containing the smoothed model.
    """
    smoothed_model = np.zeros(len(model))
    print('Smoothing the model:')
    print('1 - Alpha-beta filter')
    print('2 - Alpha-beta-gamma filter')
    mode = int(input('mode:'))

    if mode == 1:
        smoothed_model = ABF(model)

    if mode == 2:
        smoothed_model = ABGF(model)

    return smoothed_model
