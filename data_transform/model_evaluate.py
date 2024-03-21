import numpy as np
import math as mt
import matplotlib.pyplot as plt

from HW_Chorna.data_transform.science_package.MNK import LSM
from HW_Chorna.data_transform.model_clean import *


def statistical_characteristics(model):
    """
        Calculate and display statistical characteristics of a given model.

        Parameters:
        - model (array-like): Input array representing the model.

        Returns:
        None
    """
    mS = np.median(model)
    dS = np.var(model)
    scvS = mt.sqrt(dS)
    print('------------ Statistical characteristics -------------')
    print('Mean: ', mS)
    print('Dispersion: ', dS)
    print('Standard deviation: ', scvS)
    print('-----------------------------------------------------')
    return


def model_hist(model, text=''):
    """
        Generate and display a histogram for a given model.

        Parameters:
        - model (array-like): Input array representing the model.
        - text (str): Additional text for the y-axis label.

        Returns:
        None
    """
    plt.hist(model)
    plt.xlabel('Error value')
    plt.ylabel('Number of errors')
    plt.title(text)
    plt.show()


def model_visualise(model, text=''):
    """
        Visualize the original model and the model estimated using the method of least squares (MNK).

        Parameters:
        - model (array-like): Input array representing the original model.
        - text (str): Additional text for the y-axis label.

        Returns:
        None
    """
    plt.plot(model, label='Model data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(text)
    plt.legend()
    plt.show()


def model_with_aprox_visualise(model, Yout, text=''):
    """
        Visualize the original model and the model approximation.

        Parameters:
        - model (array-like): Input array representing the original model.
        - text (str): Additional text for the y-axis label.

        Returns:
        None
    """
    plt.plot(model, label='Model data')
    plt.plot(Yout, label='Approximation data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(text)
    plt.legend()
    plt.show()


def model_extrapolate(MNK_polynome_pow, Model):
    """
       Extrapolate a model using the method of the least squares (LSM).

       Parameters:
       - MNK_polynome_pow (int): Power of the polynomial in the LS method.
       - Model (array-like): Input array representing the model.

       Returns:
       - numpy.ndarray: Array containing the extrapolated model.
    """
    koef = int(input('Enter the length of the extrapolation segment: '))
    iter = len(Model)
    Yout_Extrapol = np.zeros((iter + koef, 1))
    MNK_coef = LSM(Model, power=MNK_polynome_pow, option='polynom')

    for i in range(iter + koef):
        for power, c in enumerate(MNK_coef):
            Yout_Extrapol[i] += c * mt.pow(i, power)  # проліноміальна крива МНК - прогнозування

    return Yout_Extrapol


def timeseries_model_analysis(Model, text=''):
    """
        Perform analysis on a time series model including:
         - visualization,
         - raw statistical characteristics,
         - optional anomaly detection,
         - optional anomaly cleaning,
         - optional smoothing,
         - detrending,
         - optional extrapolation.

        Parameters:
        - Model (array-like): Input array representing the time series model.
        - text (str): Additional text for display purposes.

        Returns:
        - numpy.ndarray: Array containing the cleaned and detrended time series model,
        i.e. array of errors that should have normal distribution with mean close to the 0.
    """
    # Full model evaluation
    print('------------ Initial model evaluation -------------')
    model_visualise(Model, text + ' Timeseries model')
    statistical_characteristics(Model)
    print('---------------------------------------------------\n\n')

    # Model cleaning if needed
    anomalies_mode = input('Do you want to clean the ' + text + ' data of outliers? (y/n)\n')
    if anomalies_mode == 'y':
        # Anomalies cleaning
        Model_without_anomalies = del_anomaly_from_model(Model)

        print('------------ After outliers detection and cleaning -------------')
        statistical_characteristics(Model_without_anomalies)
        model_visualise(Model_without_anomalies,
                        text + ' Timeseries model without outliers')
        print('---------------------------------------------------\n\n')

    if anomalies_mode == 'n':
        Model_without_anomalies = Model

    # Model smoothing if needed
    smoothing_mode = input('Do you want to smooth the ' + text + ' data? (y/n)\n')
    if smoothing_mode == 'y':

        Model_smoothed = smooth_model(Model_without_anomalies)
        model_with_aprox_visualise(Model_without_anomalies, Model_smoothed, text + ' Smoothed timeseries model without outliers')

    # Model detrending
    polynome_pow, Model_without_trend = del_trend_from_model(Model_without_anomalies)

    # Clean model evaluation
    print('------------ After data detrending -------------')
    statistical_characteristics(Model_without_trend['model'])
    model_with_aprox_visualise(Model_without_anomalies, Model_without_trend['yout'],
                               text + ' Timeseries model with LSM approximation')
    model_hist(Model_without_trend['model'], 'Errors distribution histogram')

    # Model extrapolation if needed
    extrapolation_mode = input('------------ Do you want to extrapolate model? (y/n)-------------\n')
    if extrapolation_mode == 'y':
        Model_extrapolate = model_extrapolate(polynome_pow, Model_without_anomalies)
        model_with_aprox_visualise(Model_without_anomalies, Model_extrapolate, 'Timeseries model with LSM extrapolation')
        return Model_without_anomalies, Model_extrapolate

    return Model_without_anomalies
