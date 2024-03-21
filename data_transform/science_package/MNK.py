import numpy as np
import pandas as pd
import math as mt
import matplotlib.pyplot as plt


def fit_LSM(S0):
    """
        Fit a model using the method of least squares method (LSM)
        with different polynomial powers and recommend the best-fit.

        Parameters:
        - S0 (array-like): Input array representing the timeseries model.

        Returns:
        - tuple: A tuple containing the recommended polynomial power and the best-fit model.
    """
    #Define maximum power of polynome
    max_poly_power = 5

    # Variables initialisation
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    LSM_power = np.arange(max_poly_power)
    r2_scores = np.zeros(max_poly_power)
    Youts = np.zeros((max_poly_power, iter))

    # Checking MNK with polynom powers up to max_poly_power
    for i in LSM_power:
        Youts[i] = LSM(S0, power=(i + 1)).reshape((iter,))
        r2_scores[i] = r2_score(S0, Youts[i],
                                'Evaluation of the quality of approximation by a polynomial of the ' +
                                str(i + 1) + ' degree')
    r2_score_diff = np.abs(np.diff(r2_scores))
    best_r2_score = np.argmin(r2_score_diff)
    polynom_recommendation = best_r2_score+1
    polynom_userchoise = polynom_recommendation

    print('For approximation, it is recommended to use a polynomial of the ' + str(polynom_recommendation) + ' degree')
    if_userchoise = input('Do you want manually specify the polynomial power? y/n:\n')
    if if_userchoise == 'y':
        polynom_userchoise = int(input('Specify the polynomial power: \n'))
    if if_userchoise == 'n':
        polynom_userchoise = polynom_recommendation

    return polynom_userchoise, Youts[best_r2_score]


def LSM(S0, power=2, option='sample'):
    """
        Perform the method of least squares method (LSM) to estimate the coefficients of a timeseries model.

        Parameters:
        - S0 (array-like): Input array of statistical data.

        Returns:
        - numpy.ndarray: Output array representing the estimated values based on the MNK model.
        """
    iter = len(S0)
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, (power + 1)))
    for i in range(iter):  # формування структури вхідних матриць МНК
        Yin[i, 0] = float(S0[i])  # формування матриці вхідних даних
        for j in range(1, (power+1)):
            F[i, j] = float(mt.pow(i, j))
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(Yin)
    Yout = F.dot(C)
    if option == 'sample':
        return Yout
    if option == 'polynom':
        return C


def r2_score(SL, Yout, Text):
    """
        Calculate the coefficient of determination (R^2 score) for a model.

        Parameters:
        - SL (array-like): Observed values.
        - Yout (numpy.ndarray): Predicted values from the model.
        - Text (str): Additional text for display purposes.

        Returns:
        - float: Coefficient of determination (R^2 score).
    """
    # статистичні характеристики вибірки з урахуванням тренду
    iter = len(Yout)
    numerator = 0
    denominator_1 = 0
    for i in range(iter):
        # numerator = numerator + (SL[i] - Yout[i, 0]) ** 2
        numerator = numerator + (SL[i] - Yout[i]) ** 2
        denominator_1 = denominator_1 + SL[i]
    denominator_2 = 0
    for i in range(iter):
        denominator_2 = denominator_2 + (SL[i] - (denominator_1 / iter)) ** 2
    R2_score_our = 1 - (numerator / denominator_2)

    # global linear deviation of the estimate - dynamic error of the model
    Delta = 0
    for i in range(iter):
        Delta = Delta + abs(SL[i] - Yout[i])
    Delta_average_Out = Delta / (iter + 1)

    print('------------', Text, '-------------')
    print('Sample size: ', iter)
    print('Coefficient of determination (probability of approximation): ', R2_score_our)
    print('Dynamic model error: ', Delta_average_Out)

    return R2_score_our


if __name__ == '__main__':
    pass
