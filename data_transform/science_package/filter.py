import numpy as np
import matplotlib.pyplot as plt

# # ------------------------- алгоритм -а-b фільтрa ------------------------
def ABF(S0):
    iter = len(S0)
    Yin = np.zeros(iter)
    YoutAB = np.zeros(iter)
    T0 = 1
    for i in range(iter):
        Yin[i] = float(S0[i])
    # -------------- початкові дані для запуску фільтра
    Yspeed_retro = (Yin[1] - Yin[0]) / T0
    Yextra = Yin[0] + Yspeed_retro
    alpha = 2 * (2 * 1 - 1) / (1 * (1 + 1))
    beta = (6 / 1) * (1 + 1)
    YoutAB[0] = Yin[0] + alpha * (Yin[0])
    # -------------- рекурентний прохід по вимірам
    for i in range(1, iter):
        YoutAB[i] = Yextra + alpha * (Yin[i] - Yextra)
        Yspeed = Yspeed_retro + (beta / T0) * (Yin[i] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i] + Yspeed_retro
        alpha = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 / (i * (i + 1))

    return YoutAB


# ------------------------- алгоритм -а-b-g фільтрa ------------------------
def ABGF(S0):
    iter = len(S0)
    Yin = np.zeros(iter)
    YoutABG = np.zeros(iter)
    T0 = 1
    for i in range(iter):
        Yin[i] = float(S0[i])

    # initial value 0

    YoutABG[0] = Yin[0]

    Yspeed = (Yin[1] - Yin[0])
    Yextra = Yin[0] + Yspeed


    # initial value 1

    n = 1
    alpha = (3 * (3 * (n ** 2) - 3 * n + 2)) / (n * (n + 1) * (n + 2))
    beta = (18 * (2 * n - 1)) / (n * (n + 1) * (n + 2))
    gamma = 60 / (n * (n + 1) * (n + 2))

    YoutABG[1] = Yextra + alpha * (Yin[1] - Yextra)
    Yspeed = Yspeed + beta * (Yin[1] - Yextra)
    Yacceleration = (Yin[2] - 2 * Yin[1] + Yin[0])

    Yextra = YoutABG[1] + Yspeed + (Yacceleration * 0.5)
    Yspeedextra = Yspeed + Yacceleration
    Yaccelerationextra = Yacceleration

    # -------------- рекурентний прохід по вимірам
    for i in range(2, iter):

        # smoothing factors
        alpha = (3 * (3 * (i ** 2) - 3 * i + 2)) / (i * (i + 1) * (i + 2))
        beta = (18 * (2 * i - 1)) / (i * (i + 1) * (i + 2))
        gamma = 60 / (i * (i + 1) * (i + 2))

        YoutABG[i] = Yextra + alpha * (Yin[i] - Yextra)
        Yspeed = Yspeedextra + beta * (Yin[i] - Yextra)
        Yacceleration = Yaccelerationextra + gamma * (Yin[i] - Yextra)


        Yextra = YoutABG[i] + Yspeed + Yacceleration * 0.5
        Yspeedextra = Yspeed + Yacceleration
        Yaccelerationextra = Yacceleration

    return YoutABG
