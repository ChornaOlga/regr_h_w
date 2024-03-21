import numpy as np
import math as mt
import cProfile

from HW_Chorna.data_transform.science_package.MNK import LSM


def profileit(func):
    """
        Decorator (function wrapper) that profiles a single function
        @profileit()
        def func1(...)
                # do something
            pass
    """

    def wrapper(*args, **kwargs):
        cp = cProfile.Profile()  # використовуємо профайлер
        cp.enable()
        some_result = func(*args, **kwargs)
        cp.disable()
        cp.print_stats()
        return some_result

    return wrapper


# ------------------------------ Виявлення АВ за алгоритмом medium -------------------------------------
@profileit
def Sliding_Window_AV_Detect_medium(S0, n_Wind, Q):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    # -------- еталон  ---------
    j = 0
    for i in range(n_Wind):
        l = (j + i)
        S0_Wind[i] = S0[l]
        dS_standart = np.var(S0_Wind)
        scvS_standart = mt.sqrt(dS_standart)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        mS = np.median(S0_Wind)
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # --- детекція та заміна АВ --
        if scvS > (Q * scvS_standart):
            # детектор виявлення АВ
            S0[l] = mS
    return S0


# ------------------------------ Виявлення АВ за МНК -------------------------------------
@profileit
def Sliding_Window_AV_Detect_MNK(S0, Q, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    # -------- еталон  ---------
    MNK_polynom = LSM(S0, option='polynom')
    Speed_standart = MNK_polynom[1, 0]
    Yout_S0 = LSM(S0)
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS)
        # --- детекція та заміна АВ --
        Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter))
        Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
        if Speed_1 > Speed_standart_1:
            # детектор виявлення АВ
            S0[l] = Yout_S0[l, 0]
    return S0


# ------------------------------ Виявлення АВ за алгоритмом sliding window -------------------------------------
@profileit
def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    # ---- параметри циклів ----
    iter = len(S0)
    j_Wind = mt.ceil(iter - n_Wind) + 1
    S0_Wind = np.zeros((n_Wind))
    Midi = np.zeros((iter))
    # ---- ковзне вікно ---------
    for j in range(j_Wind):
        for i in range(n_Wind):
            l = (j + i)
            S0_Wind[i] = S0[l]
        # - Стат хар ковзного вікна --
        Midi[l] = np.median(S0_Wind)
    # ---- очищена вибірка  -----
    S0_Midi = np.zeros((iter))
    for j in range(iter):
        S0_Midi[j] = Midi[j]
    for j in range(n_Wind):
        S0_Midi[j] = S0[j]
    return S0_Midi
