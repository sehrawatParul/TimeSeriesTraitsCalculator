import numpy as np
import datetime as dt


def exp(x, a, b, c):
    y = c + a * (x ** (b - 1) * b)
    return y


def rev_logi(x, a, b, c, d):
    return b + (a / (1. + np.exp(-c * (-x - d))))


def logi(x, a, b, c, d):
    return b + (a / (1. + np.exp(-c * (x - d))))


def double_logi(x, a, b, c, d, e, f):
    y = a + (b / (1 + (np.exp(-1 * ((x - c + (d / 2)) / e))))) * (
            1 - (1 / (1 + np.exp(-1 * ((x - c - (d / 2)) / f)))))
    return y


def gaussian(x, a, b, c):
    return a * np.exp(-np.power(x - b, 2) / (2 * np.power(c, 2)))


def gompertz_model(x, a, b, c):
    return c * np.exp(-b * np.exp(-x / a))


def model2(x, a, b, c, d, m):
    return c * np.exp(-b * np.exp(-x / a)) + d * x + m


def date_from_index_NDVI(input_index):
    dt_x1 = []
    for i in input_index:
        t = i[10:]  # removing prefix from date
        dt_x1.append(t)
    dt_x1 = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dt_x1]
    dt_x1 = np.array(dt_x1)
    return dt_x1



def date_from_index_ht(input_index):
    dt_x1 = []
    for i in input_index:
        t = i[11:]  # removing prefix from date
        dt_x1.append(t)
    dt_x1 = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dt_x1]
    dt_x1 = np.array(dt_x1)
    return dt_x1


def remove_init_vals(y):
    idx_y = []
    if y[0] > y[1]:
        idx_y.append(0)
        i = 1
        while y[i] > y[i + 1]:
            idx_y.append(i)
            i = i + 1
    '''        
    if idx_y:
        idx_y = int(idx_y[-1])
        if idx_y != 0:
            idx_y = np.arange(idx_y + 1)'''
    return idx_y


def remove_last_vals(y):
    idx_y = []
    if y[-1] > y[-2]:
        idx_y.append(-1)
        i = -2
        while y[i] > y[i - 1]:
            idx_y.append(i)
            i = i - 1
    return idx_y


def percentile_method(yp, yt):
    diff = np.array(yp - yt)
    upper_p, lower_p = np.percentile(diff, [93.5, 0])
    idx_y = []
    for j in diff:
        if j > upper_p:
            id = np.where(diff == j)
            idx_y.append(id[0])
        if j < lower_p:
            id = np.where(diff == j)
            idx_y.append(id[0])

    return idx_y


def percentile_method_ht(yp, yt):
    diff = np.array(yp - yt)
    upper_p, lower_p = np.percentile(diff, [98.25, 1.75])
    idx_y = []
    for j in diff:
        if j > upper_p:
            id = np.where(diff == j)
            idx_y.append(id[0])
        if j < lower_p:
            id = np.where(diff == j)
            idx_y.append(id[0])
    return idx_y


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
