from func_file import date_from_index_ht, gompertz_model, model2, find_nearest, percentile_method_ht
from sklearn.metrics import r2_score
from intersect import intersection
from scipy.stats import linregress
import time
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import pwlf
import scipy.optimize as opt
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('qtagg')

warnings.filterwarnings("ignore")
random.seed(20)


def trait_extractor_date(input_file, sh_name=None, img_loc=None, mentioned_ht=None, mentioned_cal_dt=None,
                         mentioned_perc_maxht=None, mentioned_perc_maxslp_dt=None,
                         mentioned_perc_maxslp_ht=None, val_at_mentioned_max_perc=None, outlier_removal=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot')
    df = raw_df.drop(columns=['ComonVar', 'Nrate', 'Pesticide', 'Sowing date', 'Replicate', 'Row', 'Column'])
    df_inv = df.T
    record_date = df_inv.index
    only_date = record_date[0].strip('Ht_Prcnt99 ')
    year_only = only_date[:4]

    df_inv['datex'] = np.array(df_inv.index)
    df_inv['datex'] = df_inv['datex'].apply(lambda x: str(x).removeprefix('Ht_Prcnt99 '))

    df_inv['datex'] = pd.to_datetime(df_inv['datex'])
    df_inv['datex'] = df_inv['datex'].map(dt.datetime.toordinal)

    dt_x_copy = date_from_index_ht(df_inv.index)

    df_inv[df_inv < 0] = 0

    X_t = np.array(df_inv['datex'])
    X_t_1 = X_t.copy()
    idx = np.array(df.index.values)

    def get_x(y):
        test_df = df_inv[y.notna()]
        X_t = np.array(test_df['datex'])
        X_t_mod = X_t
        DAS_var = []
        addition = 0
        DAS_var.insert(0, 0)
        for i in range(1, len(X_t)):
            yy = X_t[i] - X_t[i - 1]
            yyy = addition + yy
            addition = yyy
            DAS_var.append(yyy)

        X_t = np.array(DAS_var)
        return X_t, X_t_mod

    inflection_height1 = []
    inflection_date1 = []
    slope_at_inflection1 = []
    max_slope1 = []
    max_slope_height1 = []
    max_slope_date1 = []
    max_height1 = []
    max_height_date1 = []
    fitness_score_model1 = []
    execution_time1 = []

    inflection_height2 = []
    inflection_date2 = []
    slope_at_inflection_m2 = []
    max_slope2 = []
    max_slope_height2 = []
    max_slope_date2 = []
    max_height2 = []
    max_height_date2 = []
    fitness_score_model2 = []
    execution_time2 = []

    height_declining = []

    dt_from_ht_arr = []
    ht_from_dt_arr = []
    mentioned_perc_maxht_arr = []

    dt_at_mentioned_perc_max_slope_arr = []
    ht_at_mentioned_perc_max_slope_arr = []
    val_at_mentioned_max_perc_arr1 = []
    val_at_mentioned_max_perc_arr2 = []

    fig_j = 1
    # (1, len(df_inv.columns))
    for ij in range(1, len(df_inv.columns)):
        Y_t = df_inv[ij]
        Y_t_copy = Y_t.copy()
        X_t, X_t_mod = get_x(Y_t)
        Y_t = Y_t.dropna()
        mymodel = np.poly1d(np.polyfit(X_t, Y_t, 5))
        y_pred = mymodel(X_t)

        x_hat = np.arange(max(X_t) + 1, dtype=int)
        x_hat2 = np.arange(min(X_t_1), max(X_t_1) + 1)

        def inter_dt(ip):
            pt_1 = int(np.round(ip))
            idx_pt_1 = np.where(x_hat == pt_1)
            idx_pt_1 = int(idx_pt_1[0])
            intersect_dtt = x_hat2[idx_pt_1]
            intersect_date = dt.date.fromordinal(intersect_dtt)
            return intersect_date
        
        if outlier_removal is not None:
            idx_y = percentile_method_ht(y_pred, Y_t)
            for k in idx_y:
                Y_t[k] = np.nan
            idx_y = (np.where(Y_t.isnull()))[0]
            for ix in idx_y:
                X_t = np.where(np.isclose(X_t, X_t[ix]), np.nan, X_t)
            nan_X_t = np.isnan(X_t)
            not_nan_X_t = ~ nan_X_t
            X_t = X_t[not_nan_X_t]
            nan_Y_t = np.isnan(Y_t)
            not_nan_Y_t = ~ nan_Y_t
            Y_t = Y_t[not_nan_Y_t]
        
        df_idx = Y_t.index
        dt_x = date_from_index_ht(df_idx)

        date_arr = []
        delta = dt_x[-1] - dt_x[0]  # returns timedelta
        for i in range(delta.days + 1):
            day = dt_x[0] + timedelta(days=i)
            date_arr.append(day)

        x_hat = np.arange(min(X_t), max(X_t) + 1, dtype=int)

        # model1
        start1 = time.time()
        my_pwlf = pwlf.PiecewiseLinFit(X_t, Y_t)
        breaks = my_pwlf.fit(3)
        ijk = breaks[2]

        bp = find_nearest(X_t, ijk)
        idx_bp = np.where(X_t == bp)
        idx_bp = int(idx_bp[0])
        linear_x = X_t[idx_bp:]
        slopes = my_pwlf.calc_slopes()

        my_pwlf4 = pwlf.PiecewiseLinFit(X_t, Y_t)
        breaks4 = my_pwlf4.fit(4)
        ijk4 = breaks4[3]

        bp4 = find_nearest(X_t, ijk4)
        idx_bp4 = np.where(X_t == bp4)
        idx_bp4 = int(idx_bp4[0])
        linear_x4 = X_t[idx_bp4:]
        slopes4 = my_pwlf4.calc_slopes()

        if slopes[2] > slopes4[3] and len(linear_x4) > 3:
            gom_x = X_t[:idx_bp4 + 1]
            gom_dt = dt_x[:idx_bp4 + 1]
            gom_y = Y_t[:idx_bp4 + 1]
            lin_x = linear_x4
            lin_y = my_pwlf4.predict(lin_x)
            lin_dt = dt_x[idx_bp4:]
            grad_lin_y = np.gradient(lin_y, lin_x)
            if round(grad_lin_y[0], 6) > round(grad_lin_y[-1], 6):
                gom_x = X_t[:idx_bp4 + 2]
                gom_dt = dt_x[:idx_bp4 + 2]
                gom_y = Y_t[:idx_bp4 + 2]
                lin_x = X_t[idx_bp4 + 1:]
                lin_y = my_pwlf4.predict(lin_x)
                lin_dt = dt_x[idx_bp4 + 1:]
        else:
            gom_x = X_t[:idx_bp + 1]
            gom_dt = dt_x[:idx_bp + 1]
            gom_y = Y_t[:idx_bp + 1]
            lin_x = linear_x
            lin_y = my_pwlf.predict(lin_x)
            lin_dt = dt_x[idx_bp:]
            if len(lin_x) < 2:
                grad_lin_y = [0]
            else:
                grad_lin_y = np.gradient(lin_y, lin_x)
            if round(grad_lin_y[0], 6) > round(grad_lin_y[-1], 6):
                gom_x = X_t[:idx_bp + 2]
                gom_dt = dt_x[:idx_bp + 2]
                gom_y = Y_t[:idx_bp + 2]
                lin_x = X_t[idx_bp + 1:]
                lin_y = my_pwlf.predict(lin_x)
                lin_dt = dt_x[idx_bp + 1:]

        
        gom_x_cont = np.arange(min(gom_x), max(gom_x) + 1, dtype=int)
        gom_dt_cont = []
        delta = gom_dt[-1] - gom_dt[0]  # returns timedelta
        for i in range(delta.days + 1):
            day = gom_dt[0] + timedelta(days=i)
            gom_dt_cont.append(day)

        if len(lin_x) < 2:
            sl, intr = 0, 0
        else:
            sl, intr, rval, pval, stdr = linregress(lin_x, lin_y)
        y_plot = x_hat * sl + intr
        y_plot_X_t = X_t * sl + intr

        if sl < 0:
            height_declining.append('y')
        else:
            height_declining.append('n')

        popt, _ = opt.curve_fit(gompertz_model, gom_x, gom_y, method="trf", maxfev=1000000)
        ai, bi, ci = popt
        y_fit_gom = gompertz_model(gom_x, *popt)
        y_fit_gom_X_t = gompertz_model(X_t, *popt)
        y_gom_plot = gompertz_model(x_hat, *popt)
        y_fit_gom_cont = gompertz_model(gom_x_cont, *popt)


        x_intersect, y_intersect = intersection(x_hat, y_plot, x_hat, y_gom_plot)

        if y_intersect.size == 0:
            intersect_dt = lin_dt[0]
            y_intersect = lin_y[0]
            intersect_constant = 0
        elif y_intersect.size > 1:
            x_intersect = x_intersect[0]
            y_intersect = y_intersect[0]
            intersect_dt = inter_dt(x_intersect)
            intersect_constant = 1
        else:
            intersect_dt = inter_dt(x_intersect)
            intersect_constant = 1

        idx_x = np.where(x_hat == gom_x[-1])
        idx_x = int(idx_x[0])
        x_hat_gom = x_hat[:idx_x]
        y_fit_gom1 = gompertz_model(x_hat_gom, *popt)
        slop_gom_val = np.gradient(y_fit_gom1)
        max_gom_slope_idx = np.where(slop_gom_val == max(slop_gom_val))[0]
        if len(max_gom_slope_idx) > 1:
            max_gom_slope_idx = max_gom_slope_idx[0]
        max_gom_slope_idx = int(max_gom_slope_idx)
        max_slope = slop_gom_val[max_gom_slope_idx]
        dt_max_slope = dt.date.fromordinal(x_hat2[max_gom_slope_idx])
        ht_max_slope = y_fit_gom1[max_gom_slope_idx]

        grad_array = np.gradient(slop_gom_val)
        increment_grad_array = []
        for i in range(0, len(grad_array)):
            if grad_array[i] < grad_array[i + 1]:
                increment_grad_array.append(grad_array[i + 1])
            else:
                break

        if increment_grad_array:
            incr_grad_array = increment_grad_array[-1]
            inflect_constant = 0
        else:
            incr_grad_array = grad_array[1]
            inflect_constant = 1
        inflec_idx = np.where(grad_array == incr_grad_array)[0]
        if len(inflec_idx) > 1:
            inflec_idx = inflec_idx[0]
        inflec_idx = int(inflec_idx)
        dt_first_inflection = dt.date.fromordinal(x_hat2[inflec_idx])
        slp_at_inflection1 = grad_array[inflec_idx]
        first_inflection_pt = y_fit_gom1[inflec_idx]

        idx_intersect_dt = find_nearest(dt_x, intersect_dt)
        idx_intersect_dt = int(np.where(dt_x == idx_intersect_dt)[0])
        lin_start_idx = int(len(dt_x) - idx_intersect_dt - 1)
        a1 = np.array(y_fit_gom_X_t[:idx_intersect_dt + 1])
        if lin_start_idx == 0:
            comb_array = a1
        else:
            a2 = np.array(y_plot_X_t[-lin_start_idx:])
            comb_array = np.concatenate((a1, a2))

        r2_score_model1 = round(r2_score(Y_t, comb_array), 4)
        fitness_score_model1.append(r2_score_model1)
        end1 = time.time()
        time_model1 = round((end1 - start1), 4)
        execution_time1.append(time_model1)
        start2 = time.time()
        initial_guess = [ai, bi, ci, sl, intr]
        poptt, _ = opt.curve_fit(model2, X_t, Y_t, initial_guess, method="trf", maxfev=1000000)
        y_fit_model2 = model2(X_t, *poptt)
        y_model2_full = model2(x_hat, *poptt)

        # model 2 traits
        def model2_traits():
            max_ht2 = max(y_model2_full)
            idx_max_ht = int(np.where(y_model2_full == max_ht2)[0])
            max_ht_date2 = x_hat[idx_max_ht]
            max_ht_date2 = inter_dt(max_ht_date2)
            growth_arr_m2 = y_model2_full[:idx_max_ht]
            grad_growth_arr_m2 = np.gradient(growth_arr_m2)
            pos_slope_only = [item for item in grad_growth_arr_m2 if item >= 0]
            idx_pos_slp_only = np.where(grad_growth_arr_m2 == pos_slope_only[0])
            if len(idx_pos_slp_only[0]) > 1:
                idx_pos_slp_only = int(idx_pos_slp_only[0][0])
            else:
                idx_pos_slp_only = int(idx_pos_slp_only[0])
            inflection_pt_m2 = np.gradient(pos_slope_only)  # double gradient of y
            inflect_pos_slope_only = [item for item in inflection_pt_m2 if item > 0]
            if inflect_pos_slope_only:
                try:
                    increment_grad_array_m2 = []  # incremental double gradient values
                    for i in range(0, len(inflect_pos_slope_only)):
                        if inflect_pos_slope_only[i] < inflect_pos_slope_only[i + 1]:
                            increment_grad_array_m2.append(inflect_pos_slope_only[i + 1])
                        else:
                            break
                    if increment_grad_array_m2:
                        inflec_grad_m2 = increment_grad_array_m2[-1]
                        inflect_constant_m2 = 0
                    else:
                        inflec_grad_m2 = inflect_pos_slope_only[0]
                        inflect_constant_m2 = 1
                    add_comp = np.where(inflection_pt_m2 == inflec_grad_m2)[0][0]
                    idx_inflec_grad_m2 = idx_pos_slp_only + add_comp
                    inflec_ht_m2 = y_model2_full[idx_inflec_grad_m2]
                    inflect_dt_m2 = dt.date.fromordinal(x_hat2[idx_inflec_grad_m2])
                    slop_at_inflection2 = inflection_pt_m2[idx_inflec_grad_m2]
                except:
                    inflec_ht_m2 = np.nan
                    inflect_dt_m2 = np.nan
                    slop_at_inflection2 = np.nan
                    inflect_constant_m2 = np.nan
            else:
                inflec_ht_m2 = np.nan
                inflect_dt_m2 = np.nan
                slop_at_inflection2 = np.nan
                inflect_constant_m2 = np.nan

            max_growth_m2 = max(grad_growth_arr_m2)
            idx_comp = np.where(grad_growth_arr_m2 == max_growth_m2)[0]
            if len(idx_comp) > 1:
                idx_comp = idx_comp[0]
            idx_max_growth_m2 = int(idx_comp)
            max_slope_ht2 = y_model2_full[idx_max_growth_m2]
            max_slope_dt2 = dt.date.fromordinal(x_hat2[idx_max_growth_m2])
            if max_ht2 == y_model2_full[-1]:
                max_ht2 = "{}{}".format(max_ht2, '*')
                max_ht_date2 = "{}{}".format(max_ht_date2, '*')
            if inflect_constant_m2 == 1:
                inflec_ht_m2 = "{}{}".format(inflec_ht_m2, '*')
                inflect_dt_m2 = "{}{}".format(inflect_dt_m2, '*')
                slop_at_inflection2 = "{}{}".format(slop_at_inflection2, '*')
            return inflec_ht_m2, inflect_dt_m2, slop_at_inflection2, max_growth_m2, max_slope_ht2, max_slope_dt2, max_ht2, max_ht_date2
        
        r2_score_model2 = round(r2_score(Y_t, y_fit_model2), 4)
        if r2_score_model2 > r2_score_model1:
            inflec_ht_m2, inflect_dt_m2, slop_at_inflection2, max_growth_m2, max_slope_ht2, max_slope_dt2, max_ht2, max_ht_date2 = model2_traits()
            inflection_height2.append(inflec_ht_m2)
            inflection_date2.append(inflect_dt_m2)
            slope_at_inflection_m2.append(slop_at_inflection2)
            max_slope2.append(max_growth_m2)
            max_slope_height2.append(max_slope_ht2)
            max_slope_date2.append(max_slope_dt2)
            max_height2.append(max_ht2)
            max_height_date2.append(max_ht_date2)
            fitness_score_model2.append(r2_score_model2)

        else:
            inflection_height2.append(np.nan)
            inflection_date2.append(np.nan)
            slope_at_inflection_m2.append(np.nan)
            max_slope2.append(np.nan)
            max_slope_height2.append(np.nan)
            max_slope_date2.append(np.nan)
            max_height2.append(np.nan)
            max_height_date2.append(np.nan)
            fitness_score_model2.append(np.nan)
        end2 = time.time()
        time_model2 = round((end2 - start2), 4)
        execution_time2.append(time_model2)

        y_intersect = np.float(y_intersect)

        if val_at_mentioned_max_perc:
            if y_intersect:
                perc_ht_val1 = (float(val_at_mentioned_max_perc) * y_intersect)/100
                val_at_mentioned_max_perc_arr1.append(perc_ht_val1)
            else:
                val_at_mentioned_max_perc_arr1.append('')

        if mentioned_ht:
            if mentioned_ht < y_intersect:
                dt_from_ht = np.interp([mentioned_ht], y_model2_full, x_hat)
            else:
                dt_from_ht = np.interp([mentioned_ht], y_plot, x_hat)
            dt_from_ht = int(dt_from_ht[0]) + X_t_mod[0]
            dt_from_ht = dt.date.fromordinal(dt_from_ht)
            dt_from_ht_arr.append(dt_from_ht)

        if mentioned_cal_dt:
            mentioned_cal_dt = dt.datetime.strptime(str(mentioned_cal_dt), '%Y-%m-%d').date()
            x_ordinal = dt.datetime.toordinal(mentioned_cal_dt)
            x_diff = x_ordinal - X_t_mod[0]
            if mentioned_cal_dt < intersect_dt:
                ht_from_dt = gompertz_model(x_diff, *popt)
            else:
                ht_from_dt = x_diff * sl + intr
            ht_from_dt_arr.append(ht_from_dt)

        if mentioned_perc_maxht:
            mentioned_perc_maxht = float(mentioned_perc_maxht)
            ht_at_perc = (mentioned_perc_maxht * y_intersect)/100
            dt_of_perc_ht = np.interp([ht_at_perc], y_plot, x_hat)
            dt_of_perc_ht = int(dt_of_perc_ht[0]) + X_t_mod[0]
            dt_of_perc_ht = dt.date.fromordinal(dt_of_perc_ht)
            mentioned_perc_maxht_arr.append(dt_of_perc_ht)

        if mentioned_perc_maxslp_dt:
            slp_at_perc = (max_slope*float(mentioned_perc_maxslp_dt))/100
            incr_slp_arr = slop_gom_val[:max_gom_slope_idx+1]
            nearest_slp_val = find_nearest(incr_slp_arr, slp_at_perc)
            idx_nearest_slp_val = np.where(slop_gom_val == nearest_slp_val)[0]
            dt_slp = int(x_hat2[idx_nearest_slp_val])
            dt_slp = dt.date.fromordinal(dt_slp)
            dt_at_mentioned_perc_max_slope_arr.append(dt_slp)

        if mentioned_perc_maxslp_ht:
            slp_at_perc = (max_slope*float(mentioned_perc_maxslp_ht))/100
            incr_slp_arr = slop_gom_val[:max_gom_slope_idx+1]
            nearest_slp_val = find_nearest(incr_slp_arr, slp_at_perc)
            idx_nearest_slp_val = np.where(slop_gom_val == nearest_slp_val)[0]
            ht_perc_maxslope = y_fit_gom1[idx_nearest_slp_val]
            ht_at_mentioned_perc_max_slope_arr.append(ht_perc_maxslope)

        if img_loc:
            plt.figure(ij, figsize=(10, 9))
            plt.plot(dt_x_copy, Y_t_copy, 'go')
            plt.plot(dt_x, Y_t, 'ro')
            if r2_score_model2 > r2_score_model1:
                plt.plot(date_arr, y_model2_full, 'y')
            plt.plot(lin_dt, lin_y, 'b')
            plt.plot(intersect_dt, y_intersect, "*k")
            plt.plot(gom_dt_cont, y_fit_gom_cont, 'b')
            plt.xticks(dt_x_copy)
            plt.tick_params(labelrotation=90)
            plt.grid(True)
            labl1 = raw_df["ComonVar"].iloc[ij - 1]
            labl2 = raw_df["Nrate"].iloc[ij - 1]
            labl3 = raw_df["Pesticide"].iloc[ij - 1]
            labl4 = raw_df["Replicate"].iloc[ij - 1]
            if r2_score_model2 > r2_score_model1:
                plt.title(f"Year:{year_only}, Variety: {labl1},  Nrate: {labl2}, Rep: {labl4}, Pesticide: {labl3}, \n fit-score & time (blue): {r2_score_model1} & {time_model1}, fit-score & time (yellow): {r2_score_model2} & {time_model2}")
            else:
                plt.title(f"Year:{year_only}, Variety: {labl1},  Nrate: {labl2}, Rep: {labl4}, Pesticide: {labl3}, \n fit-score & time (blue): {r2_score_model1} & {time_model1}")
            plt.tight_layout()
            plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_date.jpg')
            plt.clf()
            # plt.show()
            fig_j = fig_j + 1

        if intersect_constant == 0:
            y_intersect = "{}{}".format(y_intersect, '*')
            intersect_dt = "{}{}".format(intersect_dt, '*')

        if inflect_constant == 1:
            first_inflection_pt = "{}{}".format(first_inflection_pt, '*')
            dt_first_inflection = "{}{}".format(dt_first_inflection, '*')
            slp_at_inflection1 = "{}{}".format(slp_at_inflection1, '*')
        

        inflection_height1.append(first_inflection_pt)
        inflection_date1.append(dt_first_inflection)
        slope_at_inflection1.append(slp_at_inflection1)
        max_slope1.append(max_slope)
        max_slope_height1.append(ht_max_slope)
        max_slope_date1.append(dt_max_slope)
        max_height1.append(y_intersect)
        max_height_date1.append(intersect_dt)

    output_traits = pd.DataFrame()
    output_traits['inflect_Ht_m1'] = inflection_height1
    output_traits['inflect_Dt_m1'] = inflection_date1
    output_traits['slope_at_inflection1'] = slope_at_inflection1
    output_traits['max_Slp_m1'] = max_slope1
    output_traits['max_Slp_Ht_m1'] = max_slope_height1
    output_traits['max_Slp_Dt_m1'] = max_slope_date1
    output_traits['max_Ht_m1'] = max_height1
    output_traits['max_Ht_Dt_m1'] = max_height_date1
    output_traits['fitness_score_model1'] = fitness_score_model1
    output_traits['execution_time1'] = execution_time1
    output_traits['height_declining'] = height_declining
    output_traits['inflect_Ht_m2'] = inflection_height2
    output_traits['inflect_Dt_m2'] = inflection_date2
    output_traits['slope_at_inflection_m2'] = slope_at_inflection_m2
    output_traits['max_Slp_m2'] = max_slope2
    output_traits['max_Slp_Ht_m2'] = max_slope_height2
    output_traits['max_Slp_Dt_m2'] = max_slope_date2
    output_traits['max_Ht_m2'] = max_height2
    output_traits['max_Ht_Dt_m2'] = max_height_date2
    output_traits['execution_time2'] = execution_time2

    if dt_from_ht_arr:
        output_traits[f'date_at_{mentioned_ht}'] = dt_from_ht_arr
    if ht_from_dt_arr:
        output_traits[f'height_at_{mentioned_cal_dt}'] = ht_from_dt_arr
    if mentioned_perc_maxht_arr:
        output_traits[f'date_at_{mentioned_perc_maxht}_perc_ht'] = mentioned_perc_maxht_arr
    if dt_at_mentioned_perc_max_slope_arr:
        output_traits[f'date_at_{mentioned_perc_maxslp_dt}_perc_maxslope'] = dt_at_mentioned_perc_max_slope_arr
    if ht_at_mentioned_perc_max_slope_arr:
        output_traits[f'height_at_{mentioned_perc_maxslp_ht}_perc_maxslope'] = ht_at_mentioned_perc_max_slope_arr
    if val_at_mentioned_max_perc_arr1:
        output_traits[f'height_at_{val_at_mentioned_max_perc}_perc_maxHeight'] = val_at_mentioned_max_perc_arr1
    if fitness_score_model2:
        output_traits['fitness_score_model2'] = fitness_score_model2


    df_info = raw_df[['ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column']]
    idx = np.array(df.index.values)
    output_traits.set_index(idx, inplace=True)
    output_df = df_info.join(output_traits)

    return output_df


def properties(input_file, sh_name=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot')
    df = raw_df.drop(columns=['ComonVar', 'Nrate', 'Pesticide', 'Sowing date', 'Replicate', 'Row', 'Column'])
    df_inv = df.T
    max_val_ht = df_inv.max().max()
    df_inv['datex'] = np.array(df_inv.index)
    df_inv['datex'] = df_inv['datex'].apply(lambda x: str(x).removeprefix('Ht_Prcnt99 '))
    min_date = min(df_inv['datex'])
    min_date = dt.datetime.strptime(min_date, '%Y-%m-%d')
    max_date = max(df_inv['datex'])
    max_date = dt.datetime.strptime(max_date, '%Y-%m-%d')
    return max_val_ht, min_date, max_date


# trait_extractor_date(r'Z:/035 Parul APTID/WGIN_UAV_data_16-20_120221.xlsx', sh_name='2018 Height Data', 
# val_at_mentioned_max_perc=95)
# img_loc='Z:/035 Parul APTID/ttttttesttttt')
