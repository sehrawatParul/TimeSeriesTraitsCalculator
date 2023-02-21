from func_file import gompertz_model, model2, find_nearest, percentile_method_ht
from sklearn.metrics import r2_score
from intersect import intersection
from scipy.stats import linregress
import time
import pandas as pd
import numpy as np
import datetime as dt
import pwlf
import scipy.optimize as opt
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('qtagg')
warnings.filterwarnings("ignore")
random.seed(20)


def trait_extractor_thermal(input_file, sh_name=None, img_loc=None, mentioned_th_time=None, base_temp_given=None, 
val_at_mentioned_max_perc=None, outlier_removal=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot')
    raw_df['Sowing date'] = raw_df['Sowing date'].apply(lambda x: str(x).removesuffix(' 00:00:00'))

    gdd_df = pd.read_excel(r'GDD spreadsheet 2015 to 01-2023.xlsx')

    gdd_df.drop(index=gdd_df.index[0], axis=0, inplace=True)
    gdd_df['day'] = gdd_df['day'].apply(lambda x: str(x).removesuffix(' 00:00:00'))

    df = raw_df.drop(columns=['ComonVar', 'Nrate', 'Pesticide', 'Sowing date', 'Replicate', 'Row', 'Column'])

    xyz = np.array(df.columns)
    temp_date = [str(i).replace('Ht_Prcnt99 ', '') for i in xyz]
    if base_temp_given:
        base_temp = float(base_temp_given)
    else:
        base_temp = 0
    gdd_df[gdd_df['Unnamed: 10'] < base_temp] = base_temp
    mean_temp_df = gdd_df[gdd_df['day'].isin(temp_date)]

    mean_temp_df_index_array = np.array(mean_temp_df.index)
    mean_temp_df_index_array_ = mean_temp_df_index_array-1
    
    full_thermal_df  = gdd_df[mean_temp_df_index_array_[0]:mean_temp_df_index_array_[-1]+1]
    
    temp_col = full_thermal_df['Unnamed: 10']

    first_record_idx = np.where(gdd_df['day'] == mean_temp_df['day'].iloc[0])
    first_record_idx = int(first_record_idx[0]) + 1
    start_index = mean_temp_df_index_array[0]
    idx_arr_short = mean_temp_df.index-start_index

    df_inv = df.T

    record_date = df_inv.index
    only_date = record_date[0].strip('Ht_Prcnt99 ')
    year_only = only_date[:4]

    df_inv['datex'] = np.array(record_date)
    df_inv['datex'] = df_inv['datex'].apply(lambda x: str(x).removeprefix('Ht_Prcnt99 '))

    df_inv['datex'] = pd.to_datetime(df_inv['datex'])
    df_inv['datex'] = df_inv['datex'].map(dt.datetime.toordinal)

    df_inv[df_inv < 0] = 0

    inflection_height1 = []
    inflection_time1 = []
    slope_at_inflection1 = []
    max_slope1 = []
    max_slope_height1 = []
    max_slope_time1 = []
    max_height1 = []
    max_height_time1 = []
    fitness_score_model1 = []
    execution_time1 = []

    inflection_height2 = []
    inflection_time2 = []
    slope_at_inflection_m2 = []
    max_slope2 = []
    max_slope_height2 = []
    max_slope_time2 = []
    max_height2 = []
    max_height_time2 = []
    fitness_score_model2 = []
    execution_time2 = []

    height_declining = []

    mentioned_th_time_arr = []
    val_at_mentioned_max_perc_arr1 = []

    gom_guess = [38.7, 6.5, 0.9]

    fig_j = 1
    # len(df_inv.columns)
    for ij in range(1, len(df_inv.columns)):
        
        sowing_date = raw_df['Sowing date'].loc[ij]
        sowing_idx = np.where(gdd_df['day'] == sowing_date)
        sowing_idx = int(sowing_idx[0])
        diff_array = np.array(gdd_df['Unnamed: 10'][sowing_idx:first_record_idx])
        sum_diff = round(sum(diff_array), 1)

        diff = sum_diff
        thermal_array_full = []
        thermal_array_full.insert(0, sum_diff)
        for t in temp_col[1:]:
            diff += t
            thermal_array_full.append(round(diff, 1))
        thermal_array_full = np.array(thermal_array_full)
        thermal_array = np.array([thermal_array_full[x] for x in idx_arr_short])

        thermal_arr = np.array([x - sum_diff for x in thermal_array])  # starting from zero
        thermal_array_copy = thermal_array.copy()

        Y_t = df_inv[ij]
        Y_t_copy = Y_t.copy()
        nan_y = np.isnan(Y_t)
        not_nan_y = ~ nan_y
        thermal_arr = thermal_arr[not_nan_y]
        thermal_array = thermal_array[not_nan_y]
        Y_t = Y_t[not_nan_y]
        mymodel = np.poly1d(np.polyfit(thermal_arr, Y_t, 5))
        y_pred = mymodel(thermal_arr)

        if outlier_removal is not None:
            idx_y = percentile_method_ht(y_pred, Y_t)
            for k in idx_y:
                Y_t[k] = np.nan
            idx_y = (np.where(Y_t.isnull()))[0]
            for ix in idx_y:
                thermal_arr = np.where(np.isclose(thermal_arr, thermal_arr[ix]), np.nan, thermal_arr)
                thermal_array = np.where(np.isclose(thermal_array, thermal_array[ix]), np.nan, thermal_array)
            nan_X_t = np.isnan(thermal_arr)
            not_nan_X_t = ~ nan_X_t
            thermal_arr = thermal_arr[not_nan_X_t]
            nan_X = np.isnan(thermal_array)
            not_nan_X = ~ nan_X
            thermal_array = thermal_array[not_nan_X]
            nan_Y_t = np.isnan(Y_t)
            not_nan_Y_t = ~ nan_Y_t
            Y_t = Y_t[not_nan_Y_t]

        x_hat = np.arange(min(thermal_arr), max(thermal_arr) + 1, dtype=int) 
        x_hat2 = np.arange(min(thermal_array), max(thermal_array) + 1)

        # model1
        start1 = time.time()
        my_pwlf = pwlf.PiecewiseLinFit(thermal_arr, Y_t)
        breaks = my_pwlf.fit(3)
        ijk = breaks[2]

        bp = find_nearest(thermal_arr, ijk)
        idx_bp = np.where(thermal_arr == bp)
        idx_bp = int(idx_bp[0])
        linear_x = thermal_arr[idx_bp:]

        if len(linear_x) < 3:
            ijk = breaks[1]
            bp = find_nearest(thermal_arr, ijk)
            idx_bp = np.where(thermal_arr == bp)
            idx_bp = int(idx_bp[0])
            linear_x = thermal_arr[idx_bp:]
        slopes = my_pwlf.calc_slopes()

        my_pwlf4 = pwlf.PiecewiseLinFit(thermal_arr, Y_t)
        breaks4 = my_pwlf4.fit(4)
        ijk4 = breaks4[3]

        bp4 = find_nearest(thermal_arr, ijk4)
        idx_bp4 = np.where(thermal_arr == bp4)
        idx_bp4 = int(idx_bp4[0])
        linear_x4 = thermal_arr[idx_bp4:]
        slopes4 = my_pwlf4.calc_slopes()

        if slopes[2] > slopes4[3] and len(linear_x4) > 3:
            gom_x = thermal_arr[:idx_bp4 + 1]
            gom_plot = thermal_array[:idx_bp4 + 1]
            gom_y = Y_t[:idx_bp4 + 1]
            lin_x = linear_x4
            lin_y = my_pwlf4.predict(lin_x)
            lin_plot = thermal_array[idx_bp4:]
            grad_lin_y = np.gradient(lin_y, lin_x)
            if round(grad_lin_y[0], 6) > round(grad_lin_y[-1], 6):
                gom_x = thermal_arr[:idx_bp4 + 2]
                gom_plot = thermal_array[:idx_bp4 + 2]
                gom_y = Y_t[:idx_bp4 + 2]
                lin_x = thermal_arr[idx_bp4 + 1:]
                lin_y = my_pwlf4.predict(lin_x)
                lin_plot = thermal_array[idx_bp4 + 1:]
        else:
            gom_x = thermal_arr[:idx_bp + 1]
            gom_plot = thermal_array[:idx_bp + 1]
            gom_y = Y_t[:idx_bp + 1]
            lin_x = linear_x
            lin_y = my_pwlf.predict(lin_x)
            if len(lin_x) < 2:
                grad_lin_y = [0]
            else:
                grad_lin_y = np.gradient(lin_y, lin_x)
            lin_plot = thermal_array[idx_bp:]
            if round(grad_lin_y[0], 6) > round(grad_lin_y[-1], 6):
                gom_x = thermal_arr[:idx_bp + 2]
                gom_plot = thermal_array[:idx_bp + 2]
                gom_y = Y_t[:idx_bp + 2]
                lin_x = thermal_arr[idx_bp + 1:]
                lin_y = my_pwlf.predict(lin_x)
                lin_plot = thermal_array[idx_bp + 1:]

        gom_x_cont = np.arange(min(gom_x), max(gom_x) + 1, 0.1, dtype=float)
        gom_plot_cont = np.arange(min(gom_plot), max(gom_plot) + 1, 0.1, dtype=float)
        
        if len(lin_x) < 2:
            sl, intr = 0, 0
        else:
            sl, intr, rval, pval, stdr = linregress(lin_x, lin_y)
        y_plot = x_hat * sl + intr
        y_plot_X_t = thermal_arr * sl + intr

        if sl < 0:
            height_declining.append('y')
        else:
            height_declining.append('n')

        popt, _ = opt.curve_fit(gompertz_model, gom_x, gom_y, gom_guess, method="trf", maxfev=1000000)
        ai, bi, ci = popt

        y_fit_gom = gompertz_model(gom_x, *popt)
        y_fit_gom_X_t = gompertz_model(thermal_arr, *popt)
        y_gom_plot = gompertz_model(x_hat, *popt)
        y_fit_gom_cont = gompertz_model(gom_x_cont, *popt)

        x_intersect, y_intersect = intersection(x_hat, y_plot, x_hat, y_gom_plot)

        if y_intersect.size == 0:
            x_intersect = lin_x[0]
            y_intersect = lin_y[0]
            intersect_constant = 0
        elif y_intersect.size > 1:
            x_intersect = x_intersect[0]
            y_intersect = y_intersect[0]
            intersect_constant = 1
        else:
            intersect_constant = 1

        round_gom_x = np.round(gom_x)
        idx_x = np.where(x_hat == round_gom_x[-1])
        idx_x = int(idx_x[0])
        x_hat_gom = x_hat[:idx_x]
        y_fit_gom1 = gompertz_model(x_hat_gom, *popt)
        slop_gom_val = np.gradient(y_fit_gom1)
        max_gom_slope_idx = np.where(slop_gom_val == max(slop_gom_val))[0]
        if len(max_gom_slope_idx) > 1:
            max_gom_slope_idx = max_gom_slope_idx[0]
        max_gom_slope_idx = int(max_gom_slope_idx)
        max_slope = slop_gom_val[max_gom_slope_idx]
        time_max_slope = x_hat2[max_gom_slope_idx]
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
        time_first_inflection = x_hat2[inflec_idx]
        first_inflection_pt = y_fit_gom1[inflec_idx]
        slp_at_inflection = grad_array[inflec_idx]

        idx_intersect_time = find_nearest(thermal_arr, x_intersect)
        idx_intersect_time = int(np.where(thermal_arr == idx_intersect_time)[0])
        lin_start_idx = int(len(thermal_arr) - idx_intersect_time - 1)
        a1 = np.array(y_fit_gom_X_t[:idx_intersect_time + 1])
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
        poptt, _ = opt.curve_fit(model2, thermal_arr, Y_t, initial_guess, method="trf", maxfev=1000000)
        y_fit_model2 = model2(thermal_arr, *poptt)
        y_model2_full = model2(x_hat, *poptt)

        # model 2 traits
        def model2_traits():
            max_ht2 = max(y_model2_full)
            idx_max_ht = int(np.where(y_model2_full == max_ht2)[0])
            max_ht_time2 = x_hat2[idx_max_ht]
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
                    inflect_time_m2 = x_hat2[idx_inflec_grad_m2]
                    slp_at_inflection_m2 = inflection_pt_m2[idx_inflec_grad_m2]
                except:
                    inflec_ht_m2 = np.nan
                    inflect_time_m2 = np.nan
                    slp_at_inflection_m2 = np.nan
                    inflect_constant_m2 = np.nan
            else:
                inflec_ht_m2 = np.nan
                inflect_time_m2 = np.nan
                slp_at_inflection_m2 = np.nan
                inflect_constant_m2 = np.nan

            max_growth_m2 = max(grad_growth_arr_m2)
            idx_comp = np.where(grad_growth_arr_m2 == max_growth_m2)[0]
            if len(idx_comp) > 1:
                idx_comp = idx_comp[0]
            idx_max_growth_m2 = int(idx_comp)
            max_slope_ht2 = y_model2_full[idx_max_growth_m2]
            max_slope_tm2 = x_hat2[idx_max_growth_m2]
            if max_ht2 == y_model2_full[-1]:
                max_ht2 = "{}{}".format(max_ht2, '*')
                max_ht_time2 = "{}{}".format(max_ht_time2, '*')
            if inflect_constant_m2 == 1:
                inflec_ht_m2 = "{}{}".format(inflec_ht_m2, '*')
                inflect_time_m2 = "{}{}".format(inflect_time_m2, '*')
                slp_at_inflection_m2 = "{}{}".format(slp_at_inflection_m2, '*')

            return inflec_ht_m2, inflect_time_m2, slp_at_inflection_m2, max_growth_m2, max_slope_ht2, max_slope_tm2, max_ht2, max_ht_time2

        r2_score_model2 = round(r2_score(Y_t, y_fit_model2), 4)

        if r2_score_model2 > r2_score_model1:
            inflec_ht_m2, inflect_time_m2, slp_at_inflection_m2, max_growth_m2, max_slope_ht2, max_slope_tm2, max_ht2, max_ht_time2 = model2_traits()
            inflection_height2.append(inflec_ht_m2)
            inflection_time2.append(inflect_time_m2)
            slope_at_inflection_m2.append(slp_at_inflection_m2)
            max_slope2.append(max_growth_m2)
            max_slope_height2.append(max_slope_ht2)
            max_slope_time2.append(max_slope_tm2)
            max_height2.append(max_ht2)
            max_height_time2.append(max_ht_time2)
            fitness_score_model2.append(r2_score_model2)

        else:
            inflection_height2.append(np.nan)
            inflection_time2.append(np.nan)
            slope_at_inflection_m2.append(np.nan)
            max_slope2.append(np.nan)
            max_slope_height2.append(np.nan)
            max_slope_time2.append(np.nan)
            max_height2.append(np.nan)
            max_height_time2.append(np.nan)
            fitness_score_model2.append(np.nan)
        end2 = time.time()
        time_model2 = round((end2 - start2), 4)
        execution_time2.append(time_model2)

        def inter_dt(ip):
            pt_1 = int(np.round(ip))
            idx_pt_1 = np.where(x_hat == pt_1)
            idx_pt_1 = int(idx_pt_1[0])
            intersect_time = x_hat2[idx_pt_1]
            return intersect_time

        x_intersect = inter_dt(x_intersect)
        y_intersect = np.float(y_intersect)

        if val_at_mentioned_max_perc:
            if y_intersect:
                perc_ht_val1 = (float(val_at_mentioned_max_perc) * y_intersect)/100
                val_at_mentioned_max_perc_arr1.append(perc_ht_val1)
            else:
                val_at_mentioned_max_perc_arr1.append('')

        if mentioned_th_time:
            mentioned_th_time = float(mentioned_th_time)
            mentioned_th_t = mentioned_th_time - sum_diff
            if mentioned_th_time < x_intersect:
                ht_from_tht = gompertz_model(mentioned_th_t, *popt)
            else:
                ht_from_tht = mentioned_th_t * sl + intr
            mentioned_th_time_arr.append(ht_from_tht)

        if img_loc:
            plt.figure(ij, figsize=(10, 9))
            plt.plot(thermal_array_copy, Y_t_copy, 'go')
            plt.plot(thermal_array, Y_t, 'ro')
            plt.plot(gom_plot_cont, y_fit_gom_cont, 'b')
            plt.plot(lin_plot, lin_y, 'b')
            if r2_score_model2 > r2_score_model1:
                plt.plot(x_hat2, y_model2_full, 'y')
            plt.plot(x_intersect, y_intersect, "*k")
            plt.xticks(thermal_array_copy)
            plt.tick_params(labelrotation=90)
            labl1 = raw_df["ComonVar"].iloc[ij - 1]
            labl2 = raw_df["Nrate"].iloc[ij - 1]
            labl4 = raw_df["Replicate"].iloc[ij - 1]
            labl3 = raw_df["Pesticide"].iloc[ij - 1]
            if r2_score_model2 > r2_score_model1:
                plt.title(f"Year: {year_only}, Variety: {labl1},  Nrate: {labl2}, Rep: {labl4}, Pesticide: {labl3}, \n fit-score & time (blue): {r2_score_model1} & {time_model1}, fit-score & time (yellow): {r2_score_model2} & {time_model2}")
            else:
                plt.title(f"Year: {year_only}, Variety: {labl1},  Nrate: {labl2}, Rep: {labl4}, Pesticide: {labl3}, \n fit-score & time (blue): {r2_score_model1} & {time_model1}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_thermal.jpg')
            # plt.show()
            plt.clf()
            fig_j = fig_j + 1
        
        if intersect_constant == 0:
            y_intersect = "{}{}".format(y_intersect, '*')
            x_intersect = "{}{}".format(x_intersect, '*')

        if inflect_constant == 1:
            first_inflection_pt = "{}{}".format(first_inflection_pt, '*')
            time_first_inflection = "{}{}".format(time_first_inflection, '*')
            slp_at_inflection = "{}{}".format(slp_at_inflection, '*')

        inflection_height1.append(first_inflection_pt)
        inflection_time1.append(time_first_inflection)
        slope_at_inflection1.append(slp_at_inflection)
        max_slope1.append(max_slope)
        max_slope_height1.append(ht_max_slope)
        max_slope_time1.append(time_max_slope)
        max_height1.append(y_intersect)
        max_height_time1.append(x_intersect)
   
    output_traits = pd.DataFrame()
    output_traits['inflect_Ht_m1'] = inflection_height1
    output_traits['inflect_time_m1'] = inflection_time1
    output_traits['slope_at_inflection1'] = slope_at_inflection1
    output_traits['max_Slp_m1'] = max_slope1
    output_traits['max_Slp_Ht_m1'] = max_slope_height1
    output_traits['max_Slp_Dt_time1'] = max_slope_time1
    output_traits['max_Ht_m1'] = max_height1
    output_traits['max_Ht_time_m1'] = max_height_time1
    output_traits['fitness_score_model1'] = fitness_score_model1
    output_traits['execution_time1'] = execution_time1
    output_traits['height_declining'] = height_declining
    output_traits['inflect_Ht_m2'] = inflection_height2
    output_traits['inflect_time_m2'] = inflection_time2
    output_traits['slope_at_inflection_m2'] = slope_at_inflection_m2
    output_traits['max_Slp_m2'] = max_slope2
    output_traits['max_Slp_Ht_m2'] = max_slope_height2
    output_traits['max_Slp_time_m2'] = max_slope_time2
    output_traits['max_Ht_m2'] = max_height2
    output_traits['max_Ht_time_m2'] = max_height_time2
    output_traits['execution_time2'] = execution_time2

    if mentioned_th_time_arr:
        output_traits[f'height_at_{mentioned_th_time}'] = mentioned_th_time_arr
    
    if val_at_mentioned_max_perc_arr1:
        output_traits[f'height_at_{val_at_mentioned_max_perc}_perc_maxHeight'] = val_at_mentioned_max_perc_arr1

    if fitness_score_model2:
        output_traits['fitness_score_model2'] = fitness_score_model2


    df_info = raw_df[['ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column']]
    idx = np.array(df.index.values)
    output_traits.set_index(idx, inplace=True)
    output_df = df_info.join(output_traits)
    return output_df


def thermal_properties(input_file, sh_name=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot')
    raw_df['Sowing date'] = raw_df['Sowing date'].apply(lambda x: str(x).removesuffix(' 00:00:00'))
    gdd_df = pd.read_excel(r'GDD spreadsheet 2015 to 01-2023.xlsx')
    gdd_df.drop(index=gdd_df.index[0], axis=0, inplace=True)
    gdd_df['day'] = gdd_df['day'].apply(lambda x: str(x).removesuffix(' 00:00:00'))
    df = raw_df.drop(columns=['ComonVar', 'Nrate', 'Pesticide', 'Sowing date', 'Replicate', 'Row', 'Column'])
    xyz = np.array(df.columns)
    temp_date = [str(i).replace('Ht_Prcnt99 ', '') for i in xyz]
    mean_temp_df = gdd_df[gdd_df['day'].isin(temp_date)]
    # temp_col = mean_temp_df['Unnamed: 10']
    mean_temp_df_index_array = np.array(mean_temp_df.index)
    mean_temp_df_index_array_ = mean_temp_df_index_array-1
    
    full_thermal_df  = gdd_df[mean_temp_df_index_array_[0]:mean_temp_df_index_array_[-1]+1]
    
    temp_col = full_thermal_df['Unnamed: 10']

    first_record_idx = np.where(gdd_df['day'] == mean_temp_df['day'].iloc[0])
    first_record_idx = int(first_record_idx[0]) + 1
    start_index = mean_temp_df_index_array[0]
    idx_arr_short = mean_temp_df.index-start_index
    df_inv = df.T
    df_inv['datex'] = np.array(df_inv.index)
    df_inv['datex'] = df_inv['datex'].apply(lambda x: str(x).removeprefix('Ht_Prcnt99 '))
    df_inv['datex'] = pd.to_datetime(df_inv['datex'])
    df_inv['datex'] = df_inv['datex'].map(dt.datetime.toordinal)
    df_inv[df_inv < 0] = 0

    for ij in range(1, len(df_inv.columns)):
        sowing_date = raw_df['Sowing date'].loc[ij]
        sowing_idx = np.where(gdd_df['day'] == sowing_date)
        sowing_idx = int(sowing_idx[0])
        diff_array = np.array(gdd_df['Unnamed: 10'][sowing_idx:first_record_idx])
        sum_diff = sum(diff_array)
        diff = sum_diff
        thermal_array_full = []
        thermal_array_full.insert(0, sum_diff)
        for t in temp_col[1:]:
            diff += t
            thermal_array_full.append(round(diff, 1))
        thermal_array_full = np.array(thermal_array_full)
    thermal_var = thermal_array_full
    min_val = min(thermal_var)
    max_val = max(thermal_var)
    return min_val, max_val

# trait_extractor_thermal(r'Z:/035 Parul APTID/WGIN_UAV_data_16-20_120221.xlsx', 
# input_file_grassroot = r'Z:/013 Data/Phenotyping/Field Data 2016/Grassroots 2016/DIV2016.xlsx',
# sh_name='2017 Height Data')
# img_loc=r'Z:/035 Parul APTID/ttttttesttttt'
# outlier_removal=1)
