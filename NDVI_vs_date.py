from func_file import date_from_index_NDVI, remove_init_vals, find_nearest, percentile_method, double_logi, remove_last_vals
from sklearn.metrics import r2_score
from datetime import timedelta
import time
import pandas as pd
import numpy as np
import datetime as dt
import scipy.optimize as opt
import warnings
import matplotlib.pyplot as plt
import matplotlib
import random

matplotlib.use('qtagg')
warnings.filterwarnings("ignore")
random.seed(20)


def NDVI_trait_extractor_date(input_file, ndvi_method, sh_name=None, img_loc=None, data_output_file=None, mentioned_ht=None, mentioned_cal_dt=None,
                              mentioned_perc_maxht=None, mentioned_perc_maxslp_dt=None,
                              mentioned_perc_maxslp_NDVI=None, val_at_mentioned_max_perc=None, outlier_removal=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot ID') # Plot
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot ID')
    
    raw_df = raw_df[raw_df.index.notnull()]
    raw_df = raw_df[raw_df['PECO:0007167']=='SFP']
    raw_df_cols = raw_df.columns
    plot_details_cols = list(filter(lambda col_nm: 'NDVI' not in col_nm.split('_'), raw_df_cols))
    if ndvi_method=='Sample 1 - 25m altitude S900':
        sample1_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_1', raw_df_cols))
        combined_cols = plot_details_cols + sample1_cols
        sample1_raw_df = raw_df[combined_cols]
        df = raw_df[sample1_cols]
        filtered_sample1_cols = list(map(lambda x: x.split(' ')[-2], sample1_cols))
        df.rename(columns=dict(zip(sample1_cols, filtered_sample1_cols)), inplace=True)
        df_inv = df.T

    elif ndvi_method=='Sample 2 - 12m altitude S900':
        sample2_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_2', raw_df_cols))
        combined_cols = plot_details_cols + sample2_cols
        sample2_raw_df = raw_df[combined_cols]
        df = raw_df[sample2_cols]
        filtered_sample2_cols = list(map(lambda x: x.split(' ')[-2], sample2_cols))
        df.rename(columns=dict(zip(sample2_cols, filtered_sample2_cols)), inplace=True)
        df_inv = df.T


    elif ndvi_method=='Sample 3 - M600':
        sample3_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_3', raw_df_cols))
        combined_cols = plot_details_cols + sample3_cols
        sample3_raw_df = raw_df[combined_cols]
        df = raw_df[sample3_cols]
        filtered_sample3_cols = list(map(lambda x: x.split(' ')[-2], sample3_cols))
        df.rename(columns=dict(zip(sample3_cols, filtered_sample3_cols)), inplace=True)
        df_inv = df.T
    
    else:        
        sampletec5_cols = list(filter(lambda col_nm: 'HHInd' in col_nm.split('_')[-1], raw_df_cols))
        combined_cols = plot_details_cols + sampletec5_cols
        tec5_raw_df = raw_df[combined_cols]
        df = raw_df[sampletec5_cols]
        filtered_tec5_cols = list(map(lambda x: x.split(' ')[-1], sampletec5_cols))
        df.rename(columns=dict(zip(sampletec5_cols, filtered_tec5_cols)), inplace=True)
        df_inv = df.T

    year_only = df_inv.index[0][:4]
    df_inv.index = pd.to_datetime(df_inv.index)
    df_inv['datex'] = df_inv.index.map(dt.datetime.toordinal)

    dt_x_copy = df_inv.index

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

    guess_logi = np.array([0.4, 0.4, 77, 105.8, 3.8, 7])
    guess_logi1 = np.array([0.2, 0.6, 119, 77.8, 13.7, 6.8])
    guess_logi2 = np.array([0.3, 0.59, 80, 130.9, 23.8, 7.5])

    inflection_NDVI = []
    inflection_date = []
    slope_at_inflection = []
    max_slope = []
    max_slope_NDVI = []
    max_slope_date = []
    max_NDVI = []
    max_NDVI_date = []
    fitness_score = []
    execution_time = []

    dt_from_NDVI_arr = []
    NDVI_from_dt_arr = []
    mentioned_perc_maxNDVI_arr = []

    dt_at_mentioned_perc_max_slope_arr = []
    NDVI_at_mentioned_perc_max_slope_arr = []
    val_at_mentioned_max_perc_arr1 = []

    loop_cols = df_inv.columns
    loop_cols = loop_cols[:-1]

    fig_j = 1

    for ij in loop_cols:
        Y_t = df_inv[ij]
        Y_t_orig = Y_t
        X_t, X_t_mod = get_x(Y_t)
        Y_t = Y_t.dropna()

        idx_y_init = remove_init_vals(Y_t)
        idx_y_last = remove_last_vals(Y_t)
        idx_y = idx_y_init + idx_y_last

        if idx_y:
            for k in idx_y:
                Y_t[k] = np.nan
            idx_y = (np.where(Y_t.isnull()))[0]
            for ix in idx_y:
                X_t = np.where(np.isclose(X_t, X_t[ix]), np.nan, X_t)

        nan_X_t = np.isnan(X_t)
        not_nan_X_t = ~ nan_X_t
        X_t = X_t[not_nan_X_t]
        nan_Y_t = pd.isna(Y_t)
        not_nan_Y_t = ~ nan_Y_t
        Y_t = Y_t[not_nan_Y_t]
        Y_t = Y_t.astype(float)
        X_t = X_t.astype(float)

        mymodel = np.poly1d(np.polyfit(X_t, Y_t, 5))
        y_pred = mymodel(X_t)
        
        if outlier_removal is not None:
            idx_y = percentile_method(y_pred, Y_t)
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
        
        dt_x = Y_t.index
        
        date_arr = []
        delta = dt_x[-1] - dt_x[0]  # returns timedelta
        for i in range(delta.days + 1):
            day = dt_x[0] + timedelta(days=i)
            date_arr.append(day)

        x_hat = np.arange(min(X_t), max(X_t) + 1, dtype=int)
        start = time.time()
        poptt, _ = opt.curve_fit(double_logi, X_t, Y_t, guess_logi, method="trf", maxfev=1000000)
        y_fit_logi = double_logi(X_t, *poptt)

        poptt1, _ = opt.curve_fit(double_logi, X_t, Y_t, guess_logi1, method="trf", maxfev=1000000)
        y_fit_logi1 = double_logi(X_t, *poptt1)

        poptt2, _ = opt.curve_fit(double_logi, X_t, Y_t, guess_logi2, method="trf", maxfev=1000000)
        y_fit_logi2 = double_logi(X_t, *poptt2)

        r2_double_logi = r2_score(Y_t, y_fit_logi)
        r2_double_logi1 = r2_score(Y_t, y_fit_logi1)
        r2_double_logi2 = r2_score(Y_t, y_fit_logi2)

        list_double_logi = [r2_double_logi, r2_double_logi1, r2_double_logi2]
        max_fit = max(list_double_logi)
        index_max_fit = int(np.where(list_double_logi == max_fit)[0])
        poptt_list = [poptt, poptt1, poptt2]
        final_popt = poptt_list[index_max_fit]
        y_logi_final = double_logi(x_hat, *final_popt)
        y_fittt = double_logi(X_t, *final_popt)
        r2_double_final = round(r2_score(Y_t, y_fittt), 4)

        fitness_score.append(r2_double_final)

        a_max_NDVI_val = max(y_logi_final)
        max_NDVI.append(a_max_NDVI_val)

        if val_at_mentioned_max_perc:
            perc_ht_val1 = (float(val_at_mentioned_max_perc) * a_max_NDVI_val)/100
            val_at_mentioned_max_perc_arr1.append(perc_ht_val1)

        idx_max_NDVI = np.where(y_logi_final == a_max_NDVI_val)[0][0]
        date_max_ndvi = date_arr[idx_max_NDVI]
        max_NDVI_date.append(date_max_ndvi)

        growth_logi_y = y_logi_final[:idx_max_NDVI]
        if len(growth_logi_y) <= 1:
            max_slope.append('')
            max_slope_date.append('')
            max_slope_NDVI.append('')
            inflection_date.append('')
            inflection_NDVI.append('')
            slope_at_inflection.append('')
            if mentioned_perc_maxslp_dt:
                dt_at_mentioned_perc_max_slope_arr.append('')
            if mentioned_perc_maxslp_NDVI:
                NDVI_at_mentioned_perc_max_slope_arr.append('')
        else:
            slope_vals = np.gradient(growth_logi_y)
            max_slope_pt = max(slope_vals)
            max_slope.append(max_slope_pt)
            idx_max_slope_pt = np.where(slope_vals == max_slope_pt)[0][0]
            max_slope_d = date_arr[idx_max_slope_pt]
            max_slope_date.append(max_slope_d)
            max_slope_ndvi = y_logi_final[idx_max_slope_pt]
            max_slope_NDVI.append(max_slope_ndvi)

            double_grad = np.gradient(slope_vals)
            max_double_grad = max(double_grad)
            idx_max_double_grad = np.where(double_grad == max_double_grad)[0][0]
            inflection_d = date_arr[idx_max_double_grad]
            inflection_date.append(inflection_d)
            inflection_ndvi = y_logi_final[idx_max_double_grad]
            inflection_NDVI.append(inflection_ndvi)
            slp_at_inflection = double_grad[idx_max_double_grad]
            slope_at_inflection.append(slp_at_inflection)

            if mentioned_perc_maxslp_dt:
                slp_at_perc = (max_slope_pt * float(mentioned_perc_maxslp_dt)) / 100
                nearest_slp_val = find_nearest(slope_vals, slp_at_perc)
                idx_nearest_slp_val = np.where(slope_vals == nearest_slp_val)[0][0]
                dt_slp = int(x_hat[idx_nearest_slp_val]) + X_t_mod[0]
                dt_slp = dt.date.fromordinal(dt_slp)
                dt_at_mentioned_perc_max_slope_arr.append(dt_slp)

            if mentioned_perc_maxslp_NDVI:
                slp_at_perc = (max_slope_pt * float(mentioned_perc_maxslp_NDVI)) / 100
                nearest_slp_val = find_nearest(slope_vals, slp_at_perc)
                idx_nearest_slp_val = np.where(slope_vals == nearest_slp_val)[0][0]
                NDVI_perc_maxslope = y_logi_final[idx_nearest_slp_val]
                NDVI_at_mentioned_perc_max_slope_arr.append(NDVI_perc_maxslope)

        if mentioned_ht:
            x_from_NDVI = np.interp([mentioned_ht], y_logi_final, x_hat)
            x_from_NDVI = int(x_from_NDVI[0]) + X_t_mod[0]
            dt_from_NDVI = dt.date.fromordinal(x_from_NDVI)
            dt_from_NDVI_arr.append(dt_from_NDVI)

        if mentioned_cal_dt:
            mentioned_cal_dt = dt.datetime.strptime(str(mentioned_cal_dt), '%Y-%m-%d').date()
            x_ordinal = dt.datetime.toordinal(mentioned_cal_dt)
            x_diff = x_ordinal - X_t_mod[0]
            NDVI_from_dt = double_logi(x_diff, *final_popt)
            NDVI_from_dt_arr.append(NDVI_from_dt)

        if mentioned_perc_maxht:
            mentioned_perc_maxNDVI = float(mentioned_perc_maxht)
            NDVI_at_perc = (mentioned_perc_maxNDVI * a_max_NDVI_val)/100
            dt_of_perc_NDVI = np.interp([NDVI_at_perc], y_logi_final, x_hat)
            dt_of_perc_NDVI = int(dt_of_perc_NDVI[0]) + X_t_mod[0]
            dt_of_perc_NDVI = dt.date.fromordinal(dt_of_perc_NDVI)
            mentioned_perc_maxNDVI_arr.append(dt_of_perc_NDVI)
        end = time.time()
        tot_time = round((end-start), 4)
        execution_time.append(tot_time)

        if img_loc is not None:
            plt.figure(ij, figsize=(10, 9))
            plt.plot(dt_x_copy, Y_t_orig, 'go')
            plt.plot(dt_x, Y_t, 'ro')
            plt.plot(date_arr, y_logi_final, label='Double Logistic')
            plt.xticks(dt_x_copy, rotation='vertical')
            labl1 = raw_df["Accession"][int(ij)]     # ComonVar
            labl2 = raw_df["PECO:0007102"][int(ij)]    # Nrate  # .iloc[ij - 1]
            labl3 = raw_df["PECO:0007167"][int(ij)]    # Pesticide
            labl4 = raw_df["Replicate"][int(ij)]
            plt.title(f"Year:{year_only}, Variety: {labl1},  Nrate: {labl2}, Rep: {labl4}, Pesticide: {labl3},\n fit-score & time: {r2_double_final} & {tot_time}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            if ndvi_method=='Sample 1 - 25m altitude S900':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_date_Sample1.jpg')
            elif ndvi_method=='Sample 2 - 12m altitude S900':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_date_Sample2.jpg')
            elif ndvi_method=='Sample 3 - M600':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_date_Sample3.jpg')
            else: 
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_date_tec5.jpg')
            plt.clf()
            fig_j = fig_j + 1
    
    output_traits = pd.DataFrame()
    output_traits['inflection_NDVI'] = inflection_NDVI
    output_traits['inflection_date'] = inflection_date
    output_traits['slope_at_inflection'] = slope_at_inflection
    output_traits['max_slope'] = max_slope
    output_traits['max_slope_NDVI'] = max_slope_NDVI
    output_traits['max_slope_date'] = max_slope_date
    output_traits['max_NDVI'] = max_NDVI
    output_traits['max_NDVI_date'] = max_NDVI_date
    output_traits['fitness_score'] = fitness_score
    output_traits['execution_time'] = execution_time

    if dt_from_NDVI_arr:
        output_traits[f'date_at_{mentioned_ht}'] = dt_from_NDVI_arr
    if NDVI_from_dt_arr:
        output_traits[f'height_at_{mentioned_cal_dt}'] = NDVI_from_dt_arr
    if mentioned_perc_maxNDVI_arr:
        output_traits[f'date_at_{mentioned_perc_maxht}_perc_NDVI'] = mentioned_perc_maxNDVI_arr
    if dt_at_mentioned_perc_max_slope_arr:
        output_traits[f'date_at_{mentioned_perc_maxslp_dt}_perc_maxslope'] = dt_at_mentioned_perc_max_slope_arr
    if NDVI_at_mentioned_perc_max_slope_arr:
        output_traits[f'NDVI_at_{mentioned_perc_maxslp_NDVI}_perc_maxslope'] = NDVI_at_mentioned_perc_max_slope_arr
    if val_at_mentioned_max_perc_arr1:
        output_traits[f'NDVI_at_{val_at_mentioned_max_perc}_perc_maxNDVI'] = val_at_mentioned_max_perc_arr1

    df_info = raw_df[['Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167']]
    idx = np.array(df.index.values)
    output_traits.set_index(idx, inplace=True)
    output_df = df_info.join(output_traits)
    # output_df.to_excel(data_output_file, sheet_name='NDVI_vs_dt_tec5_2022')

    return output_df
    

def NDVI_properties(input_file, ndvi_method, sh_name=None):

    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot ID')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot ID')

    raw_df = raw_df[raw_df.index.notnull()]
    raw_df = raw_df[raw_df['PECO:0007167']=='SFP']
    raw_df_cols = raw_df.columns
    plot_details_cols = list(filter(lambda col_nm: 'NDVI' not in col_nm.split('_'), raw_df_cols))
    if ndvi_method=='Sample 1 - 25m altitude S900':
        sample1_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_1', raw_df_cols))
        combined_cols = plot_details_cols + sample1_cols
        df = raw_df[sample1_cols]
        filtered_sample1_cols = list(map(lambda x: x.split(' ')[-2], sample1_cols))
        df.rename(columns=dict(zip(sample1_cols, filtered_sample1_cols)), inplace=True)
        df_inv = df.T

    
    elif ndvi_method=='Sample 2 - 12m altitude S900':
        sample2_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_2', raw_df_cols))
        combined_cols = plot_details_cols + sample2_cols
        df = raw_df[sample2_cols]
        filtered_sample2_cols = list(map(lambda x: x.split(' ')[-2], sample2_cols))
        df.rename(columns=dict(zip(sample2_cols, filtered_sample2_cols)), inplace=True)
        df_inv = df.T

    elif ndvi_method=='Sample 3 - M600':
        sample3_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_3', raw_df_cols))
        combined_cols = plot_details_cols + sample3_cols
        df = raw_df[sample3_cols]
        filtered_sample3_cols = list(map(lambda x: x.split(' ')[-2], sample3_cols))
        df.rename(columns=dict(zip(sample3_cols, filtered_sample3_cols)), inplace=True)
        df_inv = df.T
    
    else:        
        sampletec5_cols = list(filter(lambda col_nm: 'HHInd' in col_nm.split('_')[-1], raw_df_cols))
        combined_cols = plot_details_cols + sampletec5_cols
        df = raw_df[sampletec5_cols]
        filtered_tec5_cols = list(map(lambda x: x.split(' ')[-1], sampletec5_cols))
        df.rename(columns=dict(zip(sampletec5_cols, filtered_tec5_cols)), inplace=True)
        df_inv = df.T
    
    df_inv = df.T
    max_val_ndvi = df_inv.max().max()
    df_inv.index = pd.to_datetime(df_inv.index)
    df_inv['datex'] = df_inv.index.map(dt.datetime.toordinal)
    min_date = min(df_inv['datex'])
    min_date = dt.datetime.strptime(min_date, '%Y-%m-%d')
    max_date = max(df_inv['datex'])
    max_date = dt.datetime.strptime(max_date, '%Y-%m-%d')
    return max_val_ndvi, min_date, max_date


# NDVI_trait_extractor_date(r'Y:/2022_Images/M600/Diversity_Extracted_Data/Indices/WW2213_Diversity_Compiled_4_methods_Data.xlsx', 
# ndvi_method='HH - tec 5', 
# sh_name='2022 NDVI Data compiled +means',
# img_loc=r'Z:/035 Parul APTID/Blank_folder',
# data_output_file='Z:/035 Parul APTID/NDVI_output_tec5_2022.xlsx')
