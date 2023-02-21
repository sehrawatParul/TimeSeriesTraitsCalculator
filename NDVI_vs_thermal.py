from func_file import remove_init_vals, percentile_method, double_logi, remove_last_vals
from sklearn.metrics import r2_score
import time
import pandas as pd
import numpy as np
import scipy.optimize as opt
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib
import datetime as dt

matplotlib.use('qtagg')
warnings.filterwarnings("ignore")
random.seed(20)


def NDVI_trait_extractor_thermal(input_file, ndvi_method, sh_name=None, img_loc=None, data_output_file=None,
mentioned_th_time=None, base_temp_given=None, 
val_at_mentioned_max_perc=None, outlier_removal=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot ID')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot ID')

    raw_df = raw_df[raw_df.index.notnull()]
    raw_df = raw_df[raw_df['PECO:0007167']=='SFP']   
    raw_df['Sowing date'] = raw_df['Sowing date'].apply(lambda x: str(x).removesuffix(' 00:00:00'))
    raw_df_cols = raw_df.columns

    if ndvi_method=='Sample 1 - 25m altitude S900':
        sample1_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_1', raw_df_cols))
        df = raw_df[sample1_cols]
        filtered_sample1_cols = list(map(lambda x: x.split(' ')[-2], sample1_cols))
        df.rename(columns=dict(zip(sample1_cols, filtered_sample1_cols)), inplace=True)
        df_inv = df.T
    
    elif ndvi_method=='Sample 2 - 12m altitude S900':
        sample2_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_2', raw_df_cols))
        df = raw_df[sample2_cols]
        filtered_sample2_cols = list(map(lambda x: x.split(' ')[-2], sample2_cols))
        df.rename(columns=dict(zip(sample2_cols, filtered_sample2_cols)), inplace=True)
        df_inv = df.T

    elif ndvi_method=='Sample 3 - M600':
        sample3_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_3', raw_df_cols))
        df = raw_df[sample3_cols]
        filtered_sample3_cols = list(map(lambda x: x.split(' ')[-2], sample3_cols))
        df.rename(columns=dict(zip(sample3_cols, filtered_sample3_cols)), inplace=True)
        df_inv = df.T
    
    else:        
        sampletec5_cols = list(filter(lambda col_nm: 'HHInd' in col_nm.split('_')[-1], raw_df_cols))
        df = raw_df[sampletec5_cols]
        filtered_tec5_cols = list(map(lambda x: x.split(' ')[-1], sampletec5_cols))
        df.rename(columns=dict(zip(sampletec5_cols, filtered_tec5_cols)), inplace=True)
        df_inv = df.T
    
    gdd_df = pd.read_excel(r'GDD spreadsheet 2015 to 01-2023.xlsx')
    gdd_df.drop(index=gdd_df.index[0], axis=0, inplace=True)
    gdd_df['day'] = gdd_df['day'].apply(lambda x: str(x).removesuffix(' 00:00:00'))
    
    if base_temp_given:
        base_temp = float(base_temp_given)
    else:
        base_temp = 0
    
    gdd_df[gdd_df['Unnamed: 10'] < base_temp] = base_temp
    mean_temp_df = gdd_df[gdd_df['day'].isin(df_inv.index)]
    
    mean_temp_df_index_array = np.array(mean_temp_df.index)
    mean_temp_df_index_array_ = mean_temp_df_index_array-1
    
    full_thermal_df  = gdd_df[mean_temp_df_index_array_[0]:mean_temp_df_index_array_[-1]+1]
    
    temp_col = full_thermal_df['Unnamed: 10']

    first_record_idx = np.where(gdd_df['day'] == mean_temp_df['day'].iloc[0])
    first_record_idx = int(first_record_idx[0]) + 1
    start_index = mean_temp_df_index_array[0]
    idx_arr_short = mean_temp_df.index-start_index

    year_only = df_inv.index[0][:4]
    df_inv.index = pd.to_datetime(df_inv.index)

    guess_logi = np.array([0, 0.8, 769, 1507.9,187.9, 161])
    guess_logi1 = np.array([0.2, 0.5, 841.2, 1494, 161.5, 134])
    guess_logi2 = np.array([0, 0.8, 1006.6, 1271, 194.7, 170.9])

    inflection_NDVI = []
    inflection_time = []
    slope_at_inflection = []
    max_slope = []
    max_slope_NDVI = []
    max_slope_time = []
    max_NDVI = []
    max_NDVI_time = []
    fitness_score = []
    execution_time = []

    mentioned_th_time_arr = []
    val_at_mentioned_max_perc_arr1 = []
    
    fig_j = 1

    for ij in df_inv.columns:
        sowing_date = raw_df['Sowing date'][ij]
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
        thermal_array = np.array([thermal_array_full[x] for x in idx_arr_short])

        thermal_arr = np.array([x - sum_diff for x in thermal_array])  # starting from zero
        thermal_array_copy = thermal_array.copy()
        
        Y_t = df_inv[ij]
        Y_t_copy = Y_t
        nan_Y_t = pd.isna(Y_t)
        not_nan_Y_t = ~ nan_Y_t
        thermal_arr = thermal_arr[not_nan_Y_t]
        thermal_array = thermal_array[not_nan_Y_t]
        Y_t = Y_t.dropna()
        

        idx_y_init = remove_init_vals(Y_t)
        idx_y_last = remove_last_vals(Y_t)
        idx_y = idx_y_init+idx_y_last
        
        if idx_y:
            for k in idx_y:
                Y_t[k] = np.nan
                thermal_arr[k] = np.nan
                thermal_array[k] = np.nan

        nan_X_t = np.isnan(thermal_arr)
        not_nan_X_t = ~ nan_X_t
        thermal_arr = thermal_arr[not_nan_X_t]
        nan_X = np.isnan(thermal_array)
        not_nan_X = ~ nan_X
        thermal_array = thermal_array[not_nan_X]
        nan_Y_t = pd.isna(Y_t)
        not_nan_Y_t = ~ nan_Y_t
        Y_t = Y_t[not_nan_Y_t]
        Y_t = Y_t.astype(float)
        thermal_arr = thermal_arr.astype(float)
        
        mymodel = np.poly1d(np.polyfit(thermal_arr, Y_t, 5))
        y_pred = mymodel(thermal_arr)
        
        if outlier_removal is not None:
            idx_y = percentile_method(y_pred, Y_t)
            for k in idx_y:
                Y_t[k] = np.nan
                thermal_arr[k] = np.nan
                thermal_array[k] = np.nan

            nan_X_t = np.isnan(thermal_arr)
            not_nan_X_t = ~ nan_X_t
            thermal_arr = thermal_arr[not_nan_X_t]
            nan_X = np.isnan(thermal_array)
            not_nan_X = ~ nan_X
            thermal_array = thermal_array[not_nan_X]
            nan_Y_t = np.isnan(Y_t)
            not_nan_Y_t = ~ nan_Y_t
            Y_t = Y_t[not_nan_Y_t]
        
        thermal_arr_hat = np.arange(min(thermal_arr), max(thermal_arr) + 1, dtype=int)
        thermal_array_hat = np.arange(min(thermal_array), max(thermal_array) + 1, dtype=int)
        start = time.time()
        poptt, _ = opt.curve_fit(double_logi, thermal_arr, Y_t, guess_logi, method="trf", maxfev=1000000)
        y_fit_logi = double_logi(thermal_arr, *poptt)

        poptt1, _ = opt.curve_fit(double_logi, thermal_arr, Y_t, guess_logi1, method="trf", maxfev=1000000)
        y_fit_logi1 = double_logi(thermal_arr, *poptt1)

        poptt2, _ = opt.curve_fit(double_logi, thermal_arr, Y_t, guess_logi2, method="trf", maxfev=1000000)
        y_fit_logi2 = double_logi(thermal_arr, *poptt2)

        r2_double_logi = r2_score(Y_t, y_fit_logi)
        r2_double_logi1 = r2_score(Y_t, y_fit_logi1)
        r2_double_logi2 = r2_score(Y_t, y_fit_logi2)

        list_double_logi = [r2_double_logi, r2_double_logi1, r2_double_logi2]
        max_fit = max(list_double_logi)
        idxxxx = np.where(list_double_logi == max_fit)[0]
        if len(idxxxx)!=1:
            index_max_fit = int(idxxxx[0])
        else:
            index_max_fit = int(idxxxx)
        poptt_list = [poptt, poptt1, poptt2]
        final_popt = poptt_list[index_max_fit]
        y_logi_final = double_logi(thermal_arr_hat, *final_popt)
        y_fittt = double_logi(thermal_arr, *final_popt)
        r2_double_final = round(r2_score(Y_t, y_fittt), 4)

        fitness_score.append(r2_double_final)

        a_max_NDVI = max(y_logi_final)
        max_NDVI.append(a_max_NDVI)

        if val_at_mentioned_max_perc:
            perc_ht_val1 = (float(val_at_mentioned_max_perc) * a_max_NDVI)/100
            val_at_mentioned_max_perc_arr1.append(perc_ht_val1)

        idx_max_NDVI = np.where(y_logi_final == a_max_NDVI)[0][0]
        a_max_NDVI_time = thermal_array_hat[idx_max_NDVI]
        max_NDVI_time.append(a_max_NDVI_time)

        growth_logi = y_logi_final[:idx_max_NDVI]
        if len(growth_logi) <= 1:
            max_slope.append('')
            max_slope_NDVI.append('')
            max_slope_time.append('')
            inflection_NDVI.append('')
            inflection_time.append('')
            slope_at_inflection.append('')
        else:
            slope_ndvi = np.gradient(growth_logi)
            max_slope_ndvi = max(slope_ndvi)
            max_slope.append(max_slope_ndvi)
            idx_max_slope = np.where(slope_ndvi == max_slope_ndvi)[0][0]
            max_slope_n = y_logi_final[idx_max_slope]
            max_slope_NDVI.append(max_slope_n)
            time_max_slp = thermal_array_hat[idx_max_slope]
            max_slope_time.append(time_max_slp)

            double_grad = np.gradient(slope_ndvi)
            inflect_pt = max(double_grad)
            idx_inflect_pt = np.where(double_grad == inflect_pt)[0][0]
            inflect_ndvi = y_logi_final[idx_inflect_pt]
            inflection_NDVI.append(inflect_ndvi)
            inflect_time = thermal_array_hat[idx_inflect_pt]
            inflection_time.append(inflect_time)
            slp_at_inflection = double_grad[idx_inflect_pt]
            slope_at_inflection.append(slp_at_inflection)

        if mentioned_th_time:
            mentioned_th_time = float(mentioned_th_time)
            mentioned_th_t = mentioned_th_time - sum_diff
            NDVI_from_tht = double_logi(mentioned_th_t, *final_popt)
            mentioned_th_time_arr.append(NDVI_from_tht)
        end = time.time()
        tot_time = round((end-start), 4)
        execution_time.append(tot_time)

        if img_loc is not None:
            plt.figure(ij, figsize=(10, 9))
            plt.plot(thermal_array_copy, Y_t_copy, 'go')
            plt.plot(thermal_array, Y_t, 'ro')
            plt.plot(thermal_array_hat, y_logi_final, label='Double Logistic')
            plt.xticks(thermal_array_copy, rotation='vertical')
            labl1 = raw_df["Accession"][int(ij)]     # ComonVar
            labl2 = raw_df["PECO:0007102"][int(ij)]    # Nrate  # .iloc[ij - 1]
            labl3 = raw_df["PECO:0007167"][int(ij)]    # Pesticide
            labl4 = raw_df["Replicate"][int(ij)]
            plt.title(f"Year: {year_only}, Variety: {labl1},  Nrate: {labl2}, Rep: {labl4}, Pesticide: {labl3}, fit-score & time: {r2_double_final} & {tot_time}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            # plt.show()
            if ndvi_method=='Sample 1 - 25m altitude S900':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_thermal_Sample1.jpg')
            elif ndvi_method=='Sample 2 - 12m altitude S900':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_thermal_Sample2.jpg')
            elif ndvi_method=='Sample 3 - M600':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_thermal_Sample3.jpg')
            else:
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_thermal_tec5.jpg')
            plt.clf()
            fig_j = fig_j + 1
        
    output_traits = pd.DataFrame()
    output_traits['inflection_NDVI'] = inflection_NDVI
    output_traits['inflection_time'] = inflection_time
    output_traits['slope_at_inflection'] = slope_at_inflection
    output_traits['max_slope'] = max_slope
    output_traits['max_slope_NDVI'] = max_slope_NDVI
    output_traits['max_slope_time'] = max_slope_time
    output_traits['max_NDVI'] = max_NDVI
    output_traits['max_NDVI_time'] = max_NDVI_time
    output_traits['fitness_score'] = fitness_score
    output_traits['execution_time'] = execution_time

    if mentioned_th_time_arr:
        output_traits[f'NDVI_at_{mentioned_th_time}'] = mentioned_th_time_arr

    if val_at_mentioned_max_perc_arr1:
        output_traits[f'NDVI_at_{val_at_mentioned_max_perc}_perc_maxNDVI'] = val_at_mentioned_max_perc_arr1


    df_info = raw_df[['Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167']] # 'ComonVar', 'Nrate', 'Pesticide', 'Replicate', 'Row', 'Column'
    idx = np.array(df.index.values)
    output_traits.set_index(idx, inplace=True)
    output_df = df_info.join(output_traits)
    # output_df.to_excel(data_output_file, sheet_name='NDVI_vs_thermal_tec5_2022')
    return output_df


def NDVI_thermal_properties(input_file, ndvi_method, sh_name=None, base_temp_given=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot ID')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot ID')
        raw_df = raw_df[raw_df.index.notnull()]
    raw_df = raw_df[raw_df['PECO:0007167']=='SFP']   
    raw_df['Sowing date'] = raw_df['Sowing date'].apply(lambda x: str(x).removesuffix(' 00:00:00'))
    raw_df_cols = raw_df.columns

    if ndvi_method=='Sample 1 - 25m altitude S900':
        sample1_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_1', raw_df_cols))
        df = raw_df[sample1_cols]
        filtered_sample1_cols = list(map(lambda x: x.split(' ')[-2], sample1_cols))
        df.rename(columns=dict(zip(sample1_cols, filtered_sample1_cols)), inplace=True)
        df_inv = df.T
    
    elif ndvi_method=='Sample 2 - 12m altitude S900':
        sample2_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_2', raw_df_cols))
        df = raw_df[sample2_cols]
        filtered_sample2_cols = list(map(lambda x: x.split(' ')[-2], sample2_cols))
        df.rename(columns=dict(zip(sample2_cols, filtered_sample2_cols)), inplace=True)
        df_inv = df.T

    elif ndvi_method=='Sample 3 - M600':
        sample3_cols = list(filter(lambda col_nm: col_nm.split(' ')[-1]=='sample_3', raw_df_cols))
        df = raw_df[sample3_cols]
        filtered_sample3_cols = list(map(lambda x: x.split(' ')[-2], sample3_cols))
        df.rename(columns=dict(zip(sample3_cols, filtered_sample3_cols)), inplace=True)
        df_inv = df.T
    
    else:        
        sampletec5_cols = list(filter(lambda col_nm: 'HHInd' in col_nm.split('_')[-1], raw_df_cols))
        df = raw_df[sampletec5_cols]
        filtered_tec5_cols = list(map(lambda x: x.split(' ')[-1], sampletec5_cols))
        df.rename(columns=dict(zip(sampletec5_cols, filtered_tec5_cols)), inplace=True)
        df_inv = df.T
    
    gdd_df = pd.read_excel(r'GDD spreadsheet 2015 to 01-2023.xlsx')
    gdd_df.drop(index=gdd_df.index[0], axis=0, inplace=True)
    gdd_df['day'] = gdd_df['day'].apply(lambda x: str(x).removesuffix(' 00:00:00'))
    
    if base_temp_given:
        base_temp = float(base_temp_given)
    else:
        base_temp = 0
    
    gdd_df[gdd_df['Unnamed: 10'] < base_temp] = base_temp
    mean_temp_df = gdd_df[gdd_df['day'].isin(df_inv.index)]

    mean_temp_df_index_array = np.array(mean_temp_df.index)
    mean_temp_df_index_array_ = mean_temp_df_index_array-1
    
    full_thermal_df  = gdd_df[mean_temp_df_index_array_[0]:mean_temp_df_index_array_[-1]+1]
    
    temp_col = full_thermal_df['Unnamed: 10']

    first_record_idx = np.where(gdd_df['day'] == mean_temp_df['day'].iloc[0])
    first_record_idx = int(first_record_idx[0]) + 1

    df_inv.index = pd.to_datetime(df_inv.index)

    for ij in df_inv.columns:
        sowing_date = raw_df['Sowing date'][ij]
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


# NDVI_trait_extractor_thermal(r'Y:/2022_Images/M600/Diversity_Extracted_Data/Indices/WW2213_Diversity_Compiled_4_methods_Data.xlsx',
#  sh_name='2022 NDVI Data compiled +means',
#  ndvi_method='HH - tec 5',
#  img_loc=r'Z:/035 Parul APTID/new_NDVI_sample_images/thermal_time',
#  data_output_file='Z:/035 Parul APTID/new_NDVI_sample_sheets/NDVI_thermal_output_tec5_2022.xlsx')

# min_val, max_val = NDVI_thermal_properties('Z:/035 Parul APTID/WGIN_UAV_data_16-20_120221.xlsx', sh_name='2016 NDVI Data')
