from func_file import remove_init_vals, percentile_method, double_logi, remove_last_vals
from sklearn.metrics import r2_score
import time
import pandas as pd
import numpy as np
import datetime as dt
import scipy.optimize as opt
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('qtagg')

warnings.filterwarnings("ignore")
random.seed(20)


def NDVI_trait_extractor_das(input_file, ndvi_method, sh_name=None, img_loc=None, mentioned_das=None, 
val_at_mentioned_max_perc=None, outlier_removal=None):

    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot ID')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot ID')
    
    raw_df['Sowing_date_ordinal'] = np.array(raw_df['Sowing date'])
    raw_df['Sowing_date_ordinal'] = raw_df['Sowing_date_ordinal'].map(dt.datetime.toordinal)
    sowing_date_list = raw_df['Sowing_date_ordinal']

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

    # df_inv[df_inv < 0] = 0
    X_t = np.array(df_inv['datex'])

    first_recorded_date = X_t[0]
    arr_diff = np.array(first_recorded_date - sowing_date_list) # raw_df['Sowing_date_ordinal']

    def orig_X_t():
        X_t = np.array(df_inv['datex'])
        DAS_var = []
        addition = 0
        DAS_var.insert(0, 0)
        for i in range(1, len(X_t)):
            yy = X_t[i] - X_t[i - 1]
            yyy = addition + yy
            addition = yyy
            DAS_var.append(yyy)
        X_t = np.array(DAS_var)
        return X_t

    def get_x(y):
        test_df = df_inv[y.notna()]
        X_t = np.array(test_df['datex'])
        DAS_var = []
        addition = 0
        DAS_var.insert(0, 0)
        for i in range(1, len(X_t)):
            yy = X_t[i] - X_t[i - 1]
            yyy = addition + yy
            addition = yyy
            DAS_var.append(yyy)

        X_t = np.array(DAS_var)
        return X_t

    guess_logi = np.array([0.4, 0.4, 77, 105.8, 3.8, 7])
    guess_logi1 = np.array([0.2, 0.6, 119, 77.8, 13.7, 6.8])
    guess_logi2 = np.array([0.3, 0.59, 80, 130.9, 23.8, 7.5])

    inflection_NDVI = []
    inflection_day = []
    slope_at_inflection = []
    max_slope_val = []
    max_slope_NDVI = []
    max_slope_day = []
    max_NDVI = []
    max_NDVI_day = []
    fitness_score = []
    execution_time = []

    mentioned_das_arr = []
    val_at_mentioned_max_perc_arr1 = []

    loop_cols = df_inv.columns
    loop_cols = loop_cols[:-1]
    loop_cols = list(map(lambda x: int(x), loop_cols))

    fig_j = 1
    for ij in loop_cols:
        diff_days = arr_diff[ij - 1]
        Y_t = df_inv[ij]
        Y_t_orig = Y_t
        X_t = get_x(Y_t)
        X_t_copy = orig_X_t() + diff_days
        Y_t = Y_t.dropna()

        idx_y_init = remove_init_vals(Y_t)
        idx_y_last = remove_last_vals(Y_t)
        idx_y = idx_y_init+idx_y_last

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
        index_max_fit = np.where(list_double_logi == max_fit)[0][0]

        poptt_list = [poptt, poptt1, poptt2]
        final_popt = poptt_list[index_max_fit]
        y_fittt = double_logi(X_t, *final_popt)
        y_logi_final = double_logi(x_hat, *final_popt)
        r2_double_final = round(r2_score(Y_t, y_fittt), 4)

        fitness_score.append(r2_double_final)

        a_max_NDVI_val = max(y_logi_final)
        max_NDVI.append(a_max_NDVI_val)

        if val_at_mentioned_max_perc:
            perc_ht_val1 = (float(val_at_mentioned_max_perc) * a_max_NDVI_val)/100
            val_at_mentioned_max_perc_arr1.append(perc_ht_val1)

        idx_max_NDVI = np.where(y_logi_final == a_max_NDVI_val)[0][0]
        day_max_NDVI = x_hat[idx_max_NDVI]+diff_days
        max_NDVI_day.append(day_max_NDVI)

        growth_logi_y = y_logi_final[:idx_max_NDVI+1]
        if len(growth_logi_y) <= 1:
            max_slope_val.append('')
            max_slope_NDVI.append('')
            max_slope_day.append('')
            inflection_NDVI.append('')
            inflection_day.append('')
            slope_at_inflection.append('')
        else:
            slope_vals = np.gradient(growth_logi_y)
            max_slope_pt = max(slope_vals)
            max_slope_val.append(max_slope_pt)
            idx_max_slope_pt = np.where(slope_vals == max_slope_pt)[0][0]
            max_slope_d = x_hat[idx_max_slope_pt]+diff_days
            max_slope_day.append(max_slope_d)
            max_slope_ndvi = y_logi_final[idx_max_slope_pt]
            max_slope_NDVI.append(max_slope_ndvi)

            double_grad = np.gradient(slope_vals)
            max_double_grad = max(double_grad)
            idx_max_double_grad = np.where(double_grad == max_double_grad)[0][0]
            inflection_d = x_hat[idx_max_double_grad]+diff_days
            slp_at_inflection = double_grad[idx_max_double_grad]
            slope_at_inflection.append(slp_at_inflection)
            inflection_day.append(inflection_d)
            inflection_ndvi = y_logi_final[idx_max_double_grad]
            inflection_NDVI.append(inflection_ndvi)

        curve_X_t = X_t + diff_days
        curve_X_t_full = x_hat + diff_days

        if mentioned_das:
            mentioned_das = int(mentioned_das)
            if mentioned_das < diff_days or mentioned_das > max(curve_X_t):
                mentioned_das_arr.append('')
            else:
                ment_das = mentioned_das - diff_days
                NDVI_from_das = double_logi(ment_das, *final_popt)
                mentioned_das_arr.append(NDVI_from_das)
        
        end = time.time()
        tot_time = round((end - start), 4)
        execution_time.append(tot_time)
        
        if img_loc is not None:
            plt.figure(ij, figsize=(10, 9))
            plt.plot(X_t_copy, Y_t_orig, 'go')
            plt.plot(curve_X_t, Y_t, 'ro')
            plt.plot(curve_X_t_full, y_logi_final, label='Double Logistic')
            plt.xticks(X_t_copy, rotation='vertical')
            labl1 = raw_df["Accession"][int(ij)]     # ComonVar
            labl2 = raw_df["PECO:0007102"][int(ij)]    # Nrate  # .iloc[ij - 1]
            labl3 = raw_df["PECO:0007167"][int(ij)]    # Pesticide
            labl4 = raw_df["Replicate"][int(ij)]
            plt.title(f"Year: {year_only}, Variety: {labl1},  Nrate: {labl2}, Rep: {labl4}, Pesticide: {labl3}, \n fit-score & time: {r2_double_final} & {tot_time}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            # plt.show()
            if ndvi_method=='Sample 1 - 25m altitude S900':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_das_sample1.jpg')
            elif ndvi_method=='Sample 2 - 12m altitude S900':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_das_sample2.jpg')
            elif ndvi_method=='Sample 3 - M600':
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_das_sample3.jpg')
            else:
                plt.savefig(f'{img_loc}/fig{fig_j}plot{ij}_das_tec5.jpg')
            plt.clf()
            fig_j = fig_j + 1
    
    output_traits = pd.DataFrame()
    output_traits['inflection_NDVI'] = inflection_NDVI
    output_traits['inflection_Day'] = inflection_day
    output_traits['slope_at_inflection'] = slope_at_inflection
    output_traits['max_Slp'] = max_slope_val
    output_traits['max_Slp_NDVI'] = max_slope_NDVI
    output_traits['max_Slp_Day'] = max_slope_day
    output_traits['max_NDVI'] = max_NDVI
    output_traits['max_NDVI_Day'] = max_NDVI_day
    output_traits['fitness_score'] = fitness_score
    output_traits['execution_time'] = execution_time
    # df.style.applymap(add_color, subset = ['columnnames'])

    if mentioned_das_arr:
        output_traits[f'NDVI_at_{mentioned_das}'] = mentioned_das_arr

    if val_at_mentioned_max_perc_arr1:
        output_traits[f'NDVI_at_{val_at_mentioned_max_perc}_perc_maxNDVI'] = val_at_mentioned_max_perc_arr1


    df_info = raw_df[['Replicate', 'Accession', 'PECO:0007102', 'PECO:0007167']]
    idx = np.array(df.index.values)
    output_traits.set_index(idx, inplace=True)
    output_df = df_info.join(output_traits)
    return output_df


def NDVI_das_properties(input_file, ndvi_method, sh_name=None):
    if sh_name:
        raw_df = pd.read_excel(input_file, sheet_name=sh_name, index_col='Plot ID')
    else:
        raw_df = pd.read_excel(input_file, index_col='Plot ID')

    raw_df['Sowing_date_ordinal'] = np.array(raw_df['Sowing date'])
    raw_df['Sowing_date_ordinal'] = raw_df['Sowing_date_ordinal'].map(dt.datetime.toordinal)

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


    df_inv.index = pd.to_datetime(df_inv.index)
    df_inv['datex'] = df_inv.index.map(dt.datetime.toordinal)
    
    # df_inv[df_inv < 0] = 0
    X_t = np.array(df_inv['datex'])
    first_recorded_date = X_t[0]
    last_recorded_date = X_t[-1]
    arr_diff = np.array(first_recorded_date - raw_df['Sowing_date_ordinal'])
    arr_diff_last = np.array(last_recorded_date - raw_df['Sowing_date_ordinal'])
    record_start = min(arr_diff)
    record_date_last = max(arr_diff_last)
    return record_start, record_date_last


# NDVI_trait_extractor_das(r'Z:/035 Parul APTID/WGIN_UAV_data_16-20_120221.xlsx', sh_name='2016 NDVI Data', 
# ndvi_method='Sample 1 - 25m altitude S900') 
# img_loc=r'Z:/035 Parul APTID/ttttttesttttt')

# NDVI_trait_extractor_das(r'Y:/2022_Images/M600/Diversity_Extracted_Data/Indices/WW2213_Diversity_Compiled_4_methods_Data.xlsx', 
# ndvi_method='HH - tec 5', 
# sh_name='2022 NDVI Data compiled +means')
# img_loc=r'Z:/035 Parul APTID/Blank_folder',
# data_output_file='Z:/035 Parul APTID/NDVI_output_tec5_2022.xlsx')
