# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 13:11:40 2018

dataset: three xlsx files
    file 1: HT_20171031_MJZ1_Extension(20180312).xlsx
    file 2: HT_data_merge_aminoAcid_raw.xlsx
    file 3: HT_data_Pirche.xlsx

task 1: read xlsx file into dataframe using python pandas and extract the following columns respectively,
    file 1: ID + non-days-related columns
    file 2: ID + columns from 'AM_A other mismatch' to 'AM_DPB1 other mismatch', and 'D_A_aminoAcid_n24' to the end
    file 3: ID + PIRCHE_I_num + PIRCHE_II_num
    
task 2:
    merge the extracted data into one dataframe based on ID, then save the dataframe to csv file

task 3:
    find out categorical features and continuous features
    
task 4:
    select a few features and do some statistics for them, such as missing rate, mean, median.
    plot for continuous features
    
@author: xbli
"""

import pandas as pd
#import numpy as np
#import os
from datetime import datetime

#import xlrd
#from pandas import ExcelWriter
#from pandas import ExcelFile
from collections import Counter


''' get non-days-related columns '''
def data_processing(path, data_file_list):
    
    # =================  file 1 : ignore all date related data
    
    df_1 = pd.read_excel(path + data_file_list[0], 'Sheet1',skiprows=[0])
    cols = [c for c in df_1.columns if ('day' not in c.lower() and 'date' not in c.lower() and 'tx' not in c.lower()) ]
    df_1 = df_1[cols]
   
    # =================  file 2 : slice two seperate ranges of cols 
    
    ''' copy to seperate parts from excel 2 : 'AM_A' ~ 'AM_DPB1' AND 'D_A_aminoAcid_n24' ~ end '''
    
    df_2 = pd.read_excel(path + data_file_list[1], 'Sheet1')
    columns_data2 = df_2.columns.tolist()
    columns_1 = columns_data2[columns_data2.index('AM_A other mismatch'):columns_data2.index('AM_DPB1 other mismatch')]
    columns_2 = columns_data2[columns_data2.index('D_A_aminoAcid_n24'):]
    cols2 = ['ID'] + columns_1 + columns_2
    df_2 = df_2[cols2]
    
    # ==================  file 3 : only select two cols
    
    df_3 = pd.read_excel(path + data_file_list[2], 'Sheet1')
    cols3 = ['ID','PIRCHE_I_num', 'PIRCHE_II_num']
    df_3 = df_3[cols3]
    
    return df_1, df_2, df_3


''' 将血型拆开 A/A -> 患者 A/ 供者 A, 有一个数据有两个 / 所以要特别进行清洗 split 两次 '''
def seperate_patient_donor_blood_type(df_data):
    df = df_data.copy()
    
    patient = []
    donor = []
    
    # fill na
    df['BLD_D_A'] = df['BLD_D_A'].fillna('none/none') 
   
    patient = [i.split('/', 2)[0] for i in df['BLD_D_A'].tolist()]
    donor = [i.split('/', 2)[1] for i in df['BLD_D_A']]
    
    # 将 患者 A/ 供者 A 两列 加入output 表格, 删除 BLD_D_A
    df_data = {'BLD_D_A_patient':patient, 'BLD_D_A_donor':donor, 'ID': df['ID']}
    df_patient_donor = pd.DataFrame.from_dict(df_data)

    df = df.drop(['BLD_D_A'], axis=1)
    df = pd.merge(df, df_patient_donor, how='outer', on='ID')
    return df


''' combine 3 个 data frame， 组合数据 '''
def merge_df(df_1,df_2,df_3, w4):
    
    df = pd.merge(pd.merge(df_1, df_2, how='outer', on='ID'), df_3, how='outer', on='ID')
    
    # 将血型拆开
    df = seperate_patient_donor_blood_type(df)
    
    df.to_excel(w4, 'Sheet1', index=False)
    w4.save()
    
    # write out to csv
    pd.read_excel(w4).to_csv('result_out_0720.csv', index=False)
    
    return df

''' 将数据分离成 cat data / cot data / discrete data， 其中 cot 和 discrete 合为 numerical data'''
def cat_cot_dis_data(df, threshold=10):
    
    # to dictionary: count each element's frequency
    cols = df.columns.tolist()
    dic_2d = {}
    df_excel_temp = df.fillna('none')
    
    # 看每一个 column 里面出现的类型
    for i in range(2, len(cols)):
        dic_2d[cols[i]] = dict(Counter(df_excel_temp[cols[i]].tolist()))
     
    # 看每一个 column 包含了几个
    rank_touple = [(k,len(dic_2d[k])) for k in sorted(dic_2d, key=lambda k: len(dic_2d[k]), reverse=True)]
    print('选以下几个（+hla）当做数值型变量：', rank_touple[:4])
    # amino acid 应该是 categorical 的，所以以amino acid 作为分界线, 得出max size = 7 
    #max_size = max([len(dic_2d[i].keys()) for i in dic_2d.keys() if 'aminoAcid' in i])

    # 分类 categorical feature / continuous feature / discrete feature 
    col_cat = []
    col_cot = []
    col_dis = []
    
    for i in dic_2d.keys():
        # discrete data
        if 'hla' in i.lower() or 'DRIGP' == i:
            col_dis.append(i)
        # continuous data
        elif len(dic_2d[i].keys()) > threshold:
            col_cot.append(i)
        # categorical data
        else:
            # if only has one element, we can skip it
            if len(dic_2d[i].keys()) > 1:
                col_cat.append(i)
                
    return col_cat, col_cot, col_dis


''' missing rate / mean/ median / plot '''
def missing_mean_median_plot(df):
    print('*****************')
    # 看所有columns 的 missing rate, 1.1 是因为如果写1 ，有一些column 会少算
    missing_rate_list = missing_value_selection(df, df.columns, 1.1)
    mean_dic = df.mean().to_dict()
    median_dic = df.median().to_dict()
    
    #print ('missing rate is: ', missing_rate_dict)
    #print ('mean is: ', mean_dic)
    #print ('median is: ', median_dic)
    
    i = 'PIRCHE_II_num'
    ax = df[i].plot.hist(title='plot for ' + i, bins = 5)
    
    ax.set_xlabel('sample order')
    ax.set_ylabel(i)
    return missing_rate_list, mean_dic, median_dic

''' 只return missing value 小于阈值的 columns'''
def missing_value_selection(df_excel, selected_cols, threshold=0.5):
    
    df_excel_temp = df_excel.copy()
    df_excel_temp = df_excel_temp[selected_cols]
    missing_rate_dict = df_excel_temp.apply(lambda x: pd.isnull(x).sum()/df_excel_temp.shape[0],axis=0).to_dict()
    
    return [key for key,value in missing_rate_dict.items() if value < threshold]
    

if __name__ == '__main__':
    print(datetime.now())
    path = '/Users/DihuaDuolan/Desktop/BFR/task1To4/'
    data_file_list = ['data/HT_20171031_MJZ1_Extension(20180312).xlsx' #
                     ,'data/HT_data_merge_aminoAcid_raw.xlsx' #
                     ,'data/HT_data_Pirche.xlsx' #
                     ]
                     
    ''' write to new files '''
    writer = 'output_combined.xlsx'
    w = pd.ExcelWriter(path + writer)
    
    ''' 数据处理：去掉和日期相关的 columns/ 选择 amino acid columns / 选择 Pirche'''
    print('数据处理: ', datetime.now())
    df_1, df_2, df_3 = data_processing(path, data_file_list)
    
    ''' 将上一个结果的数据结合到一个 data frmae (df)  里面'''
    print('数据整合: ', datetime.now())              
    df = merge_df(df_1,df_2,df_3, w)
    
    ''' 将数据分离成 cat data / cot data / discrete data， 其中 cot 和 discrete 合为 numerical data'''
    print('数据分类: ', datetime.now())
    col_cat, col_num_cot, col_num_dis = cat_cot_dis_data(df)
    
    ''' 求每个column 的 missing rate'''
    print('求缺失率: ', datetime.now())
    missing_rate_list, mean_dic, median_dic = missing_mean_median_plot(df)
    
    print(datetime.now())
    
    pass
