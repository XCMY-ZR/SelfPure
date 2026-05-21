import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from cycler import cycler


custom_cycler = cycler(color=[ "#4a2377","#8cc5e3","#f55f74","#0d7d87" ]) 

def GR_confusion_compare_matrix(data):
    #     print("Raw result") [A&R,A&nR,nA&R,nA&nR]
    tmpdata = data[data.true_label == data.raw_clean_pred_label]
    tp = len(tmpdata[tmpdata.raw_clean_pred_label == tmpdata.raw_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label == data.raw_clean_pred_label]
    tn = len(tmpdata[tmpdata.raw_clean_pred_label != tmpdata.raw_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label != data.raw_clean_pred_label]
    fp = len(tmpdata[tmpdata.raw_clean_pred_label == tmpdata.raw_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label != data.raw_clean_pred_label]
    fn = len(tmpdata[tmpdata.raw_clean_pred_label != tmpdata.raw_adv_pred_label]) / len(data)

    #TP TF 
    rawstr = np.array([tp, tn, fp, fn])

    #     print("Random smoothing result") [A&R,A&nR,nA&R,nA&nR]
    tmpdata = data[data.true_label == data.rand_clean_pred_label]
    tp = len(tmpdata[tmpdata.rand_clean_pred_label == tmpdata.rand_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label == data.rand_clean_pred_label]
    tn = len(tmpdata[tmpdata.rand_clean_pred_label != tmpdata.rand_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label != data.rand_clean_pred_label]
    fp = len(tmpdata[tmpdata.rand_clean_pred_label == tmpdata.rand_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label != data.rand_clean_pred_label]
    fn = len(tmpdata[tmpdata.rand_clean_pred_label != tmpdata.rand_adv_pred_label]) / len(data)

    randstr = np.array([tp, tn, fp, fn])

    #     print("Adv smoothing result") [A&R,A&nR,nA&R,nA&nR]
    tmpdata = data[data.true_label == data.pro_clean_pred_label]
    tp = len(tmpdata[tmpdata.pro_clean_pred_label == tmpdata.pro_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label == data.pro_clean_pred_label]
    tn = len(tmpdata[tmpdata.pro_clean_pred_label != tmpdata.pro_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label != data.pro_clean_pred_label]
    fp = len(tmpdata[tmpdata.pro_clean_pred_label == tmpdata.pro_adv_pred_label]) / len(data)

    tmpdata = data[data.true_label != data.pro_clean_pred_label]
    fn = len(tmpdata[tmpdata.pro_clean_pred_label != tmpdata.pro_adv_pred_label]) / len(data)

    prostr = np.array([tp, tn, fp, fn])

    return rawstr, randstr, prostr

def adv_f1_score(data):
    #data [tp,tn,fp, fn]
    acc_score = data[0]/(data[0]+data[1])
    robust_score = data[0]/(data[0]+data[2])
    f1_score = (2*acc_score*robust_score)/(acc_score+robust_score)
    return (1-data[3])*f1_score

def targeted_attack_analysis(data,classnum=10):
        #data:run eval_target_xxxx.py to get csv result
    repeat_all_num = len(data)
    all_num = repeat_all_num/classnum

    print('clean x targeted attack success:',sum(data['repeat_clean_label']==data["repeat_true_label"])/repeat_all_num)
    print('adv x targeted attack success:',sum(data['repeat_adv_label']==data["repeat_true_label"])/repeat_all_num)
    print('clean x after targered acc:',sum(data['repeat_clean_label']==data["ture_label"])/repeat_all_num)
    print('adv x after targered acc:',sum(data['repeat_adv_label']==data["ture_label"])/repeat_all_num)
    print('clean x acc:',sum(data['y_clean_label']==data["ture_label"])/repeat_all_num)
    print('adv x acc:',sum(data['y_adv_label']==data["ture_label"])/repeat_all_num)
    
    true_label = data.ture_label.to_numpy().reshape(-1,classnum)[:,0]
    
    delta_norm = data.x_delta_norm.to_numpy().reshape(-1,classnum)
    print('argmax delta for clean x:',sum(delta_norm.argmax(axis=1)== true_label)/all_num)
        
    delta_norm = data.x_adv_delta_norm.to_numpy().reshape(-1,classnum)
    print('argmax delta for adv x:',sum(delta_norm.argmax(axis=1)== true_label)/all_num)
    
