# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from Metric import *
import argparse
import os
import pandas as pd
import json

def compute_metric_one(data, delay=7, num_threshold=300):
    y_true = data['label'].tolist()
    y_pred = data['predict'].tolist()
    
    if len(y_true) == 0:
        return 0, 0, 0, 0, 0, 0

    adjust_y_pred, adjust_y_true = point_adjust(y_pred, y_true)
    
    score, precision, recall = calculate_f1(adjust_y_true, adjust_y_pred)

    # print('adjust F1 Score:', score)
    # print('Precision:', precision)
    # print('Recall:', recall)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    delay_y_pred = get_range_proba(y_pred, y_true, delay=delay)
    delay_score, delay_precision, delay_recall = calculate_f1(y_true, delay_y_pred)
    return score, precision, recall, delay_score, delay_precision, delay_recall


def compute_metric_multi(path, ignore=[], type_ignore=[], type_only = "", prefix=[], delay=7, num_threshold=400, compute_all=True):
    
    folders = os.listdir(path)
    all_datas = []
    sub_res = {}
    
    for folder in folders:
        
        if folder in ignore:
            continue
        if len(prefix) > 0:
            in_prefix = False
            for pre in prefix:
                if folder.startswith(pre):
                    in_prefix = True
                    break
            if not in_prefix:
                continue
        # print(folder)
        file_path = os.path.join(path, folder, 'predict.csv')
        log_path = os.path.join(path, folder, 'log.csv')
        
        if not os.path.exists(file_path):
            # print('file not exist:', file_path)
            continue
        data = pd.read_csv(file_path)
        log_data = pd.read_csv(log_path)
        if len(type_ignore) > 0:
            for t in type_ignore:
                data['predict'] = [0 if x == t else data['predict'][i] for i, x in enumerate(data['anomaly_type'])]
        
        if compute_all:    
            score, precision, recall, delay_score, delay_precision, delay_recall = compute_metric_one(data, delay=delay, num_threshold=num_threshold)
            
            if 'scores' in log_data.columns:
                n_sim = log_data['scores']
                p_sim = log_data['ad_scores']
                n_sim = [float(json.loads(x)[0]) for x in n_sim]
                p_sim = [0,0]
                # p_sim = [float(json.loads(x)) for x in p_sim]
                # 求平均
                n_sim_ave = np.mean(n_sim)
                p_sim_ave = np.mean(p_sim)
            else :
                n_sim_ave = 0
                p_sim_ave = 0
            
            sub_res[folder] = {
                'company': folder,
                'num': len(data),
                'label_anomaly_num': len(data[data['label'] == 1]),
                'predict_anomaly_num': len(data[data['predict'] == 1]),
                'best_f1': score,
                'best_precision': precision,
                'best_recall': recall,
                'delay_f1': delay_score,
                'delay_precision': delay_precision,
                'delay_recall': delay_recall,
                'n_sim_ave': n_sim_ave,
                'p_sim_ave': p_sim_ave,
            }
        # if sum(data['predict']) > num_threshold:
        #     data['predict'] = [0 for _ in data['predict']]
        data['predict'] = [0 if x > num_threshold else data['predict'][i] for i, x in enumerate(data['ad_len'])]
        
        all_datas.append(data)
    
    all_datas = pd.concat(all_datas)
    
    type_only = type_only.lower()
    if type_only != "":
        all_datas['anomaly_type'] = all_datas['anomaly_type'].str.lower()
        all_datas = all_datas[type_only == all_datas['anomaly_type']]
        all_datas = all_datas.reset_index(drop=True)
        if len(all_datas) == 0:
            return 0, 0, 0, 0, 0, 0, 0
        
    y_true = all_datas['label'].tolist()
    y_pred = all_datas['predict'].tolist()
    # print('len(y_true):', len(y_true))
    # adjust
    adjust_y_pred, adjust_y_true = point_adjust(y_pred, y_true)
    # adjust_y_pred, adjust_y_true = y_pred, y_true
    
    score, precision, recall = calculate_f1(adjust_y_true, adjust_y_pred)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    delay_y_pred = get_range_proba(y_pred, y_true, delay=delay)
    delay_score, delay_precision, delay_recall = calculate_f1(y_true, delay_y_pred)
    # event_f1 = EventF1PA()
    # event_score, event_precision, event_recall, event_threshold = event_f1.calc(y_pred, y_true, 0)
    # print('event_score:', event_score, 'event_precision:', event_precision, 'event_recall:', event_recall, 'event_threshold:', event_threshold)
    company = 'all'
    if type_only != "":
        company = type_only
    sub_res[company] = {
        'company': company,
        'num': len(all_datas),
        'label_anomaly_num': len(all_datas[all_datas['label'] == 1]),
        'predict_anomaly_num': len(all_datas[all_datas['predict'] == 1]),
        'best_f1': score,
        'best_precision': precision,
        'best_recall': recall,
        'delay_f1': delay_score,
        'delay_precision': delay_precision,
        'delay_recall': delay_recall,
        'n_sim_ave': 0,
        'p_sim_ave': 0,
    }
    res = pd.DataFrame.from_dict(sub_res, orient='index')
    if type_only != "":
        res.to_csv(os.path.join(path, 'metric.csv'), mode='a', index=False, header=False)
    else:
        res.to_csv(os.path.join(path, 'metric.csv'), index=False)
    if len(prefix) == 1:
        all_datas.to_csv(os.path.join(path, prefix[0] + '.csv'), index=False)
    if 'ad_len' in all_datas.columns:
        ad_len = max(all_datas['ad_len'])
    else:
        ad_len = 600
    return score, precision, recall, delay_score, delay_precision, delay_recall, ad_len


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--path', type=str, default="result/Yahoo_50_prompt_71_prompt72_new_imlicity",required=False)
    arg_parser.add_argument('--use_threshold', type=bool, default=True, required=False)
    arg_parser.add_argument('--compute_type', type=bool, default=False, required=False)
    return arg_parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    path = args.path

    type_ignore = []
    ignore = []
    yahoo_prefixes = ['real','A3','A4','s']
    for yahoo_prefix in yahoo_prefixes:
        prefix = []
        prefix.append(yahoo_prefix)
        # compute_metric_multi(path, ignore=ignore)
        score, precision, recall, delay_score, delay_precision, delay_recall, right = compute_metric_multi(path, prefix=prefix, ignore=ignore, delay=7, num_threshold=10000, type_ignore=type_ignore)
        print('raw score')
        print('adjust F1 Score:', score)
        print('Precision:', precision)
        print('Recall:', recall)
        print('delay F1 Score:', delay_score)
        print('delay Precision:', delay_precision)
        print('delay Recall:', delay_recall)
        
        

        # exit(0)
        if right > 400:
            right = 400
        
        if args.use_threshold:

            best_score, best_precision, best_recall, best_delay_score, best_delay_precision, best_delay_recall = score, precision, recall, delay_score, delay_precision, delay_recall
            num_threshold = 400
            left = 0
            # right = 80
            for i in range(1, right + 2):
                
                score, precision, recall, delay_score, delay_precision, delay_recall, _ = compute_metric_multi(path, prefix=prefix, ignore=ignore, delay=7, num_threshold=i, type_ignore=type_ignore)
                # print('i:', i, 'score:', score, 'best_score:', best_score, 'recall:', recall, 'precision:', precision, 'delay_score:', delay_score, 'delay_precision:', delay_precision, 'delay_recall:', delay_recall)        
                if score >= best_score:
                    best_score, best_precision, best_recall, best_delay_score, best_delay_precision, best_delay_recall = score, precision, recall, delay_score, delay_precision, delay_recall
                    num_threshold = i
            best_score, best_precision, best_recall, best_delay_score, best_delay_precision, best_delay_recall, _ = compute_metric_multi(path, prefix=prefix, ignore=ignore, delay=7, num_threshold=num_threshold, type_ignore=type_ignore)
            print('best num_threshold:', num_threshold)
            print('best F1')
            print('adjust F1 Score:', best_score)
            print('Precision:', best_precision)
            print('Recall:', best_recall)
            print('delay F1 Score:', best_delay_score)
            print('delay Precision:', best_delay_precision)
            print('delay Recall:', best_delay_recall)
        else:
            num_threshold = 400
        all_types = ['LevelShiftUp', 'LevelShiftDown', 'SingleSpike', 'SingleDip', 'TransientLevelShiftUp', 'TransientLevelShiftDown', 'MultipleSpikes', 'MultipleDips']
        if args.compute_type:
            for t in all_types:
                print('type:', t)
                compute_metric_multi(path, prefix=prefix, ignore=ignore, delay=7, num_threshold=num_threshold, type_only=t, compute_all=False)
        