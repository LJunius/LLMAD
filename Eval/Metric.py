# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import math

def calculate_f1(y_true, y_pred):
    
    TP = sum((y_true[i] == 1 and y_pred[i] == 1) for i in range(len(y_true)))
    FP = sum((y_true[i] == 0 and y_pred[i] == 1) for i in range(len(y_true)))
    FN = sum((y_true[i] == 1 and y_pred[i] == 0) for i in range(len(y_true)))

    Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    

    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0

    return F1, Precision, Recall



def point_adjust(predict, actual):
    anomaly_state = False
    new_predict = predict.copy()
    for i in range(len(predict)):
        if actual[i] and predict[i] and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if not actual[j]:
                    break
                else:
                    new_predict[j] = True
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            new_predict[i] = True
    return new_predict, actual


def get_range_proba(predict, label, delay=7):
    splits = np.where(label[1:] != label[:-1])[0] + 1
    # import pdb; pdb.set_trace()
    is_anomaly = label[0] == 1
    new_predict = np.array(predict)
    pos = 0
    for sp in splits:
        if is_anomaly:
            if 1 in predict[pos:min(pos + delay + 1, sp)]:
                new_predict[pos: sp] = 1
            else:
                new_predict[pos: sp] = 0
        is_anomaly = not is_anomaly
        pos = sp
    sp = len(label)
    if is_anomaly:  # anomaly in the end
        if 1 in predict[pos: min(pos + delay + 1, sp)]:
            new_predict[pos: sp] = 1
        else:
            new_predict[pos: sp] = 0
    return new_predict


class EventF1PA():
    def __init__(self, mode="log", base=3) -> None:
        """
        Using the Event-based point-adjustment F1 score to evaluate the models.
        
        Parameters:
            mode (str): Defines the scale at which the anomaly segment is processed. \n
                One of:\n
                    - 'squeeze': View an anomaly event lasting t timestamps as one timepoint.
                    - 'log': View an anomaly event lasting t timestamps as log(t) timepoint.
                    - 'sqrt': View an anomaly event lasting t timestamps as sqrt(t) timepoint.
                    - 'raw': View an anomaly event lasting t timestamps as t timepoint.
                If using 'log', you can specify the param "base" to return the logarithm of x to the given base, 
                calculated as log(x) / log(base).
            base (int): Default is 3.
        """
        super().__init__()
        
        self.eps = 1e-15
        self.name = "event-based f1 under pa with mode %s"%(mode)
        if mode == "squeeze":
            self.func = lambda x: 1
        elif mode == "log":
            self.func = lambda x: math.floor(math.log(x+base, base))
        elif mode == "sqrt":
            self.func = lambda x: math.floor(math.sqrt(x))
        elif mode == "raw":
            self.func = lambda x: x
        else:
            raise ValueError("please select correct mode.")
        
    def calc(self, scores, labels, margins):
        '''
        Returns:
         A F1class (Evaluations.Metrics.F1class), including:\n
            best_f1: the value of best f1 score;\n
            precision: corresponding precision value;\n
            recall: corresponding recall value;\n
            threshold: the value of threshold when getting best f1.
        '''
        
        search_set = []
        tot_anomaly = 0
        ano_flag = 0
        ll = len(labels)
        for i in range(labels.shape[0]):
            if labels[i] > 0.5 and ano_flag == 0:
                ano_flag = 1
                start = i
            
            # alleviation
            elif labels[i] <= 0.5 and ano_flag == 1:
                ano_flag = 0
                end = i
                tot_anomaly += self.func(end - start)
                
            # marked anomaly at the end of the list
            if ano_flag == 1 and i == ll - 1:
                ano_flag = 0
                end = i + 1
                tot_anomaly += self.func(end - start)

        flag = 0
        cur_anomaly_len = 0
        cur_max_anomaly_score = 0
        for i in range(labels.shape[0]):
            if labels[i] > 0.5:
                # record the highest score in an anomaly segment
                if flag == 1:
                    cur_anomaly_len += 1
                    cur_max_anomaly_score = scores[i] if scores[i] > cur_max_anomaly_score else cur_max_anomaly_score  # noqa: E501
                else:
                    flag = 1
                    cur_anomaly_len = 1
                    cur_max_anomaly_score = scores[i]
            else:
                # reconstruct the score using the highest score
                if flag == 1:
                    flag = 0
                    search_set.append((cur_max_anomaly_score, self.func(cur_anomaly_len), True))
                    search_set.append((scores[i], 1, False))
                else:
                    search_set.append((scores[i], 1, False))
        if flag == 1:
            search_set.append((cur_max_anomaly_score, self.func(cur_anomaly_len), True))
            
        search_set.sort(key=lambda x: x[0], reverse=True)
        best_f1 = 0
        threshold = 0
        P = 0
        TP = 0
        best_P = 0
        best_TP = 0
        for i in range(len(search_set)):
            P += search_set[i][1]
            if search_set[i][2]:  # for an anomaly point
                TP += search_set[i][1]
            precision = TP / (P + self.eps)
            recall = TP / (tot_anomaly + self.eps)
            f1 = 2 * precision * recall / (precision + recall + self.eps)
            if f1 > best_f1:
                best_f1 = f1
                threshold = search_set[i][0]
                best_P = P
                best_TP = TP

        precision = best_TP / (best_P + self.eps)
        recall = best_TP / (tot_anomaly + self.eps)
        return best_f1, precision, recall, threshold

if __name__=='__main__':
    
    import pandas as pd
    
    data = pd.read_csv('exp_result/KPI_t847_prompt_13_400_trans005_r2.csv')
    
    y_true = data['label'].tolist()

    
    y_pred = data['predict'].tolist()
    # y_true = [0,0,1,1,1,0,1,1,1,1]
    # y_pred = [1,0,0,1,1,0,0,0,1,1]
    
    # adjust
    adjust_y_pred, adjust_y_true = point_adjust(y_pred, y_true)
    adjust_y_pred = [int(i) for i in adjust_y_pred]
    # import pdb; pdb.set_trace()
    score, precision, recall = calculate_f1(adjust_y_true, adjust_y_pred)

    print('adjust F1 Score:', score)
    print('Precision:', precision)
    print('Recall:', recall)
    
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    delay_y_pred = get_range_proba(y_pred, y_true, delay=7)
    
    score, precision, recall = calculate_f1(y_true, delay_y_pred)
    print ('delay F1 Score:', score)
    print('delay Precision:', precision)
    print('delay Recall:', recall)
    
    event_f1 = EventF1PA()
    score, precision, recall, threshold = event_f1.calc(np.array(y_pred), np.array(y_true), 0)
    print ('event F1 Score:', score)
    print('event Precision:', precision)
    print('event Recall:', recall)
    print('event Threshold:', threshold)
