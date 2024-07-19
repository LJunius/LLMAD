# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import re

prompt_template_WSD = """
## Instructions
Determine if there are any anomalies in the provided AIOPS flow data sequence.

## Following Rules:
1. A data point is considered an anomaly if it is part of a sequence of at least one consecutive anomalous points or continues to plummet or surge abruptly. 
2. A data point is considered an anomaly if it is identified as a continuous low/high value anomaly if it remains below/above a predefined normal threshold for a prolonged duration, deviating from the anticipated norm.  
3. Given that the vast majority of data points are expected to be '''no anomaly''', Anomalies are exceedingly rare and should only be identified with absolute certainty.
4. Normal data may exhibit volatility, which should not be mistaken for anomalies. 
5. Mislabeling normal data as an anomaly can lead to catastrophic failures. Exercise extreme caution. False positives are unacceptable.
6. '''If do not have 100 percent confidence that data is an anomaly, do not flag it as an anomaly.'''
7. '''The output of anomaly intervals needs to be accurately located and should not be excessively long. '''
8. The number of abnormal intervals within a detection range can not exceed 3.
9. anomaly_type should be one of the following:
  - **PersistentLevelShiftUp**
    - The data shifts to a higher value and maintains that level consistently, '''do not return to the original baseline.''' like `1 2 1 2 1 2 *500* *480* *510* *500* *500*`
  - **PersistentLevelShiftDown**
    - The data shifts to a lower value and maintains that level consistently, '''do not return to the original baseline.''' like `1 2 1 2 *-100* *-102* *-104* *-110* *-110*`
  - **TransientLevelShiftUp**
    - The data temporarily shifts to a higher value and then returning to the original baseline, '''the anomaly maintains for at least 5 data points.''' like `1 2 1 2 1 2 *500* *500* *499* *510* *500* 1 2 1 2`
  - **TransientLevelShiftDown**
    - The data temporarily shifts to a lower value and then returning to the original baseline, '''the anomaly maintains for at least 5 data points.''' like `1 2 1 2 *-100* *-102* *-104* *-110* *-100* 1 2 1 2`
  - **SingleSpike**
    - A brief, sharp rise in data value followed by an immediate return to the baseline. like `1 2 1 2 1 2 *200* *500* 1 2`
  - **SingleDip**
    - A brief, sharp drop in data value followed by an immediate return to the baseline. like `1 2 1 2 *-500* *-200* 1 2 1 2`
  - **MultipleSpikes**
    - '''Several''' brief, sharp rises in data value, each followed by a return to the baseline. like `1 2 *500* 3 2 *510* *200* 1 2 *480* 1 2`
  - **MultipleDips**
    - '''Several''' brief, sharp drops in data value, each followed by a return to the baseline. like `1 2 *-100* 3 2 *-110* *-200* 1 2 *-120* 1 2`
10. alarm_level should be one of the following:
  - **Urgent/Error**
    - This category is for values that represent a severe risk, potentially causing immediate damage or harm across all event types whether increases, decreases, spikes, dips, or multiple occurrences.
  - **Important**
    - Allocated for moderate value changes (both increases and decreases) that could escalate to future problems or system stress but are not immediately hazardous. This also covers upward transient level shifts that concern system longevity and potential failure indications from downward shifts.
  - **Warning**
    - Used for noticeable deviations from the norm that are not yet critical but merit close monitoring. This includes single spikes and dips that are moderate in nature, as well as multiple non-critical spikes and level shifts that are significant but not yet dangerous.
11. The briefExplanation must comprise a explicit three-step analysis '''results''' utilizing precise data (do not only repeat the rule):
  - Step 1: Assess the overall trend to ascertain if it aligns with expected patterns, thereby identifying any overarching anomalies. 
  - Step 2: Determine if there is any local data segment with any continuous low or high values compared to the normal data sequence.
  - Step 3: Reassess the identified points to confirm their anomalous nature, given the rarity of true anomalies.
12. Provide responses in a strict JSON format suitable for direct parsing, without any additional textual commentary.

## Response Format
{
  "briefExplanation": {"step1_global": analysis reason, "step2_local": analysis reason, "step3_reassess": analysis reason},
  "is_anomaly": false/true,
  "anomalies": []/[index1, index2, index3, ...],
  "reason_for_anomaly_type": "no"/"reason for anomaly type",
  "anomaly_type": "no"/"classification of main anomaly",(only one)
  "reason_for_alarm_level": "no"/"reason for alarm level",
  "alarm_level": "no"/"Urgent/Error"/"Important"/"Warning",(only one)
}

## Data
Please analyze the latest data with the highest level of diligence and caution:
- Historical normal data sequence: `{normal_data}`
- Historical anomaly data sequence(*XXX* is anomaly point), `{anomaly_data}`
- The latest `{data_len}` data points for evaluation: `{data}`
"""

prompt_template_KPI = """
## Instructions
Determine if there are any anomalies in the provided AIOPS flow data sequence.

## Following Rules:
1. A data point is considered an anomaly if it is part of a sequence of at least one consecutive anomalous points or continues to plummet or surge abruptly. 
2. Given that the vast majority of data points are expected to be '''no anomaly''', Anomalies are exceedingly rare and should only be identified with absolute certainty.
3. Normal data may exhibit volatility, which should not be mistaken for anomalies. 
4. Mislabeling normal data as an anomaly can lead to catastrophic failures. Exercise extreme caution. False positives are unacceptable.
5. '''If do not have 100 percent confidence that data is an anomaly, do not flag it as an anomaly.'''
6. '''The output of anomaly intervals needs to be accurately located and should not be excessively long. '''
7. anomaly_type should be one of the following:
  - **PersistentLevelShiftUp**
    - The data shifts to a higher value and maintains that level consistently, '''do not return to the original baseline.''' like `1 2 1 2 1 2 *500* *480* *510* *500* *500*`
  - **PersistentLevelShiftDown**
    - The data shifts to a lower value and maintains that level consistently, '''do not return to the original baseline.''' like `1 2 1 2 *-100* *-102* *-104* *-110* *-110*`
  - **TransientLevelShiftUp**
    - The data temporarily shifts to a higher value and then returning to the original baseline, '''the anomaly maintains for at least 5 data points and return to baseline''' like `1 2 1 2 1 2 *500* *500* *499* *510* *500* 1 2 1 2`
  - **TransientLevelShiftDown**
    - The data temporarily shifts to a lower value and then returning to the original baseline, '''the anomaly maintains for at least 5 data points return to baseline''' like `1 2 1 2 *-100* *-102* *-104* *-110* *-100* 1 2 1 2`
  - **SingleSpike**
    - A brief, sharp rise in data value followed by an immediate return to the baseline. like `1 2 1 2 1 2 *200* *500* 1 2`
  - **SingleDip**
    - A brief, sharp drop in data value followed by an immediate return to the baseline. like `1 2 1 2 *-500* *-200* 1 2 1 2`
  - **MultipleSpikes**
    - '''Several''' brief, sharp rises in data value, each followed by a return to the baseline. like `1 2 *500* 3 2 *510* *200* 1 2 *480* 1 2`
  - **MultipleDips**
    - '''Several''' brief, sharp drops in data value, each followed by a return to the baseline. like `1 2 *-100* 3 2 *-110* *-200* 1 2 *-120* 1 2`
8. alarm_level should be one of the following:
  - **Urgent/Error**
    - This category is for values that represent a severe risk, potentially causing immediate damage or harm across all event types whether increases, decreases, spikes, dips, or multiple occurrences.
  - **Important**
    - Allocated for moderate value changes (both increases and decreases) that could escalate to future problems or system stress but are not immediately hazardous. This also covers upward transient level shifts that concern system longevity and potential failure indications from downward shifts.
  - **Warning**
    - Used for noticeable deviations from the norm that are not yet critical but merit close monitoring. This includes single spikes and dips that are moderate in nature, as well as multiple non-critical spikes and level shifts that are significant but not yet dangerous.
9. The briefExplanation must comprise a explicit three-step analysis utilizing precise data (do not only repeat the rule):
  - Step 1: Assess the overall trend to ascertain if it aligns with expected patterns, thereby identifying any overarching anomalies.
  - Step 2: Examine the local data segments to detect any specific deviations or anomalies.
  - Step 3: Reassess the identified points to confirm their anomalous nature, given the rarity of true anomalies. This step ensures that the detected points are not merely normal fluctuations or seasonal variations.
10. Provide responses in a strict JSON format suitable for direct parsing, without any additional textual commentary.

## Response Format
{
  "briefExplanation": {"step1_global": analysis reason, "step2_local": analysis reason, "step3_reassess": analysis reason},
  "is_anomaly": false/true,
  "anomalies": []/[index1, index2, index3, ...],
  "reason_for_anomaly_type": "no"/"reason for anomaly type",
  "anomaly_type": "no"/"classification of main anomaly",(only one)
  "reason_for_alarm_level": "no"/"reason for alarm level",
  "alarm_level": "no"/"Urgent/Error"/"Important"/"Warning",(only one)
}

## Data
Please analyze the latest data with the highest level of diligence and caution:
- Historical normal data sequence: `{normal_data}`
- Historical anomaly data sequence(*XXX* is anomaly point), `{anomaly_data}`
- The latest `{data_len}` data points for evaluation: `{data}`

"""

prompt_template_Yahoo_1 = """
## Instructions
Determine if there are any anomalies in the provided AIOPS flow data sequence.

## Following Rules:
1. A data point is considered an anomaly if it is part of a sequence of at least one consecutive anomalous points or continues to plummet or surge abruptly. 
2. A data point is considered an anomaly if it is identified as a continuous low/high value anomaly if it remains below/above a predefined normal threshold for a prolonged duration, deviating from the anticipated norm.  
3. Given that the vast majority of data points are expected to be '''no anomaly''', Anomalies are exceedingly rare and should only be identified with absolute certainty.
4. Normal data may exhibit volatility, which should not be mistaken for anomalies. 
5. Mislabeling normal data as an anomaly can lead to catastrophic failures. Exercise extreme caution. False positives are unacceptable.
6. '''If do not have high percent confidence that data is an anomaly, do not flag it as an anomaly.'''
7. '''The output of anomaly intervals needs to be accurately located and should not be excessively long. '''
8. The number of abnormal intervals within a detection range can not exceed 3.
9. anomaly_type should be one of the following:
  - **PersistentLevelShiftUp**
    - The data shifts to a higher value and maintains that level consistently, '''do not return to the original baseline.''' like `1 2 1 2 1 2 *500* *480* *510* *500* *500*`
  - **PersistentLevelShiftDown**
    - The data shifts to a lower value and maintains that level consistently, '''do not return to the original baseline.''' like `1 2 1 2 *-100* *-102* *-104* *-110* *-110*`
  - **TransientLevelShiftUp**
    - The data temporarily shifts to a higher value and then returning to the original baseline, '''the anomaly maintains for at least 5 data points.''' like `1 2 1 2 1 2 *500* *500* *499* *510* *500* 1 2 1 2`
  - **TransientLevelShiftDown**
    - The data temporarily shifts to a lower value and then returning to the original baseline, '''the anomaly maintains for at least 5 data points.''' like `1 2 1 2 *-100* *-102* *-104* *-110* *-100* 1 2 1 2`
  - **SingleSpike**
    - A brief, sharp rise in data value followed by an immediate return to the baseline. like `1 2 1 2 1 2 *200* *500* 1 2`
  - **SingleDip**
    - A brief, sharp drop in data value followed by an immediate return to the baseline. like `1 2 1 2 *-500* *-200* 1 2 1 2`
  - **MultipleSpikes**
    - '''Several''' brief, sharp rises in data value, each followed by a return to the baseline. like `1 2 *500* 3 2 *510* *200* 1 2 *480* 1 2`
  - **MultipleDips**
    - '''Several''' brief, sharp drops in data value, each followed by a return to the baseline. like `1 2 *-100* 3 2 *-110* *-200* 1 2 *-120* 1 2`
10. alarm_level should be one of the following:
  - **Urgent/Error**
    - This category is for values that represent a severe risk, potentially causing immediate damage or harm across all event types whether increases, decreases, spikes, dips, or multiple occurrences.
  - **Important**
    - Allocated for moderate value changes (both increases and decreases) that could escalate to future problems or system stress but are not immediately hazardous. This also covers upward transient level shifts that concern system longevity and potential failure indications from downward shifts.
  - **Warning**
    - Used for noticeable deviations from the norm that are not yet critical but merit close monitoring. This includes single spikes and dips that are moderate in nature, as well as multiple non-critical spikes and level shifts that are significant but not yet dangerous.
11. The briefExplanation must comprise a explicit three-step analysis '''results''' utilizing precise data (do not only repeat the rule):
  - Step 1: Assess the overall trend to ascertain if it aligns with expected patterns, thereby identifying any overarching anomalies. 
  - Step 2: Determine if there is any local data segment with any continuous low or high values compared to the normal data sequence.
  - Step 3: Reassess the identified points to confirm their anomalous nature, given the rarity of true anomalies.
12. Provide responses in a strict JSON format suitable for direct parsing, without any additional textual commentary.

## Response Format
{
  "briefExplanation": {"step1_global": analysis reason, "step2_local": analysis reason, "step3_reassess": analysis reason},
  "is_anomaly": false/true,
  "anomalies": []/[index1, index2, index3, ...],
  "reason_for_anomaly_type": "no"/"reason for anomaly type",
  "anomaly_type": "no"/"classification of main anomaly",(only one)
  "reason_for_alarm_level": "no"/"reason for alarm level",
  "alarm_level": "no"/"Urgent/Error"/"Important"/"Warning",(only one)
}

## Data
Please analyze the latest data with the highest level of diligence and caution:
- Historical normal data sequence: `{normal_data}`
- Historical anomaly data sequence(*XXX* is anomaly point), `{anomaly_data}`
- The latest `{data_len}` data points for evaluation: `{data}`
"""
prompt_template_Yahoo_2 = """
## Instructions
Determine if there are any anomalies in the provided AIOPS flow data sequence.

## Following Rules:
1. A data point is considered an anomaly if it is part of a sequence of at least one consecutive anomalous points or continues to plummet or surge abruptly. 
2. Typically, anomalies are outliers such as spikes and dips, which are often isolated points. Be aware that there may be multiple anomalies present; you should identify all possible anomalous data points.
3. Given that the vast majority of data points are expected to be '''no anomaly''', Anomalies are exceedingly rare and should only be identified with absolute certainty.
4. Mislabeling normal data as an anomaly can lead to catastrophic failures. Exercise extreme caution. False positives are unacceptable.
5. '''If do not have 100 percent confidence that data is an anomaly, do not flag it as an anomaly.'''
6. '''The output of anomaly intervals needs to be accurately located and should not be excessively long. '''
7. anomaly_type should be one of the following:
  - **SingleSpike**
    - A brief, sharp rise in data value followed by an immediate return to the baseline. like `1 2 1 2 1 2 *200* *500* 1 2`
  - **SingleDip**
    - A brief, sharp drop in data value followed by an immediate return to the baseline. like `1 2 1 2 *-500* *-200* 1 2 1 2`
8. alarm_level should be one of the following:
  - **Urgent/Error**
    - This category is for values that represent a severe risk, potentially causing immediate damage or harm across all event types whether increases, decreases, spikes, dips, or multiple occurrences.
  - **Important**
    - Allocated for moderate value changes (both increases and decreases) that could escalate to future problems or system stress but are not immediately hazardous. This also covers upward transient level shifts that concern system longevity and potential failure indications from downward shifts.
  - **Warning**
    - Used for noticeable deviations from the norm that are not yet critical but merit close monitoring. This includes single spikes and dips that are moderate in nature, as well as multiple non-critical spikes and level shifts that are significant but not yet dangerous.
9. The briefExplanation must comprise a explicit two-step analysis '''results''' utilizing precise data (do not only repeat the rule):
  - Step 1: Examine the local data point to detect any specific deviations or anomalies. You should identify all possible anomalous data points.
  - Step 2: Reassess the identified points to confirm their anomalous nature, given the rarity of true anomalies.
10. Provide responses in a strict JSON format suitable for direct parsing, without any additional textual commentary.

## Response Format
{
  "briefExplanation": {"step1_local": analysis reason, "step2_reasses": analysis reason},
  "is_anomaly": false/true,
  "anomalies": []/[index1, index2, index3, ...],
  "reason_for_anomaly_type": "no"/"reason for anomaly type",
  "anomaly_type": "no"/"classification of main anomaly",(only one)
  "reason_for_alarm_level": "no"/"reason for alarm level",
  "alarm_level": "no"/"Urgent/Error"/"Important"/"Warning",(only one)
}

## Data
Please analyze the latest data with the highest level of diligence and caution:
- Historical normal data sequence: `{normal_data}`
- Historical anomaly data sequence(*XXX* is anomaly point), `{anomaly_data}`
- The latest `{data_len}` data points for evaluation: `{data}`

"""

class PromptTemplate:
    def __init__(self, prompt_mode=1):
        self.prompt_mode = prompt_mode
        
        prompt_templates = {
            1: prompt_template_WSD,
            2: prompt_template_KPI,
            3: prompt_template_Yahoo_1,
            4: prompt_template_Yahoo_2,
            # add more prompt templates here
        }
        if prompt_mode not in prompt_templates:
            raise ValueError('Invalid prompt mode: %d' % prompt_mode)
        self.prompt_template = prompt_templates[prompt_mode]
        
    def get_template(self, **kwargs):
        get_template_func = getattr(self, 'get_template_%d' % self.prompt_mode)
        prompt_res = get_template_func(**kwargs)
        return prompt_res

    def get_template_1(self, normal_data, data, data_len, anomaly_datas):
        anomaly = ''
        for idx, anomaly_data in enumerate(anomaly_datas):
            anomaly += 'sequence %d: %s\n' % (idx+1, anomaly_data)
        prompt_res = self.prompt_template.replace('{normal_data}', normal_data).replace('{data}', data).replace('{data_len}', str(data_len)).replace('{anomaly_data}', anomaly)
        return prompt_res
    
    def get_template_2(self, normal_data, data, data_len, anomaly_datas):
        anomaly = ''
        for idx, anomaly_data in enumerate(anomaly_datas):
            anomaly += 'sequence %d: %s\n' % (idx+1, anomaly_data)
        prompt_res = self.prompt_template.replace('{normal_data}', normal_data).replace('{data}', data).replace('{data_len}', str(data_len)).replace('{anomaly_data}', anomaly)
        return prompt_res
    
    def get_template_3(self, normal_data, data, data_len, anomaly_datas):
        anomaly = ''
        for idx, anomaly_data in enumerate(anomaly_datas):
            anomaly += 'sequence %d: %s\n' % (idx+1, anomaly_data)
        prompt_res = self.prompt_template.replace('{normal_data}', normal_data).replace('{data}', data).replace('{data_len}', str(data_len)).replace('{anomaly_data}', anomaly)
        return prompt_res
      
    def get_template_4(self, normal_data, data, data_len, anomaly_datas):
        anomaly = ''
        for idx, anomaly_data in enumerate(anomaly_datas):
            anomaly += 'sequence %d: %s\n' % (idx+1, anomaly_data)
        prompt_res = self.prompt_template.replace('{normal_data}', normal_data).replace('{data}', data).replace('{data_len}', str(data_len)).replace('{anomaly_data}', anomaly)
        return prompt_res
    
if __name__ == '__main__':
    prompt_template = PromptTemplate(prompt_mode=1)
    normal_data = '1,2,3,4,5,6,7,8,9,10'
    data = '11,12,13,14,15,16,17,18,19,20'
    prompt_res = prompt_template.get_template(normal_data=normal_data, data=data, data_len=10)
    print(prompt_res)
