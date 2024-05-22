# LLM Time Series Anomaly Detection


## Description

Using LLM to solve Time Series Anomaly Detection task

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)

## Installation

```shell
cd LLM_AD
pip install -U openai
pip install -U fastdtw pandas numpy scipy
```

## Usage
| File Name | Description |
|-----------|-------------|
| run.py     | program entry | 
| Prompt_template.py | The structure of prompt |
| Eval/* | Compute Metric |

## run 
set config.yaml
```yaml
   openai:
     api_key: "your-api-key"
     base_url: "https://api.openai.com/v1"
```
### Yahoo
```shell
bash script/yahoo.sh
```

### WSD
```shell
bash script/wsd.sh
```

### KPI
```shell
bash script/kpi.sh
```