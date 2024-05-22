TRANS_BETA=0
TRANS_ALPHA=95
TEST_RATIO=0.05
WINDOW_SIZE=400
RETRIEVE_POSITIVE_NUM=2
RETRIEVE_NEGATIVE_NUM=1
RETRIEVE_DATABASE_RATIO=0.1
DELETE_ZERO=1
MODEL_ENGINE="gpt-4o-2024-05-13"
DATA_ROOT_DIR="data"
SAVE_DIR="result"

RUN_NAME="WSD_5_prompt_${PROMPT_MODE}_win_${WINDOW_SIZE}_beta${TRANS_BETA}alpha${TRANS_ALPHA}_p${RETRIEVE_POSITIVE_NUM}n${RETRIEVE_NEGATIVE_NUM}_0514_gpt_o"
for i in 0 1; do
    python run.py \
        --trans_beta $TRANS_BETA \
        --trans_alpha $TRANS_ALPHA \
        --test_ratio $TEST_RATIO \
        --window_size $WINDOW_SIZE \
        --retrieve_positive_num $RETRIEVE_POSITIVE_NUM \
        --retrieve_database_ratio $RETRIEVE_DATABASE_RATIO \
        --prompt_mode 1 \
        --run_name $RUN_NAME \
        --infer_data_path "${DATA_ROOT_DIR}/WSD" \
        --retreive_data_path "${DATA_ROOT_DIR}/WSD" \
        --sub_company 'all' \
        --delete_zero $DELETE_ZERO \
        --model_engine $MODEL_ENGINE \
        --result_save_dir $SAVE_DIR
done
python3 Eval/Metric_multi.py \
    --path "${SAVE_DIR}/${RUN_NAME}"