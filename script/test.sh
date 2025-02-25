python run.py \
    --trans_beta 0 \
    --trans_alpha 95 \
    --test_ratio 0.5 \
    --window_size 400 \
    --retrieve_positive_num 2 \
    --retrieve_database_ratio 1 \
    --prompt_mode 3 \
    --run_name test \
    --infer_data_path data/Yahoo_hard \
    --result_save_dir result \
    --retreive_data_path data/Yahoo \
    --sub_company 'real' \
    --delete_zero 1 \
    --model_engine gpt-4o-2024-05-13 \