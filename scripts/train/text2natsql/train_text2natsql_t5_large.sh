set -e

# train text2natsql-t5-large model
python -u text2sql.py \
    --batch_size 8 \
    --gradient_descent_step 4 \
    --learning_rate 5e-5 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-large-threshold" \
    --tensorboard_save_path "./tensorboard_log/text2natsql-t5-large-threshold" \
    --model_name_or_path "./llm/t5-large" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_spider_natsql_threshold.json"

CUDA_VISIBLE_DEVICES=4 python text2sql.py \
    --batch_size 8 \
    --gradient_descent_step 4 \
    --learning_rate 5e-5 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-large-graph-rgat" \
    --tensorboard_save_path "./tensorboard_log/text2natsql-t5-large-graph-rgat" \
    --model_name_or_path "./llm/t5-large" \
    --model "rgat" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_spider_natsql.json"

CUDA_VISIBLE_DEVICES=6 python text2sql.py \
    --batch_size 8 \
    --gradient_descent_step 4 \
    --learning_rate 5e-5 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-large-graph-rtransformer" \
    --tensorboard_save_path "./tensorboard_log/text2natsql-t5-large-graph-rtransformer" \
    --model_name_or_path "./llm/t5-large" \
    --model "rtransformer" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_spider_natsql.json"

# select the best text2natsql-t5-large ckpt
python -u evaluate_text2sql_ckpts.py \
    --batch_size 8 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-large-threshold" \
    --eval_results_path "./eval_results/text2natsql-t5-large-threshold" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_dev_natsql_threshold.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --tables_for_natsql "./data/preprocessed_data/tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql"

CUDA_VISIBLE_DEVICES=6 python -u evaluate_text2sql_ckpts.py \
    --batch_size 8 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-large-graph-rtransformer" \
    --eval_results_path "./eval_results/text2natsql-t5-large-graph-rtransformer" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_dev_natsql.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --model "rtransformer" \
    --tables_for_natsql "./data/preprocessed_data/tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql"