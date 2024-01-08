set -e

# train text2natsql-t5-base model
CUDA_VISIBLE_DEVICES=6 python -u text2sql.py \
    --batch_size 16 \
    --gradient_descent_step 2 \
    --learning_rate 1e-4 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-base-graph-rtransformer" \
    --tensorboard_save_path "./tensorboard_log/text2natsql-t5-base-graph-rtransformer" \
    --model_name_or_path "./llm/t5-base" \
    --model "rtransformer" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_spider_natsql.json"

CUDA_VISIBLE_DEVICES=7 python -u text2sql.py \
    --batch_size 16 \
    --gradient_descent_step 2 \
    --learning_rate 1e-4 \
    --epochs 128 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-base-graph-rgat" \
    --tensorboard_save_path "./tensorboard_log/text2natsql-t5-base-graph-rgat" \
    --model_name_or_path "./llm/t5-base" \
    --model "rgat" \
    --use_adafactor \
    --mode train \
    --train_filepath "./data/preprocessed_data/resdsql_train_spider_natsql.json"
    
# select the best text2natsql-t5-base ckpt
CUDA_VISIBLE_DEVICES=7 python -u evaluate_text2sql_ckpts.py \
    --batch_size 8 \
    --seed 42 \
    --save_path "./models/text2natsql-t5-base-graph-rgat" \
    --eval_results_path "./eval_results/text2natsql-t5-base-graph-rgat" \
    --mode eval \
    --dev_filepath "./data/preprocessed_data/resdsql_dev_natsql.json" \
    --original_dev_filepath "./data/spider/dev.json" \
    --db_path "./database" \
    --model "rgat" \
    --tables_for_natsql "./data/preprocessed_data/tables_for_natsql.json" \
    --num_beams 8 \
    --num_return_sequences 8 \
    --target_type "natsql"