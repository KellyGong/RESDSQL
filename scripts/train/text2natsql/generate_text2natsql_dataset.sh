set -e

# generate text2natsql training dataset with noise_rate 0.2
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/preprocessed_train_spider_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_train_spider_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "train" \
    --noise_rate 0.2 \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

# predict probability for each schema item in the eval set
python schema_item_classifier_share.py \
    --batch_size 32 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2natsql_schema_item_classifier_share" \
    --dev_filepath "./data/preprocessed_data/preprocessed_dev_natsql.json" \
    --output_filepath "./data/preprocessed_data/dev_with_probs_natsql_share.json" \
    --use_contents \
    --mode "eval"

# generate text2natsql development dataset
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/dev_with_probs_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_dev_natsql.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "eval" \
    --use_contents \
    --output_skeleton \
    --target_type "natsql"

python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/dev_with_probs_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_dev_natsql_threshold.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "eval" \
    --use_contents \
    --use_threshold \
    --output_skeleton \
    --target_type "natsql"

python schema_item_classifier.py \
    --batch_size 32 \
    --device "0" \
    --seed 42 \
    --save_path "./models/text2natsql_schema_item_classifier" \
    --dev_filepath "./data/preprocessed_data/preprocessed_train_spider_natsql.json" \
    --output_filepath "./data/preprocessed_data/train_with_probs_natsql.json" \
    --use_contents \
    --mode "eval"

# generate text2natsql train dataset with threshold
python text2sql_data_generator.py \
    --input_dataset_path "./data/preprocessed_data/train_with_probs_natsql.json" \
    --output_dataset_path "./data/preprocessed_data/resdsql_train_spider_natsql_threshold.json" \
    --topk_table_num 4 \
    --topk_column_num 5 \
    --mode "eval" \
    --use_contents \
    --use_threshold \
    --output_skeleton \
    --target_type "natsql"
