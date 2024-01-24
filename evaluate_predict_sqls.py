import os
import re
import json
import torch
import argparse
import torch.optim as optim
import transformers

from tqdm import tqdm
from tokenizers import AddedToken

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration
from transformers.optimization import Adafactor
from transformers.trainer_utils import set_seed
from utils.spider_metric.evaluator import EvaluateTool
from utils.load_dataset import Text2SQLDataset
from utils.text2sql_decoding_utils import decode_sqls, decode_natsqls


from utils.args import ModelArguments
from transformers.models.auto import AutoConfig
from model import GraphLLModel
from tokenizer import TokenPreprocessor


token_preprocessor = TokenPreprocessor()


def parse_option():
    parser = argparse.ArgumentParser("command line arguments for fine-tuning pre-trained language model.")
    
    parser.add_argument('--batch_size', type = int, default = 8,
                        help = 'input batch size.')
    parser.add_argument('--gradient_descent_step', type = int, default = 4,
                        help = 'perform gradient descent per "gradient_descent_step" steps.')
    parser.add_argument('--learning_rate',type = float, default = 3e-5,
                        help = 'learning rate.')
    parser.add_argument('--epochs', type = int, default = 128,
                        help = 'training epochs.')
    parser.add_argument('--seed', type = int, default = 42,
                        help = 'random seed.')
    parser.add_argument('--save_path', type = str, default = "models/text2sql",
                        help = 'save path of best fine-tuned text2sql model.')
    parser.add_argument('--tensorboard_save_path', type = str, default = "tensorboard_log/text2sql",
                        help = 'save path of tensorboard log.')
    parser.add_argument('--model_name_or_path', type = str, default = "./llm/t5-base",
                        help = 
                        '''
                        pre-trained model name. 
                        options: 
                            t5-base, https://huggingface.co/t5-base;
                            t5-large, https://huggingface.co/t5-large;
                            t5-3b, https://huggingface.co/t5-3b;
                        ''')
    parser.add_argument('--use_adafactor', action='store_true',
                        help = 'whether to use adafactor optimizer.')
    parser.add_argument('--model', type = str, default = "transformer",
                        help = 'transformer or rgat or rtransformer.')
    parser.add_argument('--mode', type = str, default = "train",
                        help='trian, eval or test.')
    parser.add_argument('--train_filepath', type = str, default = "./data/preprocessed_data/resdsql_train_spider.json",
                        help = 'file path of test2sql training set.')
    parser.add_argument('--train_preprocessed_dataset', type = str, default = "./data/preprocessed_data/resdsql_train_spider.pkl",
                        help = 'preprocessed file of training dataset')
    parser.add_argument('--dev_filepath', type = str, default = "data/preprocessed_data/resdsql_dev.json",
                        help = 'file path of test2sql dev set.')
    parser.add_argument('--dev_preprocessed_dataset', type = str, default = "./data/preprocessed_data/resdsql_dev_spider.pkl",
                        help = 'preprocessed file of dev dataset')
    parser.add_argument('--original_dev_filepath', type = str, default = "data/spider/dev.json",
                        help = 'file path of the original dev set (for registing evaluator).')
    parser.add_argument('--db_path', type = str, default = "database",
                        help = 'file path of database.')
    parser.add_argument('--tables_for_natsql', type = str, default = "NatSQL/NatSQLv1_6/tables_for_natsql.json",
                        help = 'file path of tables_for_natsql.json.')
    parser.add_argument('--num_beams', type = int, default = 8,
                        help = 'beam size in model.generate() function.')
    parser.add_argument('--num_return_sequences', type = int, default = 8,
                        help = 'the number of returned sequences in model.generate() function (num_return_sequences <= num_beams).')
    parser.add_argument("--target_type", type = str, default = "sql",
                        help = "sql or natsql.")
    parser.add_argument("--predicted_sql_file", type = str, default="data/output/deepseek-moe-16b-chat.json")
    parser.add_argument("--output", type = str, default = "predicted_sql.txt",
                        help = "save file of the predicted sqls.")
    
    opt = parser.parse_args()
    return opt


def load_predicted_sql(path):
    with open(path, "r") as f:
        predicted_sqls = f.readlines()
    predicted_sqls = [predicted_sqls[i] for i in range(len(predicted_sqls)) if i % 2 == 0]
    return predicted_sqls

def _test(opt):
    set_seed(opt.seed)
    print(opt)

    predict_sqls = load_predicted_sql(opt.predicted_sql_file)
    
    evaluator = EvaluateTool()
    evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
    spider_metric_result = evaluator.evaluate(predict_sqls)
    print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
    print('exec score: {}'.format(spider_metric_result["exec"]))
    print(spider_metric_result)
    return spider_metric_result


if __name__ == "__main__":
    opt = parse_option()
    _test(opt)
