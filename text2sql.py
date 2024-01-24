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
from utils.load_dataset import Text2SQLDataset, Text2SQLDataset_Graph
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
    parser.add_argument("--output", type = str, default = "predicted_sql.txt",
                        help = "save file of the predicted sqls.")
    
    opt = parser.parse_args()

    return opt


# def update_config_with_graph_property(config, graph_property_lists):
#     max_in_degree, max_out_degree = 0, 0
#     max_path_length = 0

#     # for graph in graph_property_lists:
#     #     max_in_degree = max(max_in_degree, int(graph['in_degree'].max()))
#     #     max_out_degree = max(max_out_degree, int(graph['out_degree'].max()))
#     #     max_path_length = max(max_path_length, int(graph['dist'].max()))

#     config.max_in_degree = 20
#     config.max_out_degree = 20
#     config.max_path_length = 20

#     return config

def map_graph_dict_to_cuda(graph_dict, keys):
    for key in keys:
        graph_dict[key] = graph_dict[key].to('cuda')
    return graph_dict


def split_sentence_to_subtokens(sentence):
    split_tokens = re.split(r'([ .])', sentence)
    split_tokens = [token for token in split_tokens]
    return split_tokens


def _train(opt):
    set_seed(opt.seed)
    print(opt)

    if opt.tensorboard_save_path is not None:
        writer = SummaryWriter(opt.tensorboard_save_path)
    else:
        writer = None

    text2sql_tokenizer = T5TokenizerFast.from_pretrained(
        opt.model_name_or_path,
        add_prefix_space = True
    )

    if isinstance(text2sql_tokenizer, T5TokenizerFast):
        text2sql_tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])

    print("initializing text2sql model.")

    if opt.model == 'transformer':
        train_dataset = Text2SQLDataset(
            dir_=opt.train_filepath,
            mode="train",
            preprocessed_file=opt.train_preprocessed_dataset,
            tokenizer=text2sql_tokenizer
        )
    
    else:
        train_dataset = Text2SQLDataset_Graph(
            dir_=opt.train_filepath,
            mode="train",
            preprocessed_file=opt.train_preprocessed_dataset,
            tokenizer=text2sql_tokenizer
        )

    train_dataloder = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
        drop_last=True
    )

    if opt.model == "transformer":
        model_class = MT5ForConditionalGeneration if "mt5" in opt.model_name_or_path else T5ForConditionalGeneration
        # initialize model
        model = model_class.from_pretrained(opt.model_name_or_path)
        model.resize_token_embeddings(len(text2sql_tokenizer))

    elif opt.model == "rgat" or opt.model == "rtransformer":
        config = AutoConfig.from_pretrained(
            opt.model_name_or_path
        )
        config.structure_encoder = opt.model
        # update_config_with_graph_property(config, train_dataset.sequence_graphs_property)
        model = GraphLLModel(text2sql_tokenizer, opt.model_name_or_path, config)

    if torch.cuda.is_available():
        model = model.cuda()

    print("finished.")

    # warm up steps (10% training step)
    num_warmup_steps = int(0.1*opt.epochs*len(train_dataset)/opt.batch_size)
    # total training steps
    num_training_steps = int(opt.epochs*len(train_dataset)/opt.batch_size)
    # save checkpoint for each 1.42857 epochs (about 1.42857*7000=10000 examples for Spider's training set)
    num_checkpoint_steps = int(1.42857 * len(train_dataset)/opt.batch_size)

    if opt.use_adafactor:
        print("Let's use Adafactor!")
        optimizer = Adafactor(
            model.parameters(), 
            lr=opt.learning_rate, 
            scale_parameter=False, 
            relative_step=False, 
            clip_threshold = 1.0,
            warmup_init=False
        )
    else:
        print("Let's use AdamW!")
        optimizer = optim.AdamW(
            model.parameters(), 
            lr = opt.learning_rate
        )

    scheduler = transformers.get_cosine_schedule_with_warmup(
        optimizer, 
        num_warmup_steps = num_warmup_steps,
        num_training_steps = num_training_steps
    )

    model.train()
    train_step = 0
    for epoch in range(opt.epochs):
        print(f"This is epoch {epoch+1}.")
        if epoch > 0.8 * opt.epochs:
            break
        pbar = tqdm(train_dataloder)
        for batch in pbar:
            train_step += 1
            
            batch_inputs = [data[0] for data in batch]
            batch_sqls = [data[1] for data in batch]
            batch_db_ids = [data[2] for data in batch] # unused
            batch_tc_original = [data[3] for data in batch] # unused
            # batch_graphs = [map_graph_dict_to_cuda(data[4], ['graph']) for data in batch]
            if opt.model != 'transformer':
                batch_graphs = [map_graph_dict_to_cuda(data[4], ['graph']) for data in batch]
            
            # if epoch == 0:
            #     for batch_id in range(len(batch_inputs)):
            #         print(batch_inputs[batch_id])
            #         print(batch_sqls[batch_id])
            #         print("----------------------")

            # batch_inputs_tokens = [split_sentence_to_subtokens(sentence) for sentence in batch_inputs]
            batch_inputs_preprocessed = [token_preprocessor.preprocess(sentence) for sentence in batch_inputs]
            batch_sql_preprocessed = [token_preprocessor.preprocess(sql) for sql in batch_sqls]

            tokenized_inputs = text2sql_tokenizer(
                batch_inputs_preprocessed,
                padding="max_length",
                return_tensors="pt",
                max_length=512,
                truncation=True,
                # is_split_into_words=True,
            )

            # batch_sqls_tokens = [split_sentence_to_subtokens(sql) for sql in batch_sqls]
            
            tokenized_outputs = text2sql_tokenizer(
                text_target=batch_sql_preprocessed,
                padding="max_length",
                return_tensors='pt',
                max_length=256,
                truncation=True,
                # is_split_into_words=True,
            )
            
            encoder_input_ids = tokenized_inputs["input_ids"]
            encoder_input_attention_mask = tokenized_inputs["attention_mask"]

            decoder_labels = tokenized_outputs["input_ids"]
            decoder_labels[decoder_labels == text2sql_tokenizer.pad_token_id] = -100
            decoder_attention_mask = tokenized_outputs["attention_mask"]

            if torch.cuda.is_available():
                encoder_input_ids = encoder_input_ids.cuda()
                encoder_input_attention_mask = encoder_input_attention_mask.cuda()
                decoder_labels = decoder_labels.cuda()
                decoder_attention_mask = decoder_attention_mask.cuda()
            
            if opt.model == "rgat" or opt.model == "rtransformer":
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True,
                    graph_batch = batch_graphs
                )
            else:
                model_outputs = model(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    labels = decoder_labels,
                    decoder_attention_mask = decoder_attention_mask,
                    return_dict = True
                )

                # encoder_last_hidden_state = model_outputs.encoder_last_hidden_state

            
            loss = model_outputs["loss"]
            loss.backward()

            if scheduler is not None:
                scheduler.step()

            if writer is not None:
                # record training loss (tensorboard)
                writer.add_scalar('train loss', loss.item(), train_step)
                # record learning rate (tensorboard)
                writer.add_scalar('train lr', optimizer.state_dict()['param_groups'][0]['lr'], train_step)
            
            pbar.set_description(f"train loss: {loss.item():.4f}, lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")

            if train_step % opt.gradient_descent_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            if train_step % num_checkpoint_steps == 0 and epoch >= 6:
                print(f"At {train_step} training step, save a checkpoint.")
                os.makedirs(opt.save_path, exist_ok = True)
                model.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))
                text2sql_tokenizer.save_pretrained(save_directory = opt.save_path + "/checkpoint-{}".format(train_step))

def _test(opt):
    set_seed(opt.seed)
    print(opt)

    import time
    start_time = time.time()
    
    if opt.target_type == "natsql":
        tables = json.load(open(opt.tables_for_natsql,'r'))
        table_dict = dict()
        for t in tables:
            table_dict[t["db_id"]] = t

    # initialize tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(
        opt.save_path,
        add_prefix_space = True
    )
    
    if isinstance(tokenizer, T5TokenizerFast):
        tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <")])
    
    dev_dataset = Text2SQLDataset(
        dir_ = opt.dev_filepath,
        mode = opt.mode,
        tokenizer = tokenizer,
        preprocessed_file = opt.dev_preprocessed_dataset
    )

    dev_dataloder = DataLoader(
        dev_dataset, 
        batch_size = opt.batch_size, 
        shuffle = False,
        collate_fn = lambda x: x,
        drop_last = False
    )

    config = AutoConfig.from_pretrained(
        opt.model_name_or_path
    )

    if opt.model == "transformer":
        model_class = MT5ForConditionalGeneration if "mt5" in opt.save_path else T5ForConditionalGeneration
        # initialize model
        model = model_class.from_pretrained(opt.save_path)
        model.resize_token_embeddings(len(tokenizer))

    elif opt.model == "rgat" or opt.model == "rtransformer":
        config = AutoConfig.from_pretrained(
            opt.save_path
        )
        config.structure_encoder = opt.model
        model = GraphLLModel(tokenizer, opt.save_path, config)
    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    predict_sqls = []
    for batch in tqdm(dev_dataloder):
        batch_inputs = [data[0] for data in batch]
        batch_db_ids = [data[1] for data in batch]
        batch_tc_original = [data[2] for data in batch]
        # batch_graphs = [map_graph_dict_to_cuda(data[3], ['graph']) for data in batch]

        # batch_inputs_preprocessed = [token_preprocessor.preprocess(sentence) for sentence in batch_inputs]
        # batch_inputs_tokens = [split_sentence_to_subtokens(sentence) for sentence in batch_inputs]

        tokenized_inputs = tokenizer(
            batch_inputs, 
            return_tensors="pt",
            padding = "max_length",
            max_length = 512,
            truncation = True,
            # is_split_into_words = True
        )
        
        encoder_input_ids = tokenized_inputs["input_ids"]
        encoder_input_attention_mask = tokenized_inputs["attention_mask"]
        if torch.cuda.is_available():
            encoder_input_ids = encoder_input_ids.cuda()
            encoder_input_attention_mask = encoder_input_attention_mask.cuda()

        with torch.no_grad():
            if opt.model == "rgat" or opt.model == "rtransformer":
                model_outputs = model.generate(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    max_length = 256,
                    decoder_start_token_id = model.config.decoder_start_token_id,
                    num_beams = opt.num_beams,
                    num_return_sequences = opt.num_return_sequences,
                    graph_batch = batch_graphs
                )
            else:
                model_outputs = model.generate(
                    input_ids = encoder_input_ids,
                    attention_mask = encoder_input_attention_mask,
                    max_length = 256,
                    decoder_start_token_id = model.config.decoder_start_token_id,
                    num_beams = opt.num_beams,
                    num_return_sequences = opt.num_return_sequences
                )

            model_outputs = model_outputs.view(len(batch_inputs), opt.num_return_sequences, model_outputs.shape[1])
            if opt.target_type == "sql":
                predict_sqls += decode_sqls(
                    opt.db_path, 
                    model_outputs, 
                    batch_db_ids, 
                    batch_inputs, 
                    tokenizer, 
                    batch_tc_original
                )
            elif opt.target_type == "natsql":
                pred_batch_sqls = decode_natsqls(
                    opt.db_path,
                    model_outputs,
                    batch_db_ids,
                    batch_inputs,
                    tokenizer,
                    batch_tc_original,
                    table_dict,
                    token_preprocessor
                )
                predict_sqls += pred_batch_sqls
            else:
                raise ValueError()
    
    new_dir = "/".join(opt.output.split("/")[:-1]).strip()
    if new_dir != "":
        os.makedirs(new_dir, exist_ok = True)

    # save results
    with open(opt.output, "w", encoding = 'utf-8') as f:
        for pred in predict_sqls:
            f.write(pred + "\n")
    
    end_time = time.time()
    print("Text-to-SQL inference spends {}s.".format(end_time-start_time))
    
    if opt.mode == "eval":
        # initialize evaluator
        evaluator = EvaluateTool()
        evaluator.register_golds(opt.original_dev_filepath, opt.db_path)
        spider_metric_result = evaluator.evaluate(predict_sqls)
        print('exact_match score: {}'.format(spider_metric_result["exact_match"]))
        print('exec score: {}'.format(spider_metric_result["exec"]))
        return spider_metric_result


if __name__ == "__main__":
    opt = parse_option()
    if opt.mode in ["train"]:
        _train(opt)
    elif opt.mode in ["eval", "test"]:
        _test(opt)
