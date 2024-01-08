import torch
import torch.nn as nn

from transformers import AutoConfig, RobertaModel, XLMRobertaModel
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence



class MyClassifier_Share(nn.Module):
    def __init__(
        self,
        model_name_or_path,
        vocab_size,
        mode
    ):
        super(MyClassifier_Share, self).__init__()
        model_class = XLMRobertaModel if "xlm" in model_name_or_path else RobertaModel
        if mode in ["eval", "test"]:
            # load config
            config = AutoConfig.from_pretrained(model_name_or_path)
            # randomly initialize model's parameters according to the config
            self.plm_encoder = model_class(config)
        elif mode == "train":
            self.plm_encoder = model_class.from_pretrained(model_name_or_path)
            self.plm_encoder.resize_token_embeddings(vocab_size)
        else:
            raise ValueError()

        # column cls head
        self.column_info_cls_head_linear1 = nn.Linear(1024, 256)
        self.column_info_cls_head_linear2 = nn.Linear(256, 2)
        
        # column bi-lstm layer
        self.column_info_bilstm = nn.LSTM(
            input_size = 1024,
            hidden_size = 512,
            num_layers = 2,
            dropout = 0,
            bidirectional = True
        )

        # linear layer after column bi-lstm layer
        self.column_info_linear_after_pooling = nn.Linear(1024, 1024)

        # table cls head
        self.table_name_cls_head_linear1 = nn.Linear(1024, 256)
        self.table_name_cls_head_linear2 = nn.Linear(256, 2)
        
        # table bi-lstm pooling layer
        self.table_name_bilstm = nn.LSTM(
            input_size = 1024,
            hidden_size = 512,
            num_layers = 2,
            dropout = 0,
            bidirectional = True
        )
        # linear layer after table bi-lstm layer
        self.table_name_linear_after_pooling = nn.Linear(1024, 1024)

        # activation function
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.question_cls_head_linear = nn.Linear(1024, 256)

        # table-column cross-attention layer
        self.table_column_cross_attention_layer = nn.MultiheadAttention(embed_dim = 1024, num_heads = 8)

        # dropout function, p=0.2 means randomly set 20% neurons to 0
        self.dropout = nn.Dropout(p = 0.2)
    
    def table_column_cross_attention(
        self,
        table_name_embeddings_in_one_db, 
        column_info_embeddings_in_one_db, 
        column_number_in_each_table
    ):
        table_num = table_name_embeddings_in_one_db.shape[0]
        table_name_embedding_attn_list = []
        for table_id in range(table_num):
            table_name_embedding = table_name_embeddings_in_one_db[[table_id], :]
            column_info_embeddings_in_one_table = column_info_embeddings_in_one_db[
                sum(column_number_in_each_table[:table_id]) : sum(column_number_in_each_table[:table_id+1]), :]
            
            table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                table_name_embedding,
                column_info_embeddings_in_one_table,
                column_info_embeddings_in_one_table
            )

            table_name_embedding_attn_list.append(table_name_embedding_attn)
        
        # residual connection
        table_name_embeddings_in_one_db = table_name_embeddings_in_one_db + torch.cat(table_name_embedding_attn_list, dim = 0)
        # row-wise L2 norm
        table_name_embeddings_in_one_db = torch.nn.functional.normalize(table_name_embeddings_in_one_db, p=2.0, dim=1)

        return table_name_embeddings_in_one_db
    
    def table_column_cross_attention_one_table(
        self,
        table_name_embedding,
        column_info_embeddings,
    ):
        table_name_embedding_attn, _ = self.table_column_cross_attention_layer(
                                            table_name_embedding,
                                            column_info_embeddings,
                                            column_info_embeddings
                                        )
        
        # residual connection
        table_name_embedding = table_name_embedding + table_name_embedding_attn
        table_name_embedding = torch.nn.functional.normalize(table_name_embedding, p=2.0, dim=1)
        return table_name_embedding

    def batch_lstm_forward(
        self,
        embeddings_list,
        lstm_layer
    ):
        # output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
        # table_name_embedding = hidden_state_t[-2:, :].view(1, 1024)
        packed_sequence = pack_sequence(embeddings_list, enforce_sorted=False)
        packed_sequence_output, _ = lstm_layer(packed_sequence)
        output, output_size = pad_packed_sequence(packed_sequence_output)
        result_emb = []
        for i in range(len(embeddings_list)):
            result_emb.append(output[output_size[i]-1, i, :].view(1, 1024))
        return result_emb

    def table_column_cls(
        self,
        tokenized_questions_inputs,
        tokenized_schema_inputs_batch,
        batch_aligned_column_info_ids,
        batch_aligned_table_name_ids,
        batch_column_number_in_each_table
    ):
        batch_size = len(batch_column_number_in_each_table)

        encoder_question_output = self.plm_encoder(
            input_ids=tokenized_questions_inputs["input_ids"],
            attention_mask=tokenized_questions_inputs["attention_mask"],
            return_dict=True
        )['pooler_output']

        encoder_question_output_cls = self.question_cls_head_linear(encoder_question_output)
        encoder_question_output_cls = self.dropout(self.leakyrelu(encoder_question_output_cls))

        encoder_schema_output = [self.plm_encoder(
            input_ids=tokenized_schema_inputs["input_ids"],
            attention_mask=tokenized_schema_inputs["attention_mask"],
            return_dict=True
        ) for tokenized_schema_inputs in tokenized_schema_inputs_batch]

        batch_table_name_cls_logits, batch_column_info_cls_logits = [], []

        # handle each data in current batch
        for batch_id in range(batch_size):
            column_number_in_each_table = batch_column_number_in_each_table[batch_id]
            schema_embeddings = encoder_schema_output[batch_id]["last_hidden_state"] # (table_num * column_num * hidden_size)
            # obtain the embeddings of tokens in the question
            question_token_embeddings = encoder_question_output[batch_id]

            # obtain table ids for each table
            aligned_table_name_ids = batch_aligned_table_name_ids[batch_id]
            # obtain column ids for each column
            aligned_column_info_ids = batch_aligned_column_info_ids[batch_id]

            table_name_embedding_list, column_info_embedding_list = [], []

            # obtain table embedding via bi-lstm pooling + a non-linear layer
            for t_id, table_name_ids in enumerate(aligned_table_name_ids):
                table_name_embeddings = schema_embeddings[t_id, table_name_ids, :]
                
                # BiLSTM pooling
                # output_t, (hidden_state_t, cell_state_t) = self.table_name_bilstm(table_name_embeddings)
                # table_name_embedding = hidden_state_t[-2:, :].view(1, 1024)
                table_name_embedding_list.append(table_name_embeddings)
            table_name_embedding_list = self.batch_lstm_forward(table_name_embedding_list, self.table_name_bilstm)
            table_name_embeddings_in_one_db = torch.cat(table_name_embedding_list, dim = 0)
            # non-linear mlp layer
            table_name_embeddings_in_one_db = self.leakyrelu(self.table_name_linear_after_pooling(table_name_embeddings_in_one_db))
            
            column_info_embeddings_in_one_db = []

            # obtain column embedding via bi-lstm pooling + a non-linear layer
            for t_id, column_info_ids_table in enumerate(aligned_column_info_ids):
                column_info_embedding_list_table = []
                for column_info_ids in column_info_ids_table:
                    column_info_embeddings = schema_embeddings[t_id, column_info_ids, :]
                    
                    # BiLSTM pooling
                    # output_c, (hidden_state_c, cell_state_c) = self.column_info_bilstm(column_info_embeddings)
                    # column_info_embedding = hidden_state_c[-2:, :].view(1, 1024)
                    column_info_embedding_list_table.append(column_info_embeddings)
                column_info_embedding_list_table = self.batch_lstm_forward(column_info_embedding_list_table, self.column_info_bilstm)
                column_info_embedding_in_one_table = torch.cat(column_info_embedding_list_table, dim = 0)
                column_info_embedding_in_one_table = self.leakyrelu(self.column_info_linear_after_pooling(column_info_embedding_in_one_table))
                column_info_embeddings_in_one_db.append(column_info_embedding_in_one_table)

                # table-column (tc) cross-attention
                table_name_embeddings_in_one_db[t_id] = self.table_column_cross_attention_one_table(
                    table_name_embeddings_in_one_db[[t_id]],
                    column_info_embedding_in_one_table
                )
                # table_name_embeddings_in_one_db = self.table_column_cross_attention(
                #     table_name_embeddings_in_one_db, 
                #     column_info_embeddings_in_one_db, 
                #     column_number_in_each_table
                # )
            
            # calculate table 0-1 logits
            table_name_embeddings_in_one_db = self.table_name_cls_head_linear1(table_name_embeddings_in_one_db)
            table_name_embeddings_in_one_db = self.dropout(self.leakyrelu(table_name_embeddings_in_one_db))
            # table_name_cls_logits = self.table_name_cls_head_linear2(table_name_embeddings_in_one_db)
            table_name_cls_logits = self.cos(table_name_embeddings_in_one_db, encoder_question_output_cls[batch_id])

            # calculate column 0-1 logits
            column_info_embeddings_in_one_db = torch.cat(column_info_embeddings_in_one_db, dim = 0)
            column_info_embeddings_in_one_db = self.column_info_cls_head_linear1(column_info_embeddings_in_one_db)
            column_info_embeddings_in_one_db = self.dropout(self.leakyrelu(column_info_embeddings_in_one_db))
            # column_info_cls_logits = self.column_info_cls_head_linear2(column_info_embeddings_in_one_db)
            column_info_cls_logits = self.cos(column_info_embeddings_in_one_db, encoder_question_output_cls[batch_id])

            batch_table_name_cls_logits.append(table_name_cls_logits)
            batch_column_info_cls_logits.append(column_info_cls_logits)

        return batch_table_name_cls_logits, batch_column_info_cls_logits

    def forward(
        self,
        tokenized_questions_inputs,
        tokenized_schema_inputs_batch,
        batch_aligned_column_info_ids,
        batch_aligned_table_name_ids,
        batch_column_number_in_each_table,
    ):  
        batch_table_name_cls_logits, batch_column_info_cls_logits \
            = self.table_column_cls(
            tokenized_questions_inputs,
            tokenized_schema_inputs_batch,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table
        )

        return {
            "batch_table_name_cls_logits" : batch_table_name_cls_logits, 
            "batch_column_info_cls_logits": batch_column_info_cls_logits
        }
