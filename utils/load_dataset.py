import json
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from torch.utils.data import Dataset
import inflect
import stanfordnlp


class StringDealer:
    def __init__(self):
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse', lang='en')
        self.lemmatizer = WordNetLemmatizer()
        self.plural_engin = inflect.engine()
    
    def get_plural_form(self, word):
        return self.plural_engin.plural(word)
    
    def get_singular_form(self, word):
        return self.lemmatizer.lemmatize(word, 'n')
    
    def get_dependency(self, sentence):
        # nmod, nsubj, appos, conj
        # nsubj：名词性主语
        # nsubjpass：被动语态的名词性主语
        # csubj：从属的从句主语
        # csubjpass：被动语态的从属的从句主语
        # dobj：直接宾语
        # iobj：间接宾语
        # ccomp：从属的从句的主语
        # xcomp：空语元（通常指主语的补语）
        # advcl：状语从句修饰的动词
        # advmod：状语修饰的词语
        # amod：形容词修饰名词
        # det：限定词（冠词、代词等）
        # prep：介词关系
        # pobj：介词宾语
        # conj：连接关系，表示并列关系
        # dep：依赖关系，表示某种依赖但具体关系不明确
        # root：句子的根节点
        doc = self.nlp(sentence)
        for word in doc.sentences[0].words:
            print(f"index: {word.index.rjust(2)}\tword: {word.text.ljust(11)}\tgovernor index: {word.governor}\tgovernor: {(doc.sentences[0].words[word.governor-1].text if word.governor > 0 else 'root').ljust(11)}\tdeprel: {word.dependency_relation}")

        return doc.sentences[0].dependencies


class GraphProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.string_dealer = StringDealer()
        self.SPLIT_TOKEN_ID = self.tokenizer('|')['input_ids'][0]
        self.TABLE_COLUMN_SPLIT = self.tokenizer('.')['input_ids'][1]
        
        self.RELATION_LIST = ['question-question-dist-1', 'question-question-dist-1',
                              'question-table-exactmatch', 'question-table-partialmatch',
                              'table-column-pk', 'table-table-fkr', 'column-table-pk', 
                              'column-column-fkr', 'table-table-fk', 'column-column-fk', 
                              'question-column-exactmatch', 'question-column-partialmatch']
        
        self.RELATION_MAPPING = {v: k for k, v in enumerate(self.RELATION_LIST)}
    
    def get_all_indexes_of_values(self, input_list, value):
        return [i for i, v in enumerate(input_list) if v == value]
    
    def get_all_indexes_of_value_list(self, input_list, value_list, bias=0):
        res_indexes = []
        for i in range(len(input_list)):
            if input_list[i: i+len(value_list)] == value_list:
                res_indexes.append([j+bias for j in range(i, i+len(value_list))])
        return res_indexes
    
    def get_edge_tuple_list(self, edges, edge_type):
        edge_type_index = self.RELATION_MAPPING[edge_type]
        return [(edge[0], edge[1], edge_type_index) for edge in edges]
    
    def get_pair_index_based_on_pair_value(self, input_ids, split_token_index, value_list_1, value_list_2):
        question_value_index_list = self.get_all_indexes_of_value_list(input_ids[:split_token_index], value_list_1, 0)
        other_value_index_list = self.get_all_indexes_of_value_list(input_ids[split_token_index:], value_list_2, split_token_index)
        res = []
        for question_word_indexes in question_value_index_list:
            for other_word_indexes in other_value_index_list:
                for question_word_index in question_word_indexes:
                    for other_word_index in other_word_indexes:
                        res.append((question_word_index, other_word_index))
        return res
    
    def add_table_column_match(self, input_sequence, input_ids):
        question_table_exact_matching = []  # save token index [(question_token_index, table_token_index), (...)]
        question_column_exact_matching = []  # save token index

        question_table_partial_matching = []
        question_column_partial_matching = []

        question_sequence, other_sequence = input_sequence.split('|', 1)

        question_words = nltk.word_tokenize(question_sequence)
        question_words_pos = nltk.pos_tag(question_words)

        question_nn_words = set([word_pos[0] for word_pos in question_words_pos if word_pos[1] == 'NN'])
        question_nns_words = set([word_pos[0] for word_pos in question_words_pos if word_pos[1] == 'NNS'])

        question_nn_nns_words = []
        for word in question_nn_words:
            if word in other_sequence:
                question_nn_nns_words.append(word)
            else:
                nns_word = self.string_dealer.get_plural_form(word)
                if nns_word in other_sequence:
                    input_sequence = input_sequence.replace(word, nns_word)
                    question_nn_nns_words.append(nns_word)

        for nns_word in question_nns_words:
            if nns_word in other_sequence:
                question_nn_nns_words.append(nns_word)
            else:
                nn_word = self.string_dealer.get_singular_form(nns_word)
                if nn_word in other_sequence:
                    input_sequence = input_sequence.replace(nns_word, nn_word)
                    question_nn_nns_words.append(nn_word)

        other_sequence_list = other_sequence.split('|')
        column_list = []
        table_list = []
        for other_sequence_item in other_sequence_list:
            if '=' not in other_sequence_item:
                table_column_name_sequence = other_sequence_item.split(':')
                table_name, column_name_sequence = table_column_name_sequence[0], table_column_name_sequence[1]
                column_names = column_name_sequence.split(',')
                for column_name in column_names:
                    column_list.append(column_name.strip())
                table_list.append(table_name.strip())
            else:
                column_names = other_sequence_item.split('=')
                for column_name in column_names:
                    column_list.append(column_name.strip())
        
        column_set = set(column_list)
        table_set = set(table_list)

        input_ids = self.tokenizer(input_sequence)['input_ids']
        split_token_index = input_ids.index(self.SPLIT_TOKEN_ID)

        for nn_nns_word in question_nn_nns_words:
            nn_nns_word_token_ids = self.tokenizer(nn_nns_word)['input_ids'][:-1]
            for table_name in table_set:
                if table_name == nn_nns_word:
                    table_name_token_ids = self.tokenizer(table_name)['input_ids'][:-1]
                    q_t_exact_matching = self.get_pair_index_based_on_pair_value(input_ids, split_token_index, nn_nns_word_token_ids, table_name_token_ids)
                    question_table_exact_matching += q_t_exact_matching
                elif table_name in nn_nns_word or nn_nns_word in table_name:
                    table_name_token_ids = self.tokenizer(table_name)['input_ids'][:-1]
                    q_t_partial_matching = self.get_pair_index_based_on_pair_value(input_ids, split_token_index, nn_nns_word_token_ids, table_name_token_ids)
                    question_table_partial_matching += q_t_partial_matching
            for column_name_item in column_set:
                column_name = column_name_item.split('.')[1]
                if column_name == nn_nns_word:
                    column_name_token_ids = self.tokenizer(column_name_item)['input_ids'][:-1]
                    q_c_exact_matching = self.get_pair_index_based_on_pair_value(input_ids, split_token_index, nn_nns_word_token_ids, column_name_token_ids)
                    question_column_exact_matching += q_c_exact_matching
                elif nn_nns_word in column_name or column_name in nn_nns_word:
                    column_name_token_ids = self.tokenizer(column_name_item)['input_ids'][:-1]
                    q_c_exact_matching = self.get_pair_index_based_on_pair_value(input_ids, split_token_index, nn_nns_word_token_ids, column_name_token_ids)
                    question_column_partial_matching += q_c_exact_matching

        return self.get_edge_tuple_list(question_table_exact_matching, 'question-table-exactmatch') + \
                    self.get_edge_tuple_list(question_column_exact_matching, 'question-column-exactmatch') + \
                    self.get_edge_tuple_list(question_table_partial_matching, 'question-table-partialmatch') + \
                    self.get_edge_tuple_list(question_column_partial_matching, 'question-column-partialmatch'), input_ids, input_sequence
    
    def process_sequence(self, input_sequence):
        # split_input_sequence = [question, table 1, table 2, ..., foreign key 1, foreign key 2]
        # initialize
        split_input_sequence = input_sequence.split('|')
        input_ids = self.tokenizer(input_sequence)['input_ids']
        question_sequence = split_input_sequence[0]
        table_sequences, foreign_key_sequences = [], []
        for i in range(1, len(split_input_sequence)):
            if '=' in split_input_sequence[i]:
                foreign_key_sequences.append(split_input_sequence[i])
            else:
                table_sequences.append(split_input_sequence[i])

        question_table_column_match_edges, input_ids, input_sequence = self.add_table_column_match(input_sequence, input_ids)

        

        self.string_dealer.get_dependency(question_sequence)
        
        print('yes')



class ColumnAndTableClassifierDataset(Dataset):
    def __init__(
        self,
        dir_: str = None,
        use_contents: bool = True,
        add_fk_info: bool = True,
    ):
        super(ColumnAndTableClassifierDataset, self).__init__()

        self.questions: list[str] = []
        
        self.all_column_infos: list[list[list[str]]] = []
        self.all_column_labels: list[list[list[int]]] = []

        self.all_table_names: list[list[str]] = []
        self.all_table_labels: list[list[int]] = []
        
        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            column_names_in_one_db = []
            column_names_original_in_one_db = []
            extra_column_info_in_one_db = []
            column_labels_in_one_db = []

            table_names_in_one_db = []
            table_names_original_in_one_db = []
            table_labels_in_one_db = []

            for table_id in range(len(data["db_schema"])):
                column_names_original_in_one_db.append(data["db_schema"][table_id]["column_names_original"])
                table_names_original_in_one_db.append(data["db_schema"][table_id]["table_name_original"])

                table_names_in_one_db.append(data["db_schema"][table_id]["table_name"])
                table_labels_in_one_db.append(data["table_labels"][table_id])

                column_names_in_one_db.append(data["db_schema"][table_id]["column_names"])
                column_labels_in_one_db += data["column_labels"][table_id]
                
                extra_column_info = ["" for _ in range(len(data["db_schema"][table_id]["column_names"]))]
                if use_contents:
                    contents = data["db_schema"][table_id]["db_contents"]
                    for column_id, content in enumerate(contents):
                        if len(content) != 0:
                            extra_column_info[column_id] += " , ".join(content)
                extra_column_info_in_one_db.append(extra_column_info)
            
            if add_fk_info:
                table_column_id_list = []
                # add a [FK] identifier to foreign keys
                for fk in data["fk"]:
                    source_table_name_original = fk["source_table_name_original"]
                    source_column_name_original = fk["source_column_name_original"]
                    target_table_name_original = fk["target_table_name_original"]
                    target_column_name_original = fk["target_column_name_original"]

                    if source_table_name_original in table_names_original_in_one_db:
                        source_table_id = table_names_original_in_one_db.index(source_table_name_original)
                        source_column_id = column_names_original_in_one_db[source_table_id].index(source_column_name_original)
                        if [source_table_id, source_column_id] not in table_column_id_list:
                            table_column_id_list.append([source_table_id, source_column_id])
                    
                    if target_table_name_original in table_names_original_in_one_db:
                        target_table_id = table_names_original_in_one_db.index(target_table_name_original)
                        target_column_id = column_names_original_in_one_db[target_table_id].index(target_column_name_original)
                        if [target_table_id, target_column_id] not in table_column_id_list:
                            table_column_id_list.append([target_table_id, target_column_id])
                
                for table_id, column_id in table_column_id_list:
                    if extra_column_info_in_one_db[table_id][column_id] != "":
                        extra_column_info_in_one_db[table_id][column_id] += " , [FK]"
                    else:
                        extra_column_info_in_one_db[table_id][column_id] += "[FK]"
            
            # column_info = column name + extra column info
            column_infos_in_one_db = []
            for table_id in range(len(table_names_in_one_db)):
                column_infos_in_one_table = []
                for column_name, extra_column_info in zip(column_names_in_one_db[table_id], extra_column_info_in_one_db[table_id]):
                    if len(extra_column_info) != 0:
                        column_infos_in_one_table.append(column_name + " ( " + extra_column_info + " ) ")
                    else:
                        column_infos_in_one_table.append(column_name)
                column_infos_in_one_db.append(column_infos_in_one_table)
            
            self.questions.append(data["question"])
            
            self.all_table_names.append(table_names_in_one_db)
            self.all_table_labels.append(table_labels_in_one_db)

            self.all_column_infos.append(column_infos_in_one_db)
            self.all_column_labels.append(column_labels_in_one_db)
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, index):
        question = self.questions[index]

        table_names_in_one_db = self.all_table_names[index]
        table_labels_in_one_db = self.all_table_labels[index]

        column_infos_in_one_db = self.all_column_infos[index]
        column_labels_in_one_db = self.all_column_labels[index]

        return question, table_names_in_one_db, table_labels_in_one_db, column_infos_in_one_db, column_labels_in_one_db

class Text2SQLDataset(Dataset):
    def __init__(
        self,
        dir_: str,
        mode: str,
        tokenizer: any = None,
    ):
        super(Text2SQLDataset).__init__()
        
        self.mode = mode
        self.tokenizer = tokenizer
        self.graph_processor = GraphProcessor(tokenizer)

        self.input_sequences: list[str] = []
        self.output_sequences: list[str] = []
        self.db_ids: list[str] = []
        self.all_tc_original: list[list[str]] = []

        with open(dir_, 'r', encoding = 'utf-8') as f:
            dataset = json.load(f)
        
        for data in dataset:
            self.input_sequences.append(data["input_sequence"])
            self.db_ids.append(data["db_id"])
            self.all_tc_original.append(data["tc_original"])

            self._process_graph(data["input_sequence"])

            if self.mode == "train":
                self.output_sequences.append(data["output_sequence"])
            elif self.mode in ["eval", "test"]:
                pass
            else:
                raise ValueError("Invalid mode. Please choose from ``train``, ``eval`, and ``test``")
    
    def _process_graph(self, input_sequence):
        return self.graph_processor.process_sequence(input_sequence)

    def __len__(self):
        return len(self.input_sequences)
    
    def __getitem__(self, index):
        if self.mode == "train":
            return self.input_sequences[index], self.output_sequences[index], self.db_ids[index], self.all_tc_original[index]
        elif self.mode in ['eval', "test"]:
            return self.input_sequences[index], self.db_ids[index], self.all_tc_original[index]