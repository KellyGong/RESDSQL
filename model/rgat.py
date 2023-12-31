#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from transformers import AutoTokenizer
from collections import defaultdict
import pickle
import torch
import pdb

import dgl
from tqdm import tqdm

from torch import nn
from transformers import AutoTokenizer
from .modeling_auto import AutoModelForSeq2SeqLM
from transformers import PreTrainedModel


class GraphLLModel(PreTrainedModel):
    def __init__(self, tokenizer, path, config):
        super().__init__(config)

        # Load tokenizer and model.
        self.tokenizer = tokenizer

        # import graph part:
        # self.graph_pedia = graph_pedia

        self.max_in_degree, self.max_out_degree = 0, 0
        self.max_path_length = 0

        # self.filtered_graph_postprocess_all()

        # config.max_in_degree = self.max_in_degree
        # config.max_out_degree = self.max_out_degree
        # config.max_path_length = self.max_path_length

        # self.add_graph_feature()

        self.pretrain_model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            config=config
        )

        self.pretrain_model.resize_token_embeddings(len(self.tokenizer))

        self.config = self.pretrain_model.config
        from .modeling_t5 import T5ForConditionalGeneration
    
    def add_graph_feature(self):
        print('add graph post feature, path weight...')
        for graph_data in tqdm(self.graph_pedia):
            dist = graph_data['dist']
            dist[dist==-1] = self.max_path_length + 1
            graph_data['dist'] = dist
            graph_data['path_edge_type'] = (graph_data['graph'].edata['type'])[graph_data['paths']]
            path_average_weight = torch.where(graph_data['paths']!=-1, 1.0, 0.0)
            path_average_weight /= (torch.sum(path_average_weight, dim=-1, keepdim=True) + 1e-8)
            graph_data['path_average_weight'] = path_average_weight
    
    def filtered_graph_postprocess_all(self):
        self.new_graph_pedia = []
        for i, graph in tqdm(self.graph_pedia.items()):
            self.new_graph_pedia.append({
                'graph': graph['graph'],
                'dist': graph['dist'],
                'paths': graph['paths'],
                'in_degree': graph['in_degree'],
                'out_degree': graph['out_degree']
            })
            self.max_in_degree = max(self.max_in_degree, int(graph['in_degree'].max()))
            self.max_out_degree = max(self.max_out_degree, int(graph['out_degree'].max()))
            self.max_path_length = max(self.max_path_length, int(graph['dist'].max()))

        del self.graph_pedia
        self.graph_pedia = self.new_graph_pedia
    
    def enumerate_relation(self, relations):
        word2id = {}
        id2word = {}

        for i, r in enumerate(relations):
            word2id[r] = i
            id2word[i] = r

        return word2id, id2word

    def get_graph(self, kwargs):
        ''' load graph_idx and convert into list '''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx[0]) for idx in graph_idx_batch]
        '''load graph_node_idx and convert into list'''
        graph_node_idx_batch = kwargs.pop('graph_nodes_subwords_idx', None)

        new_graph_batch = [] # list of dicts
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_pedia[graph_idx]
            new_graph_batch.append(new_graph)

        return new_graph_batch

    def graph_factory(self, kwargs):
        '''load and postporcess graphs'''
        graph_idx_batch = kwargs.pop('graph_idx', None)
        device = graph_idx_batch.device
        graph_idx_batch_lst = [int(idx) for idx in graph_idx_batch]

        new_graph_batch = []
        for i, graph_idx in enumerate(graph_idx_batch_lst):
            new_graph = self.graph_pedia[graph_idx]
            new_graph_batch.append(new_graph)

        return new_graph_batch

    def forward(self, input_ids, attention_mask, labels, **kwargs):
        graph_batch = kwargs['sequence_graphs']
        # self.relation_init_prompt(self.rel2id)
        loss = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            labels=labels,
            graph_batch=graph_batch,
            # relation_embedding = self.relation_embedding
        ).loss
        # for graph in graph_batch:
        #     graph.ndata.pop('x')
        #     graph.edata.pop('e')
        if torch.isnan(loss).sum() != 0: pdb.set_trace()
        return {'loss': loss}

    def generate(self, input_ids, attention_mask, **kwargs):
        graph_batch = self.graph_factory(kwargs)
        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            graph_batch=graph_batch,
            # relation_embedding=self.relation_embedding
            **kwargs,
        )

        return generated_ids
