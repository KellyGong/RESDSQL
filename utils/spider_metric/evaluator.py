# encoding=utf8
import json
import os
from third_party.spider.preprocess.get_tables import dump_db_json_schema
from .spider_exact_match import compute_exact_match_metric
from .spider_test_suite import compute_test_suite_metric
from collections import defaultdict


WHERE_OPS = (
    "not",
    "between",
    "=",
    ">",
    "<",
    ">=",
    "<=",
    "!=",
    "in",
    "like",
    "is",
    "exists",
)

AGG_OPS = ("none", "max", "min", "count", "sum", "avg")


def count_component1(sql):
    count = 0
    if len(sql["where"]) > 0:
        count += 1
    if len(sql["groupBy"]) > 0:
        count += 1
    if len(sql["orderBy"]) > 0:
        count += 1
    if sql["limit"] is not None:
        count += 1
    if len(sql["from"]["table_units"]) > 0:  # JOIN
        count += len(sql["from"]["table_units"]) - 1

    ao = sql["from"]["conds"][1::2] + sql["where"][1::2] + sql["having"][1::2]
    count += len([token for token in ao if token == "or"])
    cond_units = sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]
    count += len(
        [
            cond_unit
            for cond_unit in cond_units
            if cond_unit[1] == WHERE_OPS.index("like")
        ]
    )

    return count


def get_nestedSQL(sql):
    nested = []
    for cond_unit in sql["from"]["conds"][::2] + sql["where"][::2] + sql["having"][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql["intersect"] is not None:
        nested.append(sql["intersect"])
    if sql["except"] is not None:
        nested.append(sql["except"])
    if sql["union"] is not None:
        nested.append(sql["union"])
    return nested


def has_agg(unit):
    return unit[0] != AGG_OPS.index("none")


def count_agg(units):
    return len([unit for unit in units if has_agg(unit)])


def count_others(sql):
    count = 0
    # number of aggregation
    agg_count = count_agg(sql["select"][1])
    agg_count += count_agg(sql["where"][::2])
    agg_count += count_agg(sql["groupBy"])
    if len(sql["orderBy"]) > 0:
        agg_count += count_agg(
            [unit[1] for unit in sql["orderBy"][1] if unit[1]]
            + [unit[2] for unit in sql["orderBy"][1] if unit[2]]
        )
    agg_count += count_agg(sql["having"])
    if agg_count > 1:
        count += 1

    # number of select columns
    if len(sql["select"][1]) > 1:
        count += 1

    # number of where conditions
    if len(sql["where"]) > 1:
        count += 1

    # number of group by clauses
    if len(sql["groupBy"]) > 1:
        count += 1

    return count


def count_component2(sql):
    nested = get_nestedSQL(sql)
    return len(nested)


class EvaluateTool(object):
    def __init__(self):
        # self.args = args
        self.schema_cache = dict()
        self.golds = []
        # self.difficulty2id_list = defaultdict(list)

    def register_golds(self, dataset_filepath, db_path):
        with open(dataset_filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            for idx, sample in enumerate(dataset):
                # self.difficulty2id_list[self.eval_hardness(sample["sql"])].append(idx)

                if sample['query'] == 'SELECT T1.company_name FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id JOIN Ref_Company_Types AS T3 ON T1.company_type_code  =  T3.company_type_code ORDER BY T2.contract_end_date DESC LIMIT 1':
                    sample['query'] = 'SELECT T1.company_type FROM Third_Party_Companies AS T1 JOIN Maintenance_Contracts AS T2 ON T1.company_id  =  T2.maintenance_contract_company_id ORDER BY T2.contract_end_date DESC LIMIT 1'
                    sample['query_toks'] = ['SELECT', 'T1.company_type', 'FROM', 'Third_Party_Companies', 'AS', 'T1', 'JOIN', 'Maintenance_Contracts', 'AS', 'T2', 'ON', 'T1.company_id', '=', 'T2.maintenance_contract_company_id', 'ORDER', 'BY', 'T2.contract_end_date', 'DESC', 'LIMIT', '1']
                    sample['query_toks_no_value'] =  ['select', 't1', '.', 'company_type', 'from', 'third_party_companies', 'as', 't1', 'join', 'maintenance_contracts', 'as', 't2', 'on', 't1', '.', 'company_id', '=', 't2', '.', 'maintenance_contract_company_id', 'order', 'by', 't2', '.', 'contract_end_date', 'desc', 'limit', 'value']
                    sample['question'] = 'What is the type of the company who concluded its contracts most recently?'
                    sample['question_toks'] = ['What', 'is', 'the', 'type', 'of', 'the', 'company', 'who', 'concluded', 'its', 'contracts', 'most', 'recently', '?']
                if sample['query'].startswith('SELECT T1.fname FROM student AS T1 JOIN lives_in AS T2 ON T1.stuid  =  T2.stuid WHERE T2.dormid IN'):
                    sample['query'] = sample['query'].replace('IN (SELECT T2.dormid)', 'IN (SELECT T3.dormid)')
                    index = sample['query_toks'].index('(') + 2
                    assert sample['query_toks'][index] == 'T2.dormid'
                    sample['query_toks'][index] = 'T3.dormid'
                    index = sample['query_toks_no_value'].index('(') + 2
                    assert sample['query_toks_no_value'][index] == 't2'
                    sample['query_toks_no_value'][index] = 't3'
    
                db_id = sample["db_id"]
                if db_id not in self.schema_cache:
                    self.schema_cache[db_id] = dump_db_json_schema(
                        db=os.path.join(db_path, db_id, f"{db_id}.sqlite"), f=db_id
                    )
                schema = self.schema_cache[db_id]

                self.golds.append({
                    "query": sample["query"],
                    "question": sample["question"],
                    "db_id": db_id,
                    "db_path": db_path,
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": {
                        "table_id": [table_id for table_id, _ in schema["column_names_original"]],
                        "column_name": [column_name for _, column_name in schema["column_names_original"]]
                    },
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": {
                        "column_id": [column_id for column_id, _ in schema["foreign_keys"]],
                        "other_column_id": [other_column_id for _, other_column_id in schema["foreign_keys"]]
                    },
                })

    def eval_hardness(self, sql):
        count_comp1_ = count_component1(sql)
        count_comp2_ = count_component2(sql)
        count_others_ = count_others(sql)

        if count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ == 0:
            return "easy"
        elif (count_others_ <= 2 and count_comp1_ <= 1 and count_comp2_ == 0) or (
            count_comp1_ <= 2 and count_others_ < 2 and count_comp2_ == 0
        ):
            return "medium"
        elif (
            (count_others_ > 2 and count_comp1_ <= 2 and count_comp2_ == 0)
            or (2 < count_comp1_ <= 3 and count_others_ <= 2 and count_comp2_ == 0)
            or (count_comp1_ <= 1 and count_others_ == 0 and count_comp2_ <= 1)
        ):
            return "hard"
        else:
            return "extra"

    def evaluate(self, preds):
        exact_match = compute_exact_match_metric(preds, self.golds)
        test_suite = compute_test_suite_metric(preds, self.golds, db_dir = None)
        
        return {**exact_match, **test_suite}
