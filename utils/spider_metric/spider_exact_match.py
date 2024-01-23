"""Spider Exact Match metric."""
from typing import Dict, Any
from third_party.spider import evaluation as spider_evaluation

def compute_exact_match_metric(predictions, references) -> Dict[str, Any]:
    foreign_key_maps = dict()
    false_pairs = []
    for reference in references:
        if reference["db_id"] not in foreign_key_maps:
            foreign_key_maps[reference["db_id"]] = spider_evaluation.build_foreign_key_map(
                {
                    "table_names_original": reference["db_table_names"],
                    "column_names_original": list(
                        zip(
                            reference["db_column_names"]["table_id"],
                            reference["db_column_names"]["column_name"],
                        )
                    ),
                    "foreign_keys": list(
                        zip(
                            reference["db_foreign_keys"]["column_id"],
                            reference["db_foreign_keys"]["other_column_id"],
                        )
                    ),
                }
            )
    evaluator = spider_evaluation.Evaluator(references[0]["db_path"], foreign_key_maps, "match")
    for prediction, reference in zip(predictions, references):
        turn_idx = reference.get("turn_idx", 0)
        # skip final utterance-query pairs
        if turn_idx < 0:
            continue
        one_res = evaluator.evaluate_one(reference["db_id"], reference["query"], prediction)
        if not one_res['exact']:
            false_pairs.append((reference["db_id"], reference["query"], prediction))
    evaluator.finalize()
    return {
        "exact_match": evaluator.scores["all"]["exact"],
        "exact_match_easy": evaluator.scores["easy"]["exact"],
        "exact_match_medium": evaluator.scores["medium"]["exact"],
        "exact_match_hard": evaluator.scores["hard"]["exact"],
        "exact_match_extra": evaluator.scores["extra"]["exact"]
    }
