import json
import optuna
import numpy as np
import xgboost as xgb

from simulation import buy_simulation

import SekitobaLibrary as lib
import SekitobaDataManage as dm
from learn import data_adjustment
from learn.const import *

use_data = {}
use_simu_data = {}
optuna_test_years = []

def xg_objective( trial ):
    qid_data = []
    valid_qid_data = []

    for i, query in enumerate( use_data["query"] ):
        qid_data.extend( [i+1] * query )

    for i, query in enumerate( use_data["test_query"] ):
        valid_qid_data.extend( [i+1] * query )

    max_depth = trial.suggest_int( "max_depth", 2, 7 )    
    learning_rate = trial.suggest_float( 'learning_rate', 0.01, 0.1 )
    min_child_weight = trial.suggest_int( "min_child_weight", 1, 10 )
    subsample = trial.suggest_float( 'subsumple', 0.75, 0.85 )
    colsample_bytree = trial.suggest_float( 'colsample_bytree', 0.6, 0.9 )

    model = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:ndcg", # または rank:pairwise
        eval_metric='ndcg@2',
        lambdarank_num_pair_per_sample=200,
        ndcg_exp_relevance=True,
        n_estimators=1000,
        gamma=0,
        early_stopping_rounds=30,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree)


    model.fit( np.array( use_data["teacher"] ), \
               np.array( use_data["answer"] ), \
               qid = np.array( qid_data ), \
               eval_set = [( np.array( use_data["test_teacher"] ), np.array( use_data["test_answer"] ) )], \
               eval_qid = [np.array( valid_qid_data )], \
               verbose = 10 )
    
    one_win_rate, three_win_rate, mdcd_score = \
        buy_simulation.main( [ model ], use_simu_data, test_years = lib.score_years, show = True )
    score = 0
    score += one_win_rate
    score += three_win_rate
    score -= mdcd_score
    score *= -1

    return score

def xg_optuna_main( data, simu_data ):
    global use_data
    global use_simu_data
    global optuna_test_years

    use_simu_data = simu_data        
    use_data = data_adjustment.data_check( data, state = "optuna" )
    best_parames_list = []

    for i in range( 0, XG_MODEL_NUM ):
        study = optuna.create_study()
        study.optimize(xg_objective, n_trials=OPTUNA_TRAIALS)
        best_parames_list.append( study.best_params )

    f = open( XG_BEST_PARAMS_FILE, "w" )
    json.dump( best_parames_list, f )
    f.close()
