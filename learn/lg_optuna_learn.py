import json
import optuna
import numpy as np
import lightgbm as lgb

from simulation import buy_simulation

import SekitobaLibrary as lib
import SekitobaDataManage as dm
from learn import data_adjustment
from learn.const import *

use_data = {}
use_simu_data = {}
optuna_test_years = []

def lg_objective( trial ):
    max_pos = np.max( np.array( use_data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( use_data["teacher"] ), np.array( use_data["answer"] ), group = np.array( use_data["query"] ) )
    lgb_vaild = lgb.Dataset( np.array( use_data["test_teacher"] ), np.array( use_data["test_answer"] ), group = np.array( use_data["test_query"] ) )

    learning_rate = trial.suggest_float( 'learning_rate', 0.01, 0.1 )
    num_leaves =  trial.suggest_int( "num_leaves", 15, 30 )
    max_depth = trial.suggest_int( "max_depth", 4, 6 )
    min_data_in_leaf = trial.suggest_int( "min_data_in_leaf", 5, 30 )
    lambda_l1 = trial.suggest_float( "lambda_l1", 0, 0.1 )
    lambda_l2 = trial.suggest_float( "lambda_l2", 0, 0.1 )
    feature_fraction = trial.suggest_float( "feature_fraction", 0.8, 0.9 )

    lgbm_params =  {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        #'objective': 'rank_xendcg',
        'metric': 'ndcg',   # for lambdarank
        'eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( use_data["answer"], dtype = np.int32 ) ) + 1)),
        'lambdarank_truncation_level': 2,
        'early_stopping_rounds': 30,
        'lambdarank_norm': True,
        'num_iteration': 3000,
        'learning_rate': learning_rate,
        'min_data_in_bin': 1,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2,
        'feature_fraction': feature_fraction
    }

    model = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     num_boost_round = 5000 )

    one_win_rate, three_win_rate, mdcd_score = \
        buy_simulation.main( [ model ], use_simu_data, test_years = lib.score_years, show = True )
    score = 0
    score += one_win_rate
    score += three_win_rate
    score -= mdcd_score
    score *= -1

    return score

def lg_optuna_main( data, simu_data ):
    global use_data
    global use_simu_data
    global optuna_test_years

    use_simu_data = simu_data        
    use_data = data_adjustment.data_check( data, state = "optuna" )
    best_parames_list = []

    for i in range( 0, LG_MODEL_NUM ):
        study = optuna.create_study()
        study.optimize(lg_objective, n_trials=OPTUNA_TRAIALS)
        best_parames_list.append( study.best_params )

    f = open( LG_BEST_PARAMS_FILE, "w" )
    json.dump( best_parames_list, f )
    f.close()
