import optuna
import numpy as np
import lightgbm as lgb

from simulation import buy_simulation

import sekitoba_library as lib
import sekitoba_data_manage as dm

use_data = {}
use_simu_data = {}

def objective( trial ):
    max_pos = np.max( np.array( use_data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( use_data["teacher"] ), np.array( use_data["answer"] ), group = np.array( use_data["query"] ) )
    lgb_vaild = lgb.Dataset( np.array( use_data["test_teacher"] ), np.array( use_data["test_answer"] ), group = np.array( use_data["test_query"] ) )

    learning_rate = trial.suggest_float( 'learning_rate', 0.01, 0.1 )
    num_leaves =  trial.suggest_int( "num_leaves", 50, 300 )
    max_depth = trial.suggest_int( "max_depth", 50, 300 )
    num_iteration = trial.suggest_int( "num_iteration", 300, 3000 )
    min_data_in_leaf = trial.suggest_int( "min_data_in_leaf", 10, 100 )
    lambda_l1 = trial.suggest_float( "lambda_l1", 0, 1 )
    lambda_l2 = trial.suggest_float( "lambda_l2", 0, 1 )
    
    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',   # for lambdarank
        'ndcg_eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( use_data["answer"], dtype = np.int32 ) ) + 1)),
        'max_position': int( max_pos ),  # for lambdarank
        'early_stopping_rounds': 30,
        'learning_rate': learning_rate,
        'num_iteration': num_iteration,
        'min_data_in_bin': 1,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_data_in_leaf': min_data_in_leaf,
        'lambda_l1': lambda_l1,
        'lambda_l2': lambda_l2
    }

    model = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )

    models = { "rank": model }
    one_win_rate, three_win_rate, mdcd_score = buy_simulation.main( models, use_simu_data, show = True )
    score = ( one_win_rate + three_win_rate ) - mdcd_score
    score *= -1

    return score

def best_model_create( param_data ):
    max_pos = np.max( np.array( use_data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( use_data["teacher"] ), np.array( use_data["answer"] ), group = np.array( use_data["query"] ) )
    lgb_vaild = lgb.Dataset( np.array( use_data["test_teacher"] ), np.array( use_data["test_answer"] ), group = np.array( use_data["test_query"] ) )

    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',   # for lambdarank
        'ndcg_eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( use_data["answer"], dtype = np.int32 ) ) + 1)),
        'max_position': int( max_pos ),  # for lambdarank
        'early_stopping_rounds': 30,
        'learning_rate': param_data["learning_rate"],
        'num_iteration': param_data["num_iteration"],
        'min_data_in_bin': 1,
        'max_depth': param_data["max_depth"],
        'num_leaves': param_data["num_leaves"],
        'min_data_in_leaf': param_data["min_data_in_leaf"],
        'lambda_l1': param_data["lambda_l1"],
        'lambda_l2': param_data["lambda_l2"]
    }

    model = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )

    models = { "rank": model }
    buy_simulation.main( models, use_simu_data, show = True )

    dm.pickle_upload( lib.name.model_name(), model )

def data_check( data, prod = False ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    result["query"] = []
    result["test_query"] = []

    count = 0

    for i in range( 0, len( data["query"] ) ):
        q = data["query"][i]["q"]
        year = data["query"][i]["year"]

        if q < 3:
            continue

        current_data = list( data["teacher"][i] )
        current_answer = list( data["answer"][i] )
        current_level = list( data["level"][i] )
        current_diff = list( data["diff"][i] )
        current_popular = list( data["popular"][i] )

        if 1 not in current_answer and year in lib.test_years:
            continue
        
        if year in lib.test_years:
            result["test_query"].append( q )

            if prod:
                result["query"].append( q )
        else:
            result["query"].append( q )
        
        for r in range( 0, len( current_data ) ):
            answer_rank = current_answer[r]

            if answer_rank == 1:
                answer_rank = 10
            elif answer_rank == 2:
                answer_rank = 5
            elif answer_rank == 3:
                answer_rank = 3
            else:
                answer_rank = 0

            if not answer_rank == 0:
                answer_rank += current_level[0]
                answer_rank += int( current_diff[r] )

            if year in lib.test_years:
                result["test_teacher"].append( current_data[r] )
                result["test_answer"].append( float( answer_rank ) )
            else:
                result["teacher"].append( current_data[r] )
                result["answer"].append( float( answer_rank ) )

    return result

def main( data, simu_data ):
    global use_data
    global use_simu_data
    use_simu_data = simu_data
    use_data = data_check( data )

    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print( study.best_params )
    best_model_create( study.best_params )
