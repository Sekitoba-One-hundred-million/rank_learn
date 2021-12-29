import os
import math
import random
import pickle
import numpy as np
import sys
import lightgbm as lgb
import optuna.integration.lightgbm as test_lgb
from tqdm import tqdm
import matplotlib.pyplot as plt
from chainer import serializers
import xgboost as xgb
import pandas as pd

import sekitoba_library as lib
import sekitoba_data_manage as dm
from rank_learn import rank_simulation
from rank_learn import rank_multi_simulation

def lg_main( data ):
    max_pos = np.max( np.array( data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ), group = np.array( data["query"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ), group = np.array( data["test_query"] ) )
    lgbm_params =  {
        #'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',   # for lambdarank
        'ndcg_eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( data["answer"], dtype = np.int32 ) ) + 1)),
        'max_position': int( max_pos ),  # for lambdarank
        'early_stopping_rounds': 30,
        'learning_rate': 0.05,
        'num_iteration': 300,
        'min_data_in_bin': 1,
        'max_depth': 200,
        'num_leaves': 175,
        'min_data_in_leaf': 25,
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,     
                     valid_sets = [lgb_train, lgb_vaild ],
                     verbose_eval = 10,
                     num_boost_round = 5000 )
    
    #print( df_importance.head( len( x_list ) ) )
    dm.pickle_upload( "rank_model.pickle", bst )
    #lib.log.write_lightbgm( bst )
    
    return bst


def lgb_test( data ):
    max_pos = np.max( np.array( data["answer"] ) )
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), np.array( data["answer"] ), group = np.array( data["query"] ) )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), np.array( data["test_answer"] ), group = np.array( data["test_query"] ) )

    lgbm_params =  {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': 'ndcg',   # for lambdarank
        'ndcg_eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( data["answer"], dtype = np.int32 ) ) + 1)),
        'max_position': int( max_pos ),  # for lambdarank
        'early_stopping_rounds': 30,        
    }

    bst = test_lgb.train( params = lgbm_params,
                          train_set = lgb_train,
                          valid_sets = [lgb_train, lgb_vaild ],
                          verbose_eval = 10,
                          num_boost_round = 5000 )

    print( bst.params )
    lib.log.write( "best_iteration:{}".format( str( bst.best_iteration ) ) )
    lib.log.write( "best_score:{}".format( str( bst.best_score ) ) )
    lib.log.write( "best_params:{}".format( str( bst.params ) ) )
    
    return bst.params

def data_check( data, min_rank ):
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
        
        if year == lib.test_year:
            result["test_query"].append( q )
        else:
            result["query"].append( q )

        current_data = list( data["teacher"][count:count+q] )
        current_answer = list( data["answer_rank"][count:count+q] )

        for r in range( 0, len( current_data ) ):
            answer_rank = current_answer[r]

            if year == lib.test_year:
                result["test_teacher"].append( current_data[r] )
                result["test_answer"].append( float( answer_rank ) )
            else:
                result["teacher"].append( current_data[r] )
                result["answer"].append( float( answer_rank ) )

        count += q

    return result

def diff_data_check( data, min_rank ):
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
        
        if year == lib.test_year:
            result["test_query"].append( q )
        else:
            result["query"].append( q )

        current_data = list( data["teacher"][count:count+q] )
        current_answer = list( data["answer_diff"][count:count+q] )

        for r in range( 0, len( current_data ) ):            
            answer_rank = max( ( current_answer[r] + 2 ) * 10, 0 )
            answer_rank = int( answer_rank )

            if year == lib.test_year:
                result["test_teacher"].append( current_data[r] )
                result["test_answer"].append( float( answer_rank ) )
            else:
                result["teacher"].append( current_data[r] )
                result["answer"].append( float( answer_rank ) )

        count += q

    return result

def main( data, simu_data, simulation = True, learn_data = False ):
    print( "rank_learn" )
    model_list = []
    """
    for i in range( 2, 19 ):
        print( i )
        learn_data = data_check( data, i )
        rank_model = lg_main( learn_data )
        recovery_rate = None
        model_list.append( rank_model )

    rank_multi_simulation.main( model_list, simu_data )
    """
    learn_data = data_check( data, 6 )
    #learn_data = diff_data_check( data, 6 )
    #params = lgb_test( learn_data )
    rank_model = lg_main( learn_data )
    
    if simulation:
        recovery_rate, win_rate = rank_simulation.main( rank_model, simu_data )

    return rank_model, recovery_rate
