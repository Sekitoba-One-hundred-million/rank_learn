import os
import math
import random
import pickle
import numpy as np
import sys
import lightgbm as lgb
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm
from learn import simulation
from simulation import test

def lg_main( data ):
    print( len( data["test_teacher"] ), len( data["teacher"] ) )
    print( data["teacher"][0] )
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
    dm.pickle_upload( lib.name.model_name(), bst )
    #lib.log.write_lightbgm( bst )
    
    return bst

def data_check( data ):
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
        
        if year in lib.test_years:
            result["test_query"].append( q )
        else:
            result["query"].append( q )

        current_data = list( data["teacher"][count:count+q] )
        current_answer = list( data["answer"][count:count+q] )

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
                
            if year in lib.test_years:
                result["test_teacher"].append( current_data[r] )
                result["test_answer"].append( float( answer_rank ) )
            else:
                result["teacher"].append( current_data[r] )
                result["answer"].append( float( answer_rank ) )

        count += q

    return result

def main( data, simu_data ):
    result = {}
    learn_data = data_check( data )
    model = lg_main( learn_data )

    recovery_rate, win_rate = simulation.main( model, simu_data, 1 )

    result["model"] = model
    result["recovery"] = recovery_rate
    result["win"] = win_rate
    #result = test.main( model, simu_data )

    return result
