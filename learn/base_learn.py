import os
import json
import numpy as np
import lightgbm as lgb

import SekitobaLibrary as lib
import SekitobaDataManage as dm
from learn import data_adjustment

def lg_main( data, index = None ):
    params = {}
    
    if os.path.isfile( "best_params.json" ) and not index == None:
        f = open( "best_params.json", "r" )
        params = json.load( f )[index]
        f.close()
    else:
        params["learning_rate"] = 0.03
        params["num_iteration"] = 2000
        params["max_depth"] = 200
        params["num_leaves"] = 175
        params["min_data_in_leaf"] = 25
        params["lambda_l1"] = 0
        params["lambda_l2"] = 0

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
        #'max_position': int( max_pos ),  # for lambdarank
        'early_stopping_rounds': 30,
        'learning_rate': params["learning_rate"],
        'num_iteration': params["num_iteration"],
        'min_data_in_bin': 1,
        'max_depth': params["max_depth"],
        'num_leaves': params["num_leaves"],
        'min_data_in_leaf': params["min_data_in_leaf"],
        'lambda_l1': params["lambda_l1"],
        'lambda_l2': params["lambda_l2"]
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,
                     valid_sets = [lgb_train, lgb_vaild ],
                     #verbose_eval = 10,
                     num_boost_round = 5000 )
        
    return bst

def importance_check( model ):
    result = []
    importance_data = model.feature_importance()
    f = open( "common/rank_score_data.txt" )
    all_data = f.readlines()
    f.close()
    c = 0

    for i in range( 0, len( all_data ) ):
        str_data = all_data[i].replace( "\n", "" )

        if "False" in str_data:
            continue

        result.append( { "key": str_data, "score": importance_data[c] } )
        c += 1

    f = open( "importance_value.txt", "w" )
    result = sorted( result, key = lambda x: x["score"], reverse= True )

    for i in range( 0, len( result ) ):
        f.write( "{}: {}\n".format( result[i]["key"], result[i]["score"] ) )

    f.close()

def main( data, state = "test" ):
    model_list = []
    learn_data = data_adjustment.data_check( data, state = state )

    for i in range( 0, 5 ):
        model = lg_main( learn_data, index = i )
        importance_check( model )
        model_list.append( model )

    dm.pickle_upload( lib.name.model_name(), model_list )
    return model_list
