import os
import json
import numpy as np
import lightgbm as lgb
import xgboost as xgb

import SekitobaLibrary as lib
import SekitobaDataManage as dm
from learn import data_adjustment
from learn.const import *

def xg_main( data, index = None ):
    qid_data = []
    valid_qid_data = []

    for i, query in enumerate( data["query"] ):
        qid_data.extend( [i+1] * query )

    for i, query in enumerate( data["test_query"] ):
        valid_qid_data.extend( [i+1] * query )

    params = {}
    
    if os.path.isfile( XG_BEST_PARAMS_FILE ) and not index == None:
        f = open( XG_BEST_PARAMS_FILE, "r" )
        params = json.load( f )[index]
        f.close()
    else:
        params["learning_rate"] = 0.05
        params["max_depth"] = 4
        params["min_child_weight"] = 1
        params["subsumple"] = 0.75
        params["colsample_bytree"] = 0.6
        
    model = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:ndcg", # または rank:pairwise
        eval_metric='ndcg@2',
        lambdarank_num_pair_per_sample=200,
        ndcg_exp_relevance=True,
        n_estimators=1000,
        early_stopping_rounds=30,
        gamma=0,
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsumple"],
        colsample_bytree=params["colsample_bytree"]
    )

    model.fit( np.array( data["teacher"] ), \
               np.array( data["answer"] ), \
               qid = np.array( qid_data ), \
               eval_set = [( np.array( data["test_teacher"] ), np.array( data["test_answer"] ) )], \
               eval_qid = [np.array( valid_qid_data )], \
               verbose = 10 )

    return model
    
def lg_main( data, category_index_list, index = None ):
    params = {}
    
    if os.path.isfile( LG_BEST_PARAMS_FILE ) and not index == None:
        f = open( LG_BEST_PARAMS_FILE, "r" )
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
    lgb_train = lgb.Dataset( np.array( data["teacher"] ), \
                             np.array( data["answer"] ), \
                             group = np.array( data["query"] ), \
                             categorical_feature = category_index_list )
    lgb_vaild = lgb.Dataset( np.array( data["test_teacher"] ), \
                             np.array( data["test_answer"] ), \
                             group = np.array( data["test_query"] ), \
                             categorical_feature = category_index_list )
    lgbm_params =  {
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        #'objective': 'rank_xendcg',
        'metric': 'ndcg',   # for lambdarank
        'eval_at': [1,2,3],  # for lambdarank
        'label_gain': list(range(0, np.max( np.array( data["answer"], dtype = np.int32 ) ) + 1)),
        'lambdarank_truncation_level': 2,
        'early_stopping_rounds': 30,
        'lambdarank_norm': True,
        'num_iteration': 3000,
        'learning_rate': params["learning_rate"],
        'min_data_in_bin': 1,
        'max_depth': params["max_depth"],
        'num_leaves': params["num_leaves"],
        'min_data_in_leaf': params["min_data_in_leaf"],
        'lambda_l1': params["lambda_l1"],
        'lambda_l2': params["lambda_l2"],
        'feature_fraction': params["feature_fraction"]
    }

    bst = lgb.train( params = lgbm_params,
                     train_set = lgb_train,
                     valid_sets = [lgb_train, lgb_vaild ],
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
    category_index_list = lib.create_category_index( data["category"] )
    learn_data = data_adjustment.data_check( data, state = state )

    for i in range( 0, LG_MODEL_NUM ):
        lg_model = lg_main( learn_data, category_index_list, index = i )
        model_list.append( lg_model )

    for i in range( 0, XG_MODEL_NUM ):
        xg_model = xg_main( learn_data, i )
        model_list.append( xg_model )

    importance_check( model_list[0] )
    dm.pickle_upload( lib.name.model_name(), model_list )
    return model_list
