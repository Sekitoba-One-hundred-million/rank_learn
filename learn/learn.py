import numpy as np
import lightgbm as lgb

import sekitoba_library as lib
import sekitoba_data_manage as dm
#from learn import simulation

def lg_main( data, prod = False ):
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
        'learning_rate': 0.03,
        'num_iteration': 2000,
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
    
    if prod:
        dm.pickle_upload( lib.name.model_name() + ".prod", bst )
    else:
        dm.pickle_upload( lib.name.model_name(), bst )
        
    return bst

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

    result = sorted( result, key = lambda x: x["score"], reverse= True )

    for i in range( 0, len( result ) ):
        print( "{}: {}".format( result[i]["key"], result[i]["score"] ) )

def main( data ):
    result = {}
    learn_data = data_check( data )
    model = lg_main( learn_data )
    importance_check( model )
    result["rank"] = model

    #recovery_rate, win_rate = simulation.main( model, simu_data, 1 )
    #simulation.main( model, simu_data, 1 )

    #result["model"] = model
    #result["recovery"] = recovery_rate
    #result["win"] = win_rate

    return result
