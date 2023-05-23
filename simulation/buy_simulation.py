import math
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm

def standardization( data ):
    result = []
    ave = sum( data ) / len( data )
    conv = 0

    for d in data:
        conv = math.pow( d - ave, 2 )

    conv /= len( data )
    conv = math.sqrt( conv )

    for d in data:
        result.append( ( d - ave ) / conv )

    return result

def softmax( data ):
    result = []
    sum_data = 0

    for i in range( 0, len( data ) ):
        sum_data += math.exp( data[i] )

    for i in range( 0, len( data ) ):
        result.append( math.exp( data[i] ) / sum_data )

    return result

def score_add( score_data ):
    score_keys = list( score_data.keys() )
    result = [ 0 ] * len( score_data[score_keys[0]] )
    rate_data = { "rank": 1, "one": 0, "two": 0, "three": 1 }
    
    for k in score_keys:
        score_data[k] = softmax( score_data[k] )

    for k in score_keys:
        for i in range( 0, len( score_data[k] ) ):
            result[i] += score_data[k][i] * rate_data[k]

    return result
    #return softmax( result )

def main( models, data ):
    recovery_rate = 0
    test = {}
    test_result = { "count": 0, "bet_count": 0, "one_money": 0, "three_money": 0, "one_win": 0, "three_win": 0, "three_money": 0 }
    money = 50000
    money_list = []
    ave_score = 0
    win_score = 0
    lose_score = 0
    mdcd_score = 0
    mdcd_count = 0
    rate_data = {}
    t = 1

    odds_data = dm.pickle_load( "odds_data.pickle" )
    high_popular_rank_data = dm.pickle_load( "high_popular_rank_data.pickle" )
    #users_score_data = dm.pickle_load( "users_score_data.pickle")
    
    for race_id in tqdm( data.keys() ):
        year = race_id[0:4]
        number = race_id[-2:]
        #if not year in lib.test_years or int( race_place_num ) == 8:
        if not year in lib.test_years:
            continue
        
        horce_list = []
        score_data = {}
        current_odds = odds_data[race_id]

        #if not race_id in users_score_data:
        #    continue
        
        for horce_id in data[race_id].keys():
            scores = {}
            ex_value = {}
            
            for model_key in models.keys():
                p_data = models[model_key].predict( np.array( [ data[race_id][horce_id]["data"] ] ) )
                scores[model_key] = p_data[0]
                lib.dic_append( score_data, model_key, [] )
                score_data[model_key].append( p_data[0] )
                
            ex_value["score"] = -1
            ex_value["rank"] = data[race_id][horce_id]["answer"]["rank"]
            ex_value["odds"] = data[race_id][horce_id]["answer"]["odds"]
            ex_value["popular"] = data[race_id][horce_id]["answer"]["popular"]
            ex_value["horce_id"] = horce_id
            horce_list.append( ex_value )

        if len( horce_list ) < 3:
            continue

        #if not ( 8 <= len( horce_list ) and len( horce_list ) <= 10 ):
        #    continue
        #if len( current_odds["複勝"] ) == 3:
        #    continue
        
        score_list = score_add( score_data )
        
        for i in range( 0, len( score_list ) ):
            horce_list[i]["score"] = score_list[i]

        #softmax_score_list = sorted( softmax_score_list, reverse = True )
        sort_result = sorted( horce_list, key=lambda x:x["score"], reverse = True )

        for i in range( 0, len( sort_result ) ):
            rank = sort_result[i]["rank"]
            score = sort_result[i]["score"]
            key_score = int( min( score * 100, 40 ) )
            mdcd_score += math.pow( rank - ( i + 1 ), 2 )
            mdcd_count += 1
            lib.dic_append( rate_data, key_score, { "data": 0, "count": 0 } )
            rate_data[key_score]["count"] += 1

            if rank == 1:
                rate_data[key_score]["data"] += sort_result[i]["odds"]
        
        for i in range( 0, min( len( sort_result ), t ) ):
            bc = 1#bet_count[i]
            bet_horce = sort_result[i]
            odds = bet_horce["odds"]
            horce_id = bet_horce["horce_id"]
            rank = bet_horce["rank"]
            score = bet_horce["score"]
            popular = bet_horce["popular"]

            #if popular > 3:
            #    continue

            #if score < 0.4:
            #    continue
            
            #high_popular_score = high_popular_rank_data[race_id][horce_id]

            #if high_popular_score > 2.8:
            #    continue

            test_result["bet_count"] += bc
            test_result["count"] += 1
            
            if rank == 1:
                test_result["one_win"] += 1
                test_result["one_money"] += odds * bc

            if rank <= min( 3, len( current_odds["複勝"] ) ):
                rank_index = int( bet_horce["rank"] - 1 )
                three_odds = current_odds["複勝"][rank_index] / 100
                test_result["three_win"] += 1
                test_result["three_money"] += three_odds * bc
    
    one_recovery_rate = ( test_result["one_money"] / test_result["bet_count"] ) * 100 
    three_recovery_rate = ( test_result["three_money"] / test_result["bet_count"] ) * 100
    one_win_rate = ( test_result["one_win"] / test_result["count"] ) * 100 * t
    three_win_rate = ( test_result["three_win"] / test_result["count"] ) * 100 * t
    #print( money )
    #three_rate = test_result["three"] / test_result["three_count"]
    #three_rate *= 100
    #three_recovery_rate = test_result["three_money"] / test_result["three_count"]
    print( "" )
    print( "選択数:{}".format( t ) )
    print( "単勝 回収率{}%".format( one_recovery_rate ) )
    print( "複勝 回収率{}%".format( three_recovery_rate ) )
    print( "単勝 勝率{}%".format( one_win_rate ) )
    print( "複勝 勝率{}%".format( three_win_rate ) )

    #print( "副勝率{}%".format( three_rate ) )
    #print( "複勝回収率{}%".format( three_recovery_rate ) )
    print( "賭けた回数{}回".format( test_result["count"] ) )
    print( "mdcd:{}".format( round( mdcd_score / mdcd_count, 4 ) ) )

    rate_key_list = sorted( list( rate_data.keys() ) )

    #for rate_key in rate_key_list:
    #    rate_data[rate_key]["data"] /= rate_data[rate_key]["count"]
    #    print( "{}: {}".format( rate_key, rate_data[rate_key] ) )
