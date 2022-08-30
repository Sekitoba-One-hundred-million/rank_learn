import math
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm

S = 250

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
    value_max = max( data )

    for i in range( 0, len( data ) ):
        sum_data += math.exp( data[i] - value_max )

    for i in range( 0, len( data ) ):
        result.append( math.exp( data[i] - value_max ) / sum_data )

    return result

def main( model, data, t ):   
    recovery_rate = 0
    test = {}
    test_result = { "count": 0, "three_count": 0, "money": 0, "three_money": 0, "win": 0, "three": 0 }
    money = 50000
    money_list = []
    ave_score = 0
    win_score = 0
    lose_score = 0
    t1 = 0
    t2 = 0
    t3 = 0    

    odds_data = dm.pickle_load( "odds_data.pickle" )
    
    for race_id in tqdm( data.keys() ):
        year = race_id[0:4]

        if not year in lib.test_years:
            continue
        
        horce_list = []
        score_list = []
        current_odds = odds_data[race_id]
        
        for horce_id in data[race_id].keys():
            p_data = model.predict( np.array( [ data[race_id][horce_id]["data"] ] ) )
            ex_value = {}
            score = p_data[0]
            ex_value["score"] = score
            ex_value["rank"] = data[race_id][horce_id]["answer"]["rank"]
            ex_value["odds"] = data[race_id][horce_id]["answer"]["odds"]
            ex_value["popular"] = data[race_id][horce_id]["answer"]["popular"]  
            score_list.append( score )
            horce_list.append( ex_value )

        softmax_score_list = softmax( score_list )
        score_list = standardization( score_list )
        
        for i in range( 0, len( score_list ) ):
            horce_list[i]["score"] = score_list[i]

        softmax_score_list = sorted( softmax_score_list, reverse = True )
        sort_result = sorted( horce_list, key=lambda x:x["score"], reverse = True )

        for i in range( 0, t ):
            bet_horce = sort_result[i]
            test_result["count"] += 1
        
            if bet_horce["rank"] == 1:
                recovery_rate += bet_horce["odds"]
                test_result["win"] += 1
                test_result["money"] += bet_horce["odds"]
    
    recovery_rate = test_result["money"] / test_result["count"]
    recovery_rate *= 100
    win_rate = test_result["win"] / test_result["count"]
    win_rate *= 100 * t
    #print( money )
    #three_rate = test_result["three"] / test_result["three_count"]
    #three_rate *= 100
    #three_recovery_rate = test_result["three_money"] / test_result["three_count"]
    print( "" )
    print( "選択数:{}".format( t ) )
    print( "回収率{}%".format( recovery_rate ) )
    print( "勝率{}%".format( win_rate ) )
    #print( "副勝率{}%".format( three_rate ) )
    #print( "複勝回収率{}%".format( three_recovery_rate ) )
    print( "賭けた回数{}回".format( test_result["count"] ) )
    lib.log.write( "回収率{}%".format( recovery_rate ) )
    lib.log.write( "勝率{}%".format( win_rate ) )
    lib.log.write( "賭けた回数{}回".format( test_result["count"] ) )

    return recovery_rate, win_rate
