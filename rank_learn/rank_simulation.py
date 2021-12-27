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

def probability( p_data ):
    number = -1
    count = 0
    r_check = random.random()
    
    for i in range( 0, len( p_data ) ):
        count += p_data[i]
        
        if r_check <= count:
            number = i
            break

    if number == -1:
        number = len( p_data )

    return number

def main( model, data ):
    recovery_rate = 0
    test = {}
    test_result = { "count": 0, "money": 0, "win": 0 }
    money = 50000
    ave_score = 0
    win_score = 0
    lose_score = 0
    
    for race_id in tqdm( data.keys() ):
        horce_list = []
        score_list = []
        
        for horce_id in data[race_id].keys():
            p_data = model.predict( np.array( [ data[race_id][horce_id]["data"] ] ) )
            ex_value = {}
            score = p_data[0] * -1
            ex_value["score"] = score
            ex_value["rank"] = data[race_id][horce_id]["answer"]["rank"]
            ex_value["odds"] = data[race_id][horce_id]["answer"]["odds"]
            score_list.append( score )
            horce_list.append( ex_value )

        score_list = standardization( score_list )

        for i in range( 0, len( score_list ) ):
            horce_list[i]["score"] = score_list[i]
        
        sort_result = sorted( horce_list, key=lambda x:x["score"], reverse = True )
        max_rate = 0

        for i in range( 0, len( sort_result ) ):
            if sort_result[i]["score"] <= 0:
                continue
            
            bet_horce = sort_result[i]
            test_result["count"] += 1
        
            if bet_horce["rank"] == 1:
                recovery_rate += bet_horce["odds"]
                test_result["win"] += 1
                test_result["money"] += bet_horce["odds"]
                lib.log.write( "odds:" + str( bet_horce["odds"] ) + " score:" + str( max( score_list ) ) )

    recovery_rate = test_result["money"] / test_result["count"]
    recovery_rate *= 100
    win_rate = test_result["win"] / test_result["count"]
    win_rate *= 100

    print( "回収率{}%".format( recovery_rate ) )
    print( "勝率{}%".format( win_rate ) )
    print( "賭けた回数{}回".format( test_result["count"] ) )
    lib.log.write( "回収率{}%".format( recovery_rate ) )
    lib.log.write( "勝率{}%".format( win_rate ) )
    lib.log.write( "賭けた回数{}回".format( test_result["count"] ) )

    return recovery_rate, win_rate
