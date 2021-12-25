import math
import torch
import random
import numpy as np
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm

S = 250
def softmax( data ):
    sum_data = 0
    value_max = max( data )

    for i in range( 0, len( data ) ):
        sum_data += math.exp( data[i] - value_max )

    for i in range( 0, len( data ) ):
        data[i] = math.exp( data[i] - value_max ) / sum_data

def bet_horce_get( data ):
    max_score = 0
    result = {}
    
    for i in range( 0, len( data ) ):
        if max_score < data[i]["score"]:
            result = data[i]

    return result
        

def main( models, datas ):
    test_result = { "count": 0, "money": 0, "win": 0 }
    money = 50000
    
    for race_id in datas.keys():
        horce_list = []
        score_list = []        

        for horce_id in datas[race_id].keys():                
            instance = {}
            instance["score"] = 0

            for i in range( 0, len( models ) ):                
                score = models[i].predict( np.array( [ datas[race_id][horce_id]["data"] ] ) )[0]
                instance["score"] += score * -1
                
            instance["rank"] = datas[race_id][horce_id]["answer"]["rank"]
            instance["odds"] = datas[race_id][horce_id]["answer"]["odds"]
            score_list.append( instance["score"] )
            horce_list.append( instance )

        sort_result = sorted( horce_list, key=lambda x:x["score"], reverse = True )
        score_list = softmax( score_list )
        bet_horce = sort_result[0]    
        test_result["count"] += 1
        
        if bet_horce["rank"] == 1:
            #money += bet_money * bet_horce["odds"]
            test_result["win"] += 1
            test_result["money"] += bet_horce["odds"]
            #lib.log.write( "odds:" + str( bet_horce["odds"] ) + " score:" + str( max( score_list ) ) )

        """
        if not win:
            lib.log.write( "bet_money:{} money:{}".format( bet_money, money ) )
        else:
            lib.log.write( "bet_money:{} get_money:{} money:{}".format( bet_money, bet_money * bet_horce["odds"], money ) )
        """

    recovery_rate = test_result["money"] / test_result["count"]
    recovery_rate *= 100
    win_rate = test_result["win"] / test_result["count"]
    win_rate *= 100
    
    lib.log.write( "回収率{}%".format( recovery_rate ) )
    lib.log.write( "勝率{}%".format( win_rate ) )
    lib.log.write( "賭けた回数{}回".format( test_result["count"] ) )

    print( "回収率{}%".format( recovery_rate ) )
    print( "勝率{}%".format( win_rate ) )
    print( "賭けた回数{}回".format( test_result["count"] ) )

    return recovery_rate, win_rate
