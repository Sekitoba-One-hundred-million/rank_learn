import math
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm

S = 250

def softmax( data ):
    result = []
    sum_data = 0
    value_max = max( data )

    for i in range( 0, len( data ) ):
        sum_data += math.exp( data[i] - value_max )

    for i in range( 0, len( data ) ):
        result.append( math.exp( data[i] - value_max ) / sum_data )

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

def simulation( model, current_data ):
    result = []
    
    for horce_num in current_data.keys():
        instance = {}
        #before_change = current_data[horce_num]["change"]        
        t_data = current_data[horce_num]["data"]# + before_change
        predict_diff = model.forward( torch.tensor( np.array( [ t_data ], dtype = np.float32 ) ) ).detach().numpy()
        diff = probability( softmax( predict_diff[0] ) ) / 10
        instance["diff"] = diff
        instance["rank"] = current_data[horce_num]["answer"]["rank"]
        instance["odds"] = current_data[horce_num]["answer"]["odds"]
        instance["horce_num"] = horce_num
        result.append( instance )
        
    return result

def main( model, data ):
    bet_count = 0
    win_count = 0
    recovery_rate = 0
    money = 50000
    money_list = []
    x_data = range( 0, len( data ) )
    count = 0
    test_result = { "count": 0, "money": 0, "win": 0 }    
    
    for race_id in tqdm( data.keys() ):
        ex_value = {}
        
        for horce_id in data[race_id].keys():
            ex_value[horce_id] = {}
            ex_value[horce_id]["rank"] = data[race_id][horce_id]["answer"]["rank"]
            ex_value[horce_id]["odds"] = data[race_id][horce_id]["answer"]["odds"]
            ex_value[horce_id]["rate"] = 0
            ex_value[horce_id]["ex"] = 0
        
        for i in range( 0, S ):
            simu_result = simulation( model, data[race_id] )
            sort_result = sorted( simu_result, key=lambda x:x["diff"] )
            ex_value[sort_result[0]["horce_num"]]["ex"] += sort_result[0]["odds"]
            ex_value[sort_result[0]["horce_num"]]["rate"] += 1

        max_rate = 0
        bet_horce = {}
        
        for horce_id in ex_value.keys():
            ex_value[horce_id]["ex"] /= S
            ex_value[horce_id]["rate"] /= S
            bet_rate = ( ex_value[horce_id]["rate"] * ( ex_value[horce_id]["odds"] + 1 ) - 1 ) / ex_value[horce_id]["odds"]
            bet_money = int( ( money / 5 ) * bet_rate / 100 ) * 100

            if max_rate < ex_value[horce_id]["rate"]:
                max_rate = ex_value[horce_id]["rate"]
                bet_horce = ex_value[horce_id]
            
            if 0 < bet_money:
                bet_count += 1
                money -= bet_money
                
                if ex_value[horce_id]["rank"] == 1:
                    money += ex_value[horce_id]["odds"] * bet_money
                    win_count += 1

        if bet_horce["rank"] == 1:
            recovery_rate += bet_horce["odds"]
            test_result["win"] += 1
            test_result["money"] += bet_horce["odds"]
            lib.log.write( "rate:" + str( bet_horce["rate"] ) + " odds:" + str( bet_horce["odds"] ) )

        money_list.append( money )
        test_result["count"] += 1
        #print( money )

    recovery_rate = test_result["money"] / test_result["count"]
    recovery_rate *= 100
    win_rate = test_result["win"] / test_result["count"]
    win_rate *= 100
    
    lib.log.write( "福利で賭けた回数{}回".format( bet_count ) )
    lib.log.write( "福利で当たった回数{}回".format( win_count ) )
    lib.log.write( "福利での最終金額{}円".format( money ) )    
    lib.log.write( "回収率{}%".format( recovery_rate ) )
    lib.log.write( "勝率{}%".format( win_rate ) )
