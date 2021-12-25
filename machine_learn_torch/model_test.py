import math
import copy
import torch
import numpy as np

import sekitoba_library as lib
import sekitoba_data_manage as dm


def softmax( data ):
    result = []
    sum_data = 0
    value_max = max( data )

    for i in range( 0, len( data ) ):
        sum_data += math.exp( data[i] - value_max )

    for i in range( 0, len( data ) ):
        result.append( math.exp( data[i] - value_max ) / sum_data )

    return result

def main( model, data ):
    rank_model = dm.pickle_load( "rank_model.pickle" )
    test_result = { "count": 0, "money": 0, "win": 0 }
    
    for race_id in data.keys():
        horce_list = []
        score_list = []
        
        for horce_id in data[race_id].keys():
            current_data = copy.copy( data[race_id][horce_id]["data"] )
            rank_score = rank_model.predict( np.array( [ current_data ] ) )[0]
            current_data.append( rank_score )
            #score = model.forward( np.array( [ current_data ], dtype = np.float32 ) )[1]
            score = model.forward( torch.tensor( np.array( [ current_data ], dtype = np.float32 ) ) ).detach().numpy()[0][1]
            ex_value = {}
            ex_value["score"] = score
            ex_value["rank"] = data[race_id][horce_id]["answer"]["rank"]
            ex_value["odds"] = data[race_id][horce_id]["answer"]["odds"]
            score_list.append( score )
            horce_list.append( ex_value )
            
        sort_result = sorted( horce_list, key=lambda x:x["score"], reverse = True )
        score_list = softmax( score_list )
        bet_horce = sort_result[0]
        test_result["count"] += 1

        if bet_horce["rank"] == 1:
            #money += bet_money * bet_horce["odds"]
            #recovery_rate += bet_horce["odds"]
            test_result["win"] += 1
            test_result["money"] += bet_horce["odds"]
            lib.log.write( "odds:" + str( bet_horce["odds"] ) + " score:" + str( max( score_list ) ) )

    recovery_rate = test_result["money"] / test_result["count"]
    recovery_rate *= 100
    win_rate = test_result["win"] / test_result["count"]
    win_rate *= 100
    
    lib.log.write( "回収率{}%".format( recovery_rate ) )
    lib.log.write( "勝率{}%".format( win_rate ) )
    lib.log.write( "賭けた回数{}回".format( test_result["count"] ) )    
