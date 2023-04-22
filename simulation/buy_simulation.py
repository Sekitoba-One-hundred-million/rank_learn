import numpy as np
import os

import sekitoba_library as lib
import sekitoba_data_manage as dm
from simulation.money_distribution import MoneyDistribution

lib.name.set_name( "rank" )

def arrangement_predict_data( data ):
    softmax_key_list = [ "one", "two", "three" ]
    co_data = { "one": 1, "two": 2, "three": 3 }

    for sk in softmax_key_list:
        sum_value = 0
        
        for i in range( 0, len( data ) ):
            sum_value += data[i][sk]

        for i in range( 0, len( data ) ):
            data[i][sk] = ( data[i][sk] / sum_value ) * co_data[sk]

    for i in range( 0, len( data ) ):
        data[i]["three"] -= data[i]["two"]
        data[i]["two"] -= data[i]["one"]
        data[i]["not"] = 1 - ( data[i]["one"] + data[i]["two"] + data[i]["three"] )

    data = sorted( data, key=lambda x:x["rank"], reverse = True )
    print( data[0] )
    #for d in data:
    #    print( d )
    
    return data

def main():
    try:
        os.remove( "check.txt" )
    except Exception as e:
        print( e )
        
    models = {}
    #models["one"] = dm.pickle_load( "one_" + lib.name.model_name() )
    #models["two"] = dm.pickle_load( "two_" + lib.name.model_name() )
    #models["three"] = dm.pickle_load( "three_" + lib.name.model_name() )   
    models["rank"] = dm.pickle_load( lib.name.model_name() )
    simu_data = dm.pickle_load( lib.name.simu_name() )
    #users_score_data = dm.pickle_load( "users_score_data.pickle" )
    odds_data = dm.pickle_load( "odds_data.pickle" )
    money = 0
    md = MoneyDistribution()

    for race_id in simu_data.keys():
        year = race_id[0:4]

        if not year in lib.test_years:
            continue
        
        predict_data = []
        #current_odds = odds_data[race_id]
        one_odds_data = {}
        result_rank = [ 0, 0, 0 ]
        
        for horce_id in simu_data[race_id].keys():
            rank_predict = models["rank"].predict( np.array( [ simu_data[race_id][horce_id]["data"] ] ) )[0]
            #users_score = 0

            #for k in users_score_data[race_id][horce_id]:
            #    users_score += users_score_data[race_id][horce_id][k]
                
            #rank_predict = users_score * rank_data
            #one_rate_predict = models["one"].predict( np.array( [ simu_data[race_id][horce_id]["data"] ] ) )[0]
            #two_rate__predict = models["two"].predict( np.array( [ simu_data[race_id][horce_id]["data"] ] ) )[0]
            #three_rate_predict = models["three"].predict( np.array( [ simu_data[race_id][horce_id]["data"] ] ) )[0]
            horce_num = simu_data[race_id][horce_id]["answer"]["horce_num"]
            key_horce_num = str( int( horce_num ) )
            one_odds_data[key_horce_num] = simu_data[race_id][horce_id]["answer"]["odds"]
            predict_data.append( { "horce_num": simu_data[race_id][horce_id]["answer"]["horce_num"],
                                  "rank": rank_predict,
                                  #"one": one_rate_predict,
                                  #"two": two_rate__predict,
                                  #"three": three_rate_predict
                                  } )

            rank = int( simu_data[race_id][horce_id]["answer"]["rank"] )
            if rank < 4:
                result_rank[int(rank-1)] = horce_num

        if 0 in result_rank:
            continue
        
        #predict_data = arrangement_predict_data( predict_data )
        predict_data = sorted( predict_data, key=lambda s:s["rank"], reverse=True )
        l = min( len( predict_data ), 5 )
        md.set_race_id( race_id )
        md.set_one_odds( one_odds_data )
        md.set_result_rank( result_rank )
        move_money, use_horce_num_list = md.main( predict_data[0:l] )
        print( move_money )
        money += move_money
        f = open( "check.txt", "a" )
        f.write( "money:{} move_money:{} horce_num:{} race_id:{}\n".format( money, move_money, str( use_horce_num_list ), race_id ) )
        f.close()
