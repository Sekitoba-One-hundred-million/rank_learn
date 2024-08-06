import math
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm
from simulation.select_horce import SelectHorce

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

def main( model_list, data, test_years = lib.test_years, show = True ):
    test_result = { "count": 0, "bet_count": 0, "one_money": 0, "three_money": 0, "one_win": 0, "three_win": 0, "three_money": 0 }
    money = 3000
    money_list = []
    test_check = {}
    odds_data = dm.pickle_load( "odds_data.pickle" )
    wide_odds_data = dm.pickle_load( "wide_odds_data.pickle" )
    race_id_list = list( data.keys() )
    random.shuffle( race_id_list )
    
    for race_id in tqdm( race_id_list ):
        year = race_id[0:4]
        number = race_id[-2:]

        if not year in test_years:
            continue

        horce_list = []
        score_list = []
        instance_list = []
        
        for horce_id in data[race_id].keys():
            scores = {}
            ex_value = {}
            p_score = 0

            for model in model_list:
                p_score += model.predict( np.array( [ data[race_id][horce_id]["data"] ] ) )[0]

            p_score /= len( model_list )
            score_list.append( p_score )
            ex_value["score"] = p_score
            ex_value["rank"] = data[race_id][horce_id]["answer"]["rank"]
            ex_value["odds"] = data[race_id][horce_id]["answer"]["odds"]
            ex_value["popular"] = data[race_id][horce_id]["answer"]["popular"]
            ex_value["horce_num"] = data[race_id][horce_id]["answer"]["horce_num"]
            ex_value["horce_id"] = horce_id
            horce_list.append( ex_value )

        if len( horce_list ) < 5:
            continue

        all_score = 0
        min_score = min( score_list )
        
        #for i in range( 0, len( score_list ) ):
        #    score_list[i] -= min_score

        score_list = softmax( score_list )

        for i in range( 0, len( score_list ) ):
            horce_list[i]["score"] = score_list[i]
        
        sort_result = sorted( horce_list, key=lambda x:x["score"], reverse = True )
        
        select_horce = SelectHorce( wide_odds_data[race_id], sort_result )
        #select_horce.create_bet_rate( money )
        select_horce_data, select_score_list, wide_rate = select_horce.select_horce()
        #print( select_score_list, sum( select_score_list ) )

        if select_horce.bet_rate <= 0:
            break

        #if sum( select_score_list ) < 0.06:
        #    continue
        
        ex_value = int( sum( select_score_list ) * 40 )
        #if ex_value < 16:
        #    continue

        #select_horce.create_bet_rate( money, ex_value )

        #if select_horce.bet_rate <= 0:
        #    break
        #ex_value = int( ( wide_rate * 100 ) / 5 )

        #if ex_value < 11:
        #    continue
        
        lib.dic_append( test_check, ex_value, { "money": 0, "count": 0 } )
        get_money = select_horce.bet_check( select_horce_data, odds_data[race_id] )
        test_result["count"] += 1
        test_result["bet_count"] += select_horce.use_count * select_horce.bet_rate
        test_result["one_money"] += get_money * select_horce.bet_rate
        money -= select_horce.use_count * select_horce.bet_rate
        money += get_money * select_horce.bet_rate
        test_check[ex_value]["count"] += select_horce.use_count
        test_check[ex_value]["money"] += get_money

        if not get_money == 0:
            test_result["one_win"] += 1            

        money_list.append( money )

    one_recovery_rate = ( test_result["one_money"] / test_result["bet_count"] ) * 100 
    one_win_rate = ( test_result["one_win"] / test_result["count"] ) * 100

    sort_data = sorted( list( test_check.keys() ) )
    
    for key in sort_data:
        print( key, test_check[key]["money"] / test_check[key]["count"], int( test_check[key]["count"] / 10 ) )

    if show:
        print( "" )
        print( "回収率{}%".format( one_recovery_rate ) )
        print( "勝率{}%".format( one_win_rate ) )
        print( "賭けたレース数{}回".format( test_result["count"] ) )
        print( "賭けた金額{}".format( test_result["bet_count"] ) )
        print( "金額:{}".format( money ) )
        print( "最低金額:{}".format( min( money_list ) ) )
        plt.plot( list( range( 0, len( money_list ) ) ), money_list )
        plt.savefig( '/Volumes/Gilgamesh/sekitoba-data/money.png' )
