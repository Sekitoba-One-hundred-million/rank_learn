import math
import torch
import random
import numpy as np
from tqdm import tqdm
from mpi4py import MPI
import matplotlib.pyplot as plt

import sekitoba_library as lib
import sekitoba_data_manage as dm

from simulation import bet_select_check

n = 3
N = 50

class BuyData:
    def __init__( self ):
        self.win = 0
        self.money = 0
        self.count = 0
        self.race_count = 0        

def softmax( data ):
    result = []
    sum_data = 0
    value_max = max( data )

    for i in range( 0, len( data ) ):
        sum_data += math.exp( data[i] - value_max )

    for i in range( 0, len( data ) ):
        result.append( math.exp( data[i] - value_max ) / sum_data )

    return result

def buy_simu( model, data, race_id, fuku_odds_data, target ):
    buy_data = BuyData()
    
    horce_list = []
    wide_odds = {}
    horce_rate = {}

    for horce_num in data[race_id].keys():
        p_data = model.predict( np.array( [ data[race_id][horce_num]["data"] ] ) )
        ex_value = {}
        score = p_data[0]
        rank = int( data[race_id][horce_num]["answer"]["rank"] )
        odds = data[race_id][horce_num]["answer"]["odds"]        
        fuku_odds = 1#fuku_odds_data[race_id][horce_num]["min"]
        horce_list.append( { "rank": rank, "score": score, "odds": odds, "fuku_odds": fuku_odds } )

    horce_list = sorted( horce_list, key=lambda x:x["score"], reverse = True )
    rate_data = []

    for horce in horce_list:
        rate_data.append( horce["score"] )
            
    rate_data = softmax( rate_data )
    select_data = { "odds": [], "fuku_odds": [], "rate": [] }
    bet_horce = []

    if len( horce_list ) < 5:
        return None, 0, 0
        
    for i in range( 0, n ):
        select_data["odds"].append( horce_list[i]["odds"] )
        select_data["rate"].append( rate_data[i] )
        select_data["fuku_odds"].append( horce_list[i]["fuku_odds"] )
        bet_horce.append( horce_list[i] )

    bet_count = bet_select_check.main( select_data, target )
    
    if len( bet_count ) == 0:
        return None

    bet_count = softmax( bet_count )
    
    for i in range( 0, len( bet_count ) ):
        bet_count[i] = int( bet_count[i] * N )

    buy_data.race_count += 1
    #print( target, bet_count, bet_horce )
    for i in range( 0, n ):
        buy_data.count += bet_count[i]

        if bet_horce[i]["rank"] == 1:
            buy_data.money += bet_horce[i]["odds"] * bet_count[i]
            buy_data.win += 1
    
    return buy_data

def main( model, data ):
    #fuku_odds_data = dm.pickle_load( "fuku_odds_data.pickle" )    
    money = 0
    money_list = []
    min_money = 0
    buy_data = BuyData()
    race_id_list = list( data.keys() )
    random.shuffle( race_id_list )
    target = 5
    loss = 0
    
    for race_id in tqdm( race_id_list ):
        bd = buy_simu( model, data, race_id, None, target )

        if bd == None:
            continue

        money -= bd.count
        min_money = min( min_money, money )
        money += bd.money
        money_list.append( money )
        buy_data.win += bd.win
        buy_data.count += bd.count
        buy_data.money += bd.money
        buy_data.race_count += bd.race_count        
        #print( money, bd.count )
        
        if bd.count < bd.money:
            target = 5
            loss = 0
        else:
            loss += 1
            target += int( ( bd.count - bd.money ) * 100 ) / 100

    money_rate = buy_data.money / buy_data.count
    #print( min_money )
    #print( max_loss )
    print( "賭けたレース数:{}回".format( buy_data.race_count ) )
    print( "賭けた回数:{}回".format( buy_data.count ) )
    print( "回収率:{}%".format( money_rate * 100 ) )
    print( "正答率:{}%".format( ( buy_data.win / buy_data.race_count ) * 100 ) )
    #print( "儲け:{}円".format( int( buy_data.count * ( money_rate - 1 ) * 100 ) ) )

    result = {}
    result["recovery"] = money_rate * 100
    result["win"] = ( buy_data.win / buy_data.race_count ) * 100
    
    return result
    #x_data = list( range( 0, len( money_list ) ) )
    #plt.plot( x_data, money_list )
    #plt.show()
