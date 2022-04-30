import math
import copy
import itertools
from numpy.linalg import solve

import sekitoba_library as lib
import matplotlib.pyplot as plt

def synthetic_odds_calc( odds_list: list ):
    result = 0
    
    for i in range( 0, len( odds_list ) ):
        result += 1 / odds_list[i]
        
    return 1 / result


def rate_param_calc( odds_list: list ):
    result = 0
    min_odds = min( odds_list )
    min_index = odds_list.index( min( odds_list ) )

    for i in range( 0, len( odds_list ) ):
        if not i == min_index:
            result += min_odds / odds_list[i]

    return result + 1

def min_bet_calc( target: int, min_odds: float, rate_param: float ):
    return ( target * 1.0 ) / ( min_odds - rate_param )

def bet_select( odds_list: list, target: float, rate_param: float ):
    result = [0] * len( odds_list )
    min_bet = int( min_bet_calc( target, min( odds_list ), rate_param ) + 0.5 )
    min_odds = min( odds_list )
    min_index = odds_list.index( min( odds_list ) )

    for i in range( 0, len( odds_list ) ):
        if i == min_index:
            result[i] = int( min_bet )
        else:
            result[i] = int( ( min_bet * min_odds ) / odds_list[i] + 0.5 )

        if result[i] == 0:
            result[i] = 1

    return result


def equation_create( odds_data, score_data ):
    result = []
    answer = []

    s_odds = synthetic_odds_calc( odds_data )
    rate = sum( score_data )
    
    for i in range( 0, len( odds_data ) ):
        instance = []
        for r in range( 0, len( odds_data ) ):
            s = odds_data[i] * score_data[r]
            c = score_data[i] * odds_data[r]
            
            if i == r:
                s += 0.0001
                
            instance.append( s + c )

        a = ( score_data[i] * s_odds ) * -1
        result.append( instance )
        answer.append( a )

    return result, answer

def main( data: dict, target: int ):

    #if s_odds < 1.5:
    #    return None
    
    #rate_param = rate_param_calc( data["odds"] )
    #result = bet_select( data["odds"], target, rate_param )
    eq, answer = equation_create( data["odds"], data["rate"] )
    result = solve( eq, answer )

    return result
