import math
from math import factorial
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

import sekitoba_library as lib
import sekitoba_data_manage as dm
from simulation.problem import MoneyProblem

dm.dl.file_set( "test_wide_odds.pickle" )
dm.dl.file_set( "test_baren_odds.pickle" )

class MoneyDistribution:
    def __init__( self ):
        self.race_id = ""
        self.have_money = 100
        self.test_wide_odds = dm.dl.data_get( "test_wide_odds.pickle" )
        self.test_baren_odds = dm.dl.data_get( "test_baren_odds.pickle" )
        self.test_one_odds = {}
        self.curve_slope = 400
        self.result_rank = [ 0, 0, 0 ]

    def set_race_id( self, race_id ):
        self.race_id = race_id

    def set_one_odds( self, one_odds ):
        self.test_one_odds = one_odds

    def set_result_rank( self, result_rank ):
        self.result_rank = result_rank

    def linear_regression( self, x, y ):
        n = len( x )
        t_xy = sum(x*y)-(1/n)*sum(x)*sum(y)
        t_xx = sum(x**2)-(1/n)*sum(x)**2
        slope = t_xy/t_xx
        intercept = (1/n)*sum(y)-(1/n)*slope*sum(x)

        return slope, intercept
        
    def possible_rate_create( self, predict_data ):
        possible_rank_rate_list = []
        sum_score = 0
        possible_rank_list = itertools.permutations( list( range( 0, len( predict_data ) ) ), 3 )

        for possible_rank in possible_rank_list:
            rate = 1
            n1 = int( possible_rank[0] )
            n2 = int( possible_rank[1] )
            n3 = int( possible_rank[2] )
            score = math.exp( predict_data[n1]["rank"] + predict_data[n2]["rank"] + predict_data[n3]["rank"] )

            horce_num1 = predict_data[n1]["horce_num"]
            horce_num2 = predict_data[n2]["horce_num"]
            horce_num3 = predict_data[n3]["horce_num"]
            sum_score += score
            possible_rank_rate_list.append( { "score": score, "rank": ( horce_num1, horce_num2, horce_num3 ) } )

        for i in range( 0, len( possible_rank_rate_list ) ):
            possible_rank_rate_list[i]["score"] /= sum_score
                        
        return possible_rank_rate_list

    def bet_select( self, use_horce_list ):
        sum_value = 0
        bet = { "one": {}, "baren": {}, "wide": {} }

        for i in range( 0, len( use_horce_list ) ):
            key_horce_num = str( int( use_horce_list[i] ) )
            bet["one"][key_horce_num] = random.random()
            sum_value += bet["one"][key_horce_num]
            
        for i in range( 0, len( use_horce_list ) ):
            for r in range( i + 1, len( use_horce_list ) ):
                min_key_horce_num = str( int( min( use_horce_list[i], use_horce_list[r] ) ) )
                max_key_horce_num = str( int( max( use_horce_list[i], use_horce_list[r] ) ) )
                key = min_key_horce_num + "-" + max_key_horce_num
                bet["baren"][key] = random.random()
                bet["wide"][key] = random.random()
                sum_value += bet["baren"][key] + bet["wide"][key]

        result_bet = []
        result_bet_key = { }
        for k in bet.keys():
            for kk in bet[k].keys():
                bet[k][kk] = ( bet[k][kk] / sum_value )
                result_bet.append( bet[k][kk] )
                lib.dic_append( result_bet_key, k, {} )
                lib.dic_append( result_bet_key[k], kk, int( len( result_bet ) - 1 ) )

        return result_bet, result_bet_key

    def one( self, rank_list, use_horce_num_data ):
        odds = 0
        key = ""
        
        if rank_list[0] in use_horce_num_data:
            key = str( int( rank_list[0] ) )
            odds = self.test_one_odds[key]

        return odds, key

    def baren( self, rank_list, use_horce_num_data ):
        odds = 0
        key = ""
        
        if rank_list[0] in use_horce_num_data and rank_list[1] in use_horce_num_data:
            min_key_horce_num = str( int( min( rank_list[0], rank_list[1] ) ) )
            max_key_horce_num = str( int( max( rank_list[0], rank_list[1] ) ) )
            key = min_key_horce_num + "-" + max_key_horce_num
            odds = self.test_baren_odds[self.race_id][min_key_horce_num][max_key_horce_num]

        return odds, key

    def wide( self, rank_list, use_horce_num_data ):
        key_list = []
        odds_list = []

        def check( num1, num2 ):
            key = ""
            odds = 0
            
            if rank_list[num1] in use_horce_num_data and rank_list[num2] in use_horce_num_data:
                min_key_horce_num = str( int( min( rank_list[num1], rank_list[num2] ) ) )
                max_key_horce_num = str( int( max( rank_list[num1], rank_list[num2] ) ) )
                odds = self.test_wide_odds[self.race_id][min_key_horce_num][max_key_horce_num]["min"]
                key = min_key_horce_num + "-" + max_key_horce_num

            return odds, key

        for num1 in range( 0, 3 ):
            for num2 in range( num1 + 1, 3 ):
                odds, key = check( num1, num2 )

                if not odds == 0:
                    key_list.append( key )
                    odds_list.append( odds )
                    
        return odds_list, key_list

    def buy_result( self, bet_count_data, bet_count_key, use_horce_num_data ):
        move_money = 0
        one_odds, one_key = self.one( self.result_rank, use_horce_num_data )
        wide_odds, wide_key = self.wide( self.result_rank, use_horce_num_data )
        baren_odds, baren_key = self.baren( self.result_rank, use_horce_num_data )
        
        if not one_odds == 0:
            one_bet_count = bet_count_data[bet_count_key["one"][one_key]]
            move_money += one_odds * one_bet_count
            
        if not baren_odds == 0:
            baren_bet_count = bet_count_data[bet_count_key["baren"][baren_key]]
            move_money += baren_odds * baren_bet_count
                
        if not len( wide_odds ) == 0:
            for i in range( 0, len( wide_odds ) ):
                wide_bet_count = bet_count_data[bet_count_key["wide"][wide_key[i]]]
                move_money += wide_odds[i] * wide_bet_count

        move_money -= sum( bet_count_data )
        return move_money

    def softmax( self, data ):
        sum_value = sum( data )

        for i in range( 0, len( data ) ):
            data[i] = int( ( data[i] / sum_value ) * self.have_money )

        return data

    def combination( self, n, r ):
        return factorial(n) / factorial(r) / factorial(n - r)

    def main( self, predict_data ):
        possible_rank_rate_list = self.possible_rate_create( predict_data )
        use_horce_num_data = []
        
        for i in range( 0, len( predict_data ) ):
            #if predict_data[i]["rank"] < 0:
            #    continue

            #print( predict_data[i] )
            use_horce_num_data.append( predict_data[i]["horce_num"] )

        N = int( self.combination( len( use_horce_num_data ), 2 ) * 2 + len( use_horce_num_data ) )
        bet_count_data, bet_count_key = self.bet_select( use_horce_num_data )
        problem = MoneyProblem( N )
        problem.set_data( possible_rank_rate_list = possible_rank_rate_list,
                    use_horce_num_data = use_horce_num_data,
                    bet_count_key = bet_count_key,
                    test_wide_odds = self.test_wide_odds,
                    test_baren_odds = self.test_baren_odds,
                    test_one_odds = self.test_one_odds,
                    race_id = self.race_id )
        algorithm = NSGA2(pop_size=100)

        res = minimize(problem,
                algorithm,
                ('n_gen', 100),
                seed=1,
                verbose=True)
        
        min_risk = -1
        target_count = None
        max_money = -10000
        check_point_x = []
        check_point_y = []
        x_data = []
        y_data = []
        
        for i in range( 0, len( res.F ) ):
            check_money = self.buy_result( self.softmax( res.X[i] ), bet_count_key, use_horce_num_data )
            rate = res.F[i][0] * -1
            predict_money = res.F[i][1] * -1

            if 0 < check_money:
                check_point_x.append( rate )
                check_point_y.append( predict_money )
                
            x_data.append( rate )
            y_data.append( predict_money )
            
            if max_money < check_money:
                max_money = check_money

            if min_risk < rate * predict_money:
                min_risk = rate
                target_count = res.X[i]
                
        sum_count = sum( target_count )
        
        for i in range( 0, len( target_count ) ):
            target_count[i] = int( ( target_count[i] / sum_count ) * self.have_money )

        move_money = self.buy_result( target_count, bet_count_key, use_horce_num_data )
        #check_money = self.buy_result( [4] * len( target_count ), bet_count_key, use_horce_num_data )
        print( target_count )
        print( move_money, max_money )
        #plot = Scatter()
        #plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
        plt.scatter( x_data, y_data, color="red")

        if 0 < len( check_point_x ):
            plt.scatter( check_point_x, check_point_y, color="blue")
            
        plt.savefig( "/Users/kansei/Desktop/aaa/" + self.race_id + ".png" )
        plt.close()
        return move_money, use_horce_num_data
