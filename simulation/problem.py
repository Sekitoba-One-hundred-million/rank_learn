import math
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 問題設定（自作関数）
class MoneyProblem(Problem):
    def __init__( self, n ):
        self.possible_rank_rate_list = []
        self.use_horce_num_data = []
        self.bet_count_key = {}
        self.test_wide_odds = {}
        self.test_baren_odds = {}
        self.test_one_odds = {}
        self.race_id = ""
        
        super().__init__( n_var = n,
                         n_obj = 2,
                         n_constr = 0,
                         xl = np.array( [0] * n ),
                         xu = np.array( [1] * n ),
                        )

    def set_data( self, possible_rank_rate_list = None,
                 use_horce_num_data = None,
                 bet_count_key = None,
                 test_wide_odds = None,
                 test_baren_odds = None,
                 test_one_odds = None,
                 race_id = None ):
        self.possible_rank_rate_list = possible_rank_rate_list
        self.use_horce_num_data = use_horce_num_data
        self.bet_count_key = bet_count_key
        self.test_wide_odds = test_wide_odds
        self.test_baren_odds = test_baren_odds
        self.test_one_odds = test_one_odds
        self.race_id = race_id

    def one( self, rank_list ):
        odds = 0
        key = ""
        
        if rank_list[0] in self.use_horce_num_data:
            key = str( int( rank_list[0] ) )
            odds = self.test_one_odds[key]

        return odds, key

    def baren( self, rank_list ):
        odds = 0
        key = ""
        
        if rank_list[0] in self.use_horce_num_data and rank_list[1] in self.use_horce_num_data:
            min_key_horce_num = str( int( min( rank_list[0], rank_list[1] ) ) )
            max_key_horce_num = str( int( max( rank_list[0], rank_list[1] ) ) )
            key = min_key_horce_num + "-" + max_key_horce_num
            odds = self.test_baren_odds[self.race_id][min_key_horce_num][max_key_horce_num]

        return odds, key

    def wide( self, rank_list ):
        key_list = []
        odds_list = []

        def check( num1, num2 ):
            key = ""
            odds = 0
            
            if rank_list[num1] in self.use_horce_num_data and rank_list[num2] in self.use_horce_num_data:
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

    def profit_risk_create( self, bet_count_data ):
        risk = 0
        profit = 0
        sum_bet_count = 0
        risk_check_list = []
        profit_check_list = []
        sum_bet_count = sum( bet_count_data )

        for prt in self.possible_rank_rate_list:
            baren_bet_count = 0
            wide_bet_count = 0
            one_bet_count = 0
            move_money = 0
            get_true = False
            one_odds, one_key = self.one( prt["rank"] )
            baren_odds, baren_key = self.baren( prt["rank"] )
            wide_odds, wide_key = self.wide( prt["rank"] )
            
            if not one_odds == 0:
                one_bet_count = bet_count_data[self.bet_count_key["one"][one_key]]
                move_money += one_odds * one_bet_count
                get_true = True
            
            if not baren_odds == 0:
                baren_bet_count = bet_count_data[self.bet_count_key["baren"][baren_key]]
                move_money += baren_odds * baren_bet_count
                get_true = True

            if not len( wide_odds ) == 0:
                get_true = True
                for i in range( 0, len( wide_odds ) ):
                    wide_bet_count = bet_count_data[self.bet_count_key["wide"][wide_key[i]]]
                    move_money += wide_odds[i] * wide_bet_count

            move_money -= sum_bet_count
            current_score = move_money * prt["score"]

            if 0 < current_score:
                profit_check_list.append( move_money )
                risk_check_list.append( prt["score"] )
                        
        profit_rate = len( profit_check_list ) / ( len( self.possible_rank_rate_list ) )
        profit = ( sum( profit_check_list ) / len( profit_check_list ) ) * profit_rate
        risk = sum( risk_check_list ) * profit_rate

        return profit, risk

    def _evaluate(self, X, out, *args, **kwargs):
        func1 = []
        func2 = []
        
        for i in range( 0, len( X ) ):
            y, x = self.profit_risk_create( X[i] )
            func1.append( x * -1 )
            func2.append( y * -1 )
            
        out["F"] = np.column_stack([func1, func2])
