import copy
import math
from tqdm import tqdm

import SekitobaLibrary as lib
import SekitobaDataManage as dm

WIDE = "wide"
ONE = "one"

class SelectHorce:
    def __init__( self, wide_odds_data, horce_data ):
        self.use_count = 10
        self.bet_rate = 1
        self.goal_rate = 2
        self.bet_result_count = 0
        self.wide_odds_data = wide_odds_data
        self.horce_data = horce_data
        self.rate_data = []
        self.select_horce_list = []

    def create_rate( self ):
        all_rate = 0
        
        for i in range( 0, len( self.horce_data ) ):
            score_1 = self.horce_data[i]["score"]
            horce_num_1 = self.horce_data[i]["horce_num"]
            
            for r in range( i + 1, len( self.horce_data ) ):
                score_2 = self.horce_data[r]["score"]
                horce_num_2 = self.horce_data[r]["horce_num"]

                for t in range( r + 1, len( self.horce_data ) ):
                    score_3 = self.horce_data[t]["score"]
                    horce_num_3 = self.horce_data[t]["horce_num"]
                    rate = score_1 * score_2 * score_3

                    self.rate_data.append( { "rate": rate, \
                                            "horce_num_list": [ horce_num_1, horce_num_2, horce_num_3 ], \
                                            "use": False } )
                    all_rate += rate

        for i in range( 0, len( self.rate_data ) ):
            self.rate_data[i]["rate"] /= all_rate

    def one_select_rate( self, horce_num_list ):
        rate = 0

        for i in range( 0, len( self.rate_data ) ):
            if self.rate_data[i]["use"]:
                continue
            
            check = True
            
            if horce_num_list[0] == self.rate_data[i]["horce_num_list"][0]:
                rate += self.rate_data[i]["rate"]
                
        return rate

    def wide_select_rate( self, horce_num_list ):
        rate = 0

        for i in range( 0, len( self.rate_data ) ):
            if self.rate_data[i]["use"]:
                continue
            
            check = True
            
            for r in range( 0, len( horce_num_list ) ):
                if not horce_num_list[r] in self.rate_data[i]["horce_num_list"]:
                    check = False
                    break

            if check:
                rate += self.rate_data[i]["rate"]

        return rate

    def create_candidate( self ):
        candidate = []

        for horce_num_1 in self.wide_odds_data.keys():
            for horce_num_2 in self.wide_odds_data[horce_num_1].keys():
                horce_num_list = [ horce_num_1, horce_num_2 ]
                candidate.append( { "horce_num_list": horce_num_list, \
                                   "odds": self.wide_odds_data[horce_num_1][horce_num_2]["min"], \
                                   "kind": "wide",
                                   "use": False } )

        return candidate

    def move_rate( self, select_horce ):
        for i in range( 0, len( self.rate_data ) ):
            check = True

            for horce_num in select_horce["horce_num_list"]:
                if select_horce["kind"] == WIDE and \
                  not horce_num in self.rate_data[i]["horce_num_list"]:
                    check = False
                    break
                elif select_horce["kind"] == ONE and \
                  not horce_num == self.rate_data[i]["horce_num_list"][0]:
                    check = False
                    break                

            if check:
                self.rate_data[i]["use"] = True

    def bet_check( self, bet_list, current_odds ):
        get_money = 0

        for bet in bet_list:
            rank = 0
            check = True

            for hd in self.horce_data:
                if bet["kind"] == WIDE and hd["horce_num"] in bet["horce_num_list"]:
                    if hd["rank"] > 3:
                        check = False
                        break
                    else:
                        rank += hd["rank"]
                elif bet["kind"] == ONE and hd["horce_num"] in bet["horce_num_list"]:
                    if not hd["rank"] == 1:
                        check = False
                        break
                    else:
                        rank = hd["rank"]

            if check:
                if bet["kind"] == WIDE:
                    try:
                        get_money += ( current_odds["ワイド"][int(rank-3)] / 100 ) * bet["count"]
                    except:
                        pass
                elif bet["kind"] == ONE:
                    get_money += bet["odds"] * bet["count"]

        return get_money

    def select_horce( self ):
        bet_list = []
        use_index_list = []
        score_list = []
        self.create_rate()
        candiate_list = self.create_candidate()
        all_rate = 0
        bet_count = 0
        
        while 1:
            if bet_count >= self.use_count:
                break

            index = -1
            best_bc = 0
            max_score = -1

            for i in range( 0, len( candiate_list ) ):
                if candiate_list[i]["use"]:
                    continue

                odds = candiate_list[i]["odds"]
                bc = int( max( ( self.use_count * self.bet_rate ) / odds, 1 ) )
                #print( odds, self.use_count * self.bet_rate, ( self.use_count * self.bet_rate ) / odds )

                if bet_count + bc > self.use_count:
                    continue

                instance_bet_list = copy.deepcopy( bet_list )
                instance_bet_list.append( candiate_list[i] )
                score = self.create_score( instance_bet_list )

                if max_score < score:
                    index = i
                    max_score = score
                    best_bc = bc

            if index == -1:
                break

            if max_score < 0.03:
                break
            
            candiate_list[index]["use"] = True
            bet_list.append( candiate_list[index] )
            bet_list[-1]["count"] = best_bc
            bet_count += best_bc

        self.select_horce_list = bet_list
        return bet_list, self.create_score( bet_list ), bet_count

    def create_score( self, bet_list ):
        score = 0

        for rd in self.rate_data:
            for bet in bet_list:
                if bet["horce_num_list"][0] in rd["horce_num_list"] \
                  and bet["horce_num_list"][1] in rd["horce_num_list"]:
                    score += math.pow( rd["rate"], 2.2 ) * math.pow( bet["odds"], 0.6 )

        return score

    def create_ex_value( self ):
        if len( self.select_horce_list ) == 0 or len( self.rate_data ) == 0:
            return False

        sum_ex_value = 0
        wide_data = {}
        wide_horce_num = {}
        use_rate_data = []

        for i in range( 0, len( self.select_horce_list ) ):
            min_horce_num = min( self.select_horce_list[i]["horce_num_list"] )
            max_horce_num = max( self.select_horce_list[i]["horce_num_list"] )
            lib.dicAppend( wide_data, min_horce_num, {} )
            lib.dicAppend( wide_data, max_horce_num, {} )
            lib.dicAppend( wide_horce_num, min_horce_num, [] )
            lib.dicAppend( wide_horce_num, max_horce_num, [] )
            wide_data[min_horce_num][max_horce_num] = self.select_horce_list[i]
            wide_horce_num[min_horce_num].append( max_horce_num )
            wide_horce_num[max_horce_num].append( min_horce_num )

        horce_num_list = list( wide_horce_num.keys() )

        for rate in self.rate_data:
            if rate["use"]:
                use_rate_data.append( rate )
                continue

            c = 0
            for hm in horce_num_list:
                if hm in rate["horce_num_list"]:
                    c += 1

            if 2 <= c:
                use_rate_data.append( rate )

        for i in range( 0, len( self.select_horce_list ) ):
            ex_value = 0
            min_horce_num = min( self.select_horce_list[i]["horce_num_list"] )
            max_horce_num = max( self.select_horce_list[i]["horce_num_list"] )

            for rate in use_rate_data:
                if not min_horce_num in rate["horce_num_list"] or not max_horce_num in rate["horce_num_list"]:
                    continue

                for hm in wide_horce_num[min_horce_num]:
                    if hm == max_horce_num:
                        continue

                    if hm in rate["horce_num_list"]:
                        ex_value += rate["rate"] * wide_data[min(min_horce_num,hm)][max(min_horce_num,hm)]["count"] * wide_data[min(min_horce_num,hm)][max(min_horce_num,hm)]["odds"]

                for hm in wide_horce_num[max_horce_num]:
                    if hm == min_horce_num:
                        continue

                    if hm in rate["horce_num_list"]:
                        ex_value += rate["rate"] * wide_data[min(max_horce_num,hm)][max(max_horce_num,hm)]["count"] * wide_data[min(max_horce_num,hm)][max(max_horce_num,hm)]["odds"]

                ex_value += rate["rate"] * self.select_horce_list[i]["odds"] * self.select_horce_list[i]["count"]

            self.select_horce_list[i]["ex_value"] = ex_value
            sum_ex_value += ex_value

        return sum_ex_value

    def create_bet_rate( self, money, ex_value ):
        r = 0.01

        if ex_value > 21:
            r = 0.03
        elif ex_value > 18:
            r = 0.02

        use_money = money * r
        self.bet_rate = int( use_money / self.use_count )
