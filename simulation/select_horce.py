import copy
from tqdm import tqdm

import sekitoba_library as lib
import sekitoba_data_manage as dm

WIDE = "wide"
ONE = "one"

class SelectHorce:
    def __init__( self, wide_odds_data, horce_data ):
        self.use_count = 10
        self.bet_rate = 1
        self.goal_rate = 1.4
        self.bet_result_count = 0
        self.wide_odds_data = wide_odds_data
        self.horce_data = horce_data
        self.rate_data = []

    def create_bet_rate( self, money ):
        self.bet_rate = min( int( money / 400 ), 30 )

    def create_rate( self ):
        all_rate = 0
        
        for i in range( 0, len( self.horce_data ) ):
            score_1 = self.horce_data[i]["score"]
            horce_num_1 = self.horce_data[i]["horce_num"]
            
            for r in range( 0, len( self.horce_data ) ):
                score_2 = self.horce_data[r]["score"]
                horce_num_2 = self.horce_data[r]["horce_num"]

                if horce_num_1 == horce_num_2:
                    continue
                
                for t in range( r + 1, len( self.horce_data ) ):
                    score_3 = self.horce_data[t]["score"]
                    horce_num_3 = self.horce_data[t]["horce_num"]

                    if horce_num_1 == horce_num_3 or \
                      horce_num_2 == horce_num_3:
                        continue
                    
                    rate = score_1 * ( score_2 / 2 ) * ( score_3 / 3 )

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

        #for horce in self.horce_data:
        #    horce_num_list = [ horce["horce_num"] ]
        #    candidate.append( { "rate": self.one_select_rate( horce_num_list ),
        #                       "horce_num_list": horce_num_list,
        #                       "odds": horce["odds"],
        #                       "kind": "one" } )

        for horce_num_1 in self.wide_odds_data.keys():
            for horce_num_2 in self.wide_odds_data[horce_num_1].keys():
                horce_num_list = [ horce_num_1, horce_num_2 ]
                candidate.append( { "rate": self.wide_select_rate( horce_num_list ), \
                                   "horce_num_list": horce_num_list, \
                                   "odds": self.wide_odds_data[horce_num_1][horce_num_2]["min"],
                                   "kind": "wide" } )

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
        self.create_rate()
        bet_count = self.use_count
        candiate_list = self.create_candidate()
        all_rate = 0

        while 1:
            if bet_count <= 0:
                break

            max_score = 0
            select_horce = None

            for candiate in candiate_list:
                need_count = max( int( ( self.use_count * self.goal_rate ) / candiate["odds"] + 1 ), 1 )

                if bet_count < need_count:
                    continue
                
                score = candiate["rate"] / need_count

                if max_score < score:
                    select_horce = copy.deepcopy( candiate )
                    select_horce["count"] = need_count
                    max_score = score

            if select_horce == None or max_score == 0:
                break

            bet_count -= select_horce["count"]
            self.bet_result_count += select_horce["count"]
            all_rate += select_horce["rate"]
            bet_list.append( copy.deepcopy( select_horce ) )
            self.move_rate( select_horce )
            candiate_list = self.create_candidate()

        #print( bet_list )
        return bet_list, all_rate
