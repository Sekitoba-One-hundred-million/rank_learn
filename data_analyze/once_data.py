import math
import copy
from tqdm import tqdm
from mpi4py import MPI

import SekitobaLibrary as lib
import SekitobaDataManage as dm
import SekitobaPsql as ps

from SekitobaDataCreate.win_rate import WinRate
from SekitobaDataCreate.stride_ablity import StrideAblity
from SekitobaDataCreate.time_index_get import TimeIndexGet
from SekitobaDataCreate.jockey_data_get import JockeyAnalyze
from SekitobaDataCreate.trainer_data_get import TrainerAnalyze
from SekitobaDataCreate.high_level_data_get import RaceHighLevel
from SekitobaDataCreate.race_type import RaceType
from SekitobaDataCreate.before_race_score_get import BeforeRaceScore

from common.name import Name

data_name = Name()

dm.dl.file_set( "predict_first_passing_rank.pickle" )
dm.dl.file_set( "predict_last_passing_rank.pickle" )
dm.dl.file_set( "predict_up3.pickle" )

class OnceData:
    def __init__( self ):
        self.predict_first_passing_rank = dm.dl.data_get( "predict_first_passing_rank.pickle" )
        self.predict_last_passing_rank = dm.dl.data_get( "predict_last_passing_rank.pickle" )
        self.predict_up3 = dm.dl.data_get( "predict_up3.pickle" )
        #self.predict_popular = dm.dl.data_get( "predict_popular.pickle" )

        self.race_data = ps.RaceData()
        self.race_horce_data = ps.RaceHorceData()
        self.horce_data = ps.HorceData()
        self.trainer_data = ps.TrainerData()
        self.jockey_data = ps.JockeyData()

        self.stride_ablity = StrideAblity( self.race_data )
        self.race_high_level = RaceHighLevel()
        self.time_index = TimeIndexGet( self.horce_data )
        self.trainer_analyze = TrainerAnalyze( self.race_data, self.race_horce_data, self.trainer_data )
        self.jockey_analyze = JockeyAnalyze( self.race_data, self.race_horce_data, self.jockey_data )
        self.race_type = RaceType()
        self.before_race_score = BeforeRaceScore( self.race_data )

        self.data_name_list = []
        self.write_data_list = []
        self.simu_data = {}
        self.kind_score_key_list = {}
        self.kind_score_key_list[data_name.waku_three_rate] = [ "place", "dist", "limb", "baba", "kind" ]
        self.kind_score_key_list[data_name.limb_score] = [ "place", "dist", "baba", "kind" ]
        self.result = { "answer": [], "teacher": [], "query": [], "year": [], "level": [], "diff": [], "popular": [] }
        self.data_name_read()

    def data_name_read( self ):
        f = open( "common/list.txt", "r" )
        str_data_list = f.readlines()

        for str_data in str_data_list:
            self.data_name_list.append( str_data.replace( "\n", "" ) )

        self.data_name_list = sorted( self.data_name_list )

    def score_write( self ):
        f = open( "common/rank_score_data.txt", "w" )

        for data_name in self.write_data_list:
            f.write( data_name + "\n" )

        f.close()

    def data_list_create( self, data_dict ):
        result = []
        name_list = sorted( list( data_dict.keys() ) )
        write_instance = []
        
        for data_name in name_list:
            result.append( data_dict[data_name] )

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( name_list )

        return result

    def division( self, score, d ):
        if score < 0:
            score *= -1
            score /= d
            score *= -1
        else:
            score /= d

        return int( score )

    def matchRankScore( self, cd: lib.CurrentData, target_id ):
        try:
            target_data = self.horce_data[target_id]
        except:
            target_data = []
                
        target_pd = lib.PastData( target_data, [], self.race_data )
        count = 0
        score = 0
            
        for target_cd in target_pd.pastCdList():
            c = 0
                
            if target_cd.place() == cd.place():
                c += 1
                
            if target_cd.babaStatus() == cd.babaStatus():
                c += 1

            if lib.distCheck( target_cd.dist() * 1000 ) == lib.distCheck( cd.dist() * 1000 ):
                c += 1

            count += c
            score += target_cd.rank() * c

        if not count == 0:
            score /= count
                
        return int( score )

    def clear( self ):
        dm.dl.data_clear()
    
    def create( self, race_id ):
        self.race_data.get_all_data( race_id )
        self.race_horce_data.get_all_data( race_id )

        if len( self.race_horce_data.horce_id_list ) == 0:
            return

        self.horce_data.get_multi_data( self.race_horce_data.horce_id_list )
        self.trainer_data.get_multi_data( self.race_horce_data.trainer_id_list )
        self.jockey_data.get_multi_data( self.race_horce_data.jockey_id_list )

        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( self.race_data.data["place"] )
        key_dist = str( self.race_data.data["dist"] )
        key_kind = str( self.race_data.data["kind"] )      
        key_baba = str( self.race_data.data["baba"] )
        ymd = { "year": self.race_data.data["year"], \
               "month": self.race_data.data["month"], \
               "day": self.race_data.data["day"] }

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return

        predict_netkeiba_pace = lib.netkeibaPace( self.race_data.data["predict_netkeiba_pace"] )
        key_race_money_class = str( int( lib.moneyClassGet( self.race_data.data["money"] ) ) )
        current_high_level = self.race_high_level.current_high_level( race_id )
        teacher_data = []
        answer_data = []
        popular_data = []
        diff_data = []
        horce_id_list = []
        race_limb = {}
        current_race_data = {}
        new_check = False
        current_race_data[data_name.my_limb_count] = { "-1": -1 }

        for name in self.data_name_list:
            if name in current_race_data:
                continue

            current_race_data[name] = []
        
        for horce_id in self.race_horce_data.horce_id_list:
            current_data, past_data = lib.raceCheck( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.raceCheck():
                continue

            new_check = cd.newCheck()
            limb_math = lib.limbSearch( pd )
            key_limb = str( int( limb_math ) )
            before_cd = pd.beforeCd()

            before_speed = -1000
            before_diff = -1000
            before_rank = -1000
            before_race_score = -1000

            if not before_cd == None:
                before_speed = before_cd.speed()
                before_diff = before_cd.diff()
                before_rank = before_cd.rank()
                before_race_score = self.before_race_score.score_get( before_cd, limb_math, horce_id )

            if not limb_math == -1:
                lib.dicAppend( current_race_data[data_name.my_limb_count], key_limb, 0 )
                current_race_data[data_name.my_limb_count][key_limb] += 1

            jockey_id = self.race_horce_data.data[horce_id]["jockey_id"]
            trainer_id = self.race_horce_data.data[horce_id]["trainer_id"]
            race_limb[horce_id] = limb_math

            horce_true_skill = self.race_horce_data.data[horce_id]["horce_true_skill"]
            jockey_true_skill = self.race_horce_data.data[horce_id]["jockey_true_skill"]
            trainer_true_skill = self.race_horce_data.data[horce_id]["trainer_true_skill"]
            up3_horce_true_skill = self.race_horce_data.data[horce_id]["horce_up3_true_skill"]
            corner_true_skill = self.race_horce_data.data[horce_id]["horce_corner_true_skill"]

            current_year = cd.year()
            horce_birth_day = int( horce_id[0:4] )
            age = current_year - horce_birth_day
            current_time_index = self.time_index.main( horce_id, pd.pastDayList() )
            speed, up_speed, pace_speed = pd.speedIndex( self.horce_data.data[horce_id]["baba_index"] )
            corner_diff_rank_ave = pd.corner_diff_rank()
            stride_ablity_data = self.stride_ablity.ablity_create( cd, pd )

            for stride_data_key in stride_ablity_data.keys():
                for math_key in stride_ablity_data[stride_data_key].keys():
                    current_race_data[stride_data_key+"_"+math_key].append( stride_ablity_data[stride_data_key][math_key] )

            current_race_data[data_name.horce_true_skill].append( horce_true_skill )
            current_race_data[data_name.jockey_true_skill].append( jockey_true_skill )
            current_race_data[data_name.trainer_true_skill].append( trainer_true_skill )
            current_race_data[data_name.corner_true_skill].append( corner_true_skill )
            current_race_data[data_name.up3_horce_true_skill].append( up3_horce_true_skill )            
            current_race_data[data_name.corner_diff_rank_ave].append( corner_diff_rank_ave )
            current_race_data[data_name.speed_index].append( lib.maxCheck( speed ) + current_time_index["max"] )
            current_race_data[data_name.match_rank].append( pd.matchRank() )
            current_race_data[data_name.up_rate].append( pd.up_rate( key_race_money_class, self.race_data.data["up_kind_ave"] ) )
            current_race_data[data_name.burden_weight].append( cd.burdenWeight() )
            current_race_data[data_name.age].append( age )
            current_race_data[data_name.level_score].append( pd.level_score( self.race_data.data["money_class_true_skill"] ) )
            current_race_data[data_name.foot_used].append( self.race_type.foot_used_score_get( cd, pd ) )
            current_race_data[data_name.before_diff].append( before_diff )
            current_race_data[data_name.before_rank].append( before_rank )
            current_race_data[data_name.before_speed].append( before_speed )
            current_race_data[data_name.before_race_score].append( before_race_score )
            current_race_data[data_name.max_time_point].append( pd.maxTimePoint( self.race_data.data["race_time_analyze"] ) )
            current_race_data[data_name.stamina].append( pd.stamina_create( key_limb ) )
            current_race_data[data_name.best_dist].append( pd.best_dist() )
            horce_id_list.append( horce_id )

        if len( horce_id_list ) < 2:
            return

        current_key_list = []

        for data_key in current_race_data.keys():
            if not type( current_race_data[data_key] ) is list or \
              len( current_race_data[data_key] ) == 0:
                continue

            current_key_list.append( data_key )

        for data_key in current_key_list:
            current_race_data[data_key+"_index"] = sorted( current_race_data[data_key], reverse = True )
            current_race_data[data_key+"_stand"] = lib.standardization( current_race_data[data_key] )
            current_race_data[data_key+"_devi"] = lib.deviationValue( current_race_data[data_key] )            

        ave_burden_weight = lib.average( current_race_data[data_name.burden_weight] )

        for count, horce_id in enumerate( horce_id_list ):
            current_data, past_data = lib.raceCheck( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.raceCheck():
                continue
            
            before_cd = pd.beforeCd()
            place_num = int( race_place_num )
            horce_num = int( cd.horceNumber() )

            before_id_weight_score = -1000
            before_popular = -1000
            before_passing_list = [ -1000, -1000, -1000, -1000 ]
            diff_load_weight = -1000
            before_pace_up_diff = -1000

            if not before_cd == None:
                before_id_weight_score = before_cd.idWeight()
                before_popular = before_cd.popular()
                before_passing_list = before_cd.passingRank().split( "-" )
                up3 = before_cd.upTime()
                p1, p2 = before_cd.pace()
                diff_load_weight = cd.burdenWeight() - before_cd.burdenWeight()

            predict_first_passing_rank = -1
            predict_first_passing_rank_index = -1
            predict_first_passing_rank_stand = 0
            predict_last_passing_rank = -1
            predict_last_passing_rank_index = -1
            predict_last_passing_rank_stand = 0
            predict_up3 = -1
            predict_up3_index = -1
            predict_up3_stand = 0
            predict_popular = -1
            predict_popular_index = -1
            predict_popular_stand = 0

            if race_id in self.predict_first_passing_rank and horce_id in self.predict_first_passing_rank[race_id]:
                predict_first_passing_rank = self.predict_first_passing_rank[race_id][horce_id]["score"]
                predict_first_passing_rank_index = self.predict_first_passing_rank[race_id][horce_id]["index"]
                predict_first_passing_rank_stand = self.predict_first_passing_rank[race_id][horce_id]["stand"]

            if race_id in self.predict_last_passing_rank and horce_id in self.predict_last_passing_rank[race_id]:
                predict_last_passing_rank = self.predict_last_passing_rank[race_id][horce_id]["score"]
                predict_last_passing_rank_index = self.predict_last_passing_rank[race_id][horce_id]["index"]
                predict_last_passing_rank_stand = self.predict_last_passing_rank[race_id][horce_id]["stand"]

            if race_id in self.predict_up3 and horce_id in self.predict_up3[race_id]:
                predict_up3 = self.predict_up3[race_id][horce_id]["score"]
                predict_up3_index = self.predict_up3[race_id][horce_id]["index"]
                predict_up3_stand = self.predict_up3[race_id][horce_id]["stand"]

            #if race_id in self.predict_popular and horce_id in self.predict_popular[race_id]:
            #    predict_popular = self.predict_popular[race_id][horce_id]["score"]
            #    predict_popular_index = self.predict_popular[race_id][horce_id]["index"]
            #    predict_popular_stand = self.predict_popular[race_id][horce_id]["stand"]            

            before_year = int( year ) - 1
            key_before_year = str( int( before_year ) )
            high_level_score = self.race_high_level.data_get( cd, pd, ymd )
            limb_math = race_limb[horce_id]
            key_limb = str( int( limb_math ) )            
            race_interval_score = min( max( pd.race_interval(), 0 ), 20 )
            weight_score = cd.weight() / 10
            trainer_rank_score = self.trainer_analyze.rank( race_id, horce_id )
            jockey_rank_score = self.jockey_analyze.rank( race_id, horce_id )
            #win_rate_data = self.win_rate.data_get( limb_math, cd )
            base_key = {}
            kind_key_data = {}
            kind_key_data["place"] = key_place
            kind_key_data["dist"] = key_dist
            kind_key_data["baba"] = key_baba
            kind_key_data["kind"] = key_kind
            kind_key_data["limb"] = key_limb

            waku = -1

            if cd.horceNumber() < cd.allHorceNum() / 2:
                waku = 1
            else:
                waku = 2

            base_key[data_name.waku_three_rate] = str( int( waku ) )
            base_key[data_name.limb_score] = key_limb
            waku_three_rate = lib.kindScoreGet( self.race_data.data["waku_three_rate"], self.kind_score_key_list[data_name.waku_three_rate], kind_key_data, base_key[data_name.waku_three_rate] )
            ave_burden_weight_diff = lib.minus( ave_burden_weight, cd.burdenWeight() )
            money_score = pd.getMoney()
                
            burden_weight_score = cd.burdenWeight()
            before_continue_not_three_rank = pd.before_continue_not_three_rank()
            horce_sex = self.horce_data.data[horce_id]["sex"]
            dist_kind_count = pd.distKindCount()
            
            try:
                before_last_passing_rank = int( before_passing_list[-1] )
            except:
                before_last_passing_rank = 0

            try:
                before_first_passing_rank = int( before_passing_list[0] )
            except:
                before_first_passing_rank = 0

            jockey_year_rank_score = self.jockey_analyze.year_rank( horce_id, key_before_year )
            baba = cd.babaStatus()
            three_rate_score = -1
            three_rate_score_index = -1
            three_rate_score_stand = -1
            flame_evaluation_one = -1
            flame_evaluation_two = -1
            flame_evaluation_three = -1

            key_race_pace_num = str( int( race_place_num ) )
            key_day = str( int( day ) )
            key_flame_number = str( int( cd.flameNumber() ) )

            try:
                flame_evaluation_one = self.race_data.data["flame_evaluation"][key_race_pace_num][key_day][key_flame_number]["one"]
                flame_evaluation_two = self.race_data.data["flame_evaluation"][key_race_pace_num][key_day][key_flame_number]["two"]
                flame_evaluation_three = self.race_data.data["flame_evaluation"][key_race_pace_num][key_day][key_flame_number]["three"]
            except:
                pass
            
            predict_netkeiba_deployment = -1

            for t in range( 0, len( self.race_data.data["predict_netkeiba_deployment"] ) ):
                if int( horce_num ) in self.race_data.data["predict_netkeiba_deployment"][t]:
                    predict_netkeiba_deployment = t
                    break

            t_instance = {}
            #t_instance[data_name.pace] = pace
            t_instance[data_name.all_horce_num] = cd.allHorceNum()
            t_instance[data_name.ave_burden_weight_diff] = ave_burden_weight_diff
            t_instance[data_name.baba] = cd.babaStatus()
            t_instance[data_name.before_continue_not_three_rank] = before_continue_not_three_rank
            t_instance[data_name.before_first_passing_rank] = before_first_passing_rank
            t_instance[data_name.before_id_weight] = before_id_weight_score
            t_instance[data_name.before_last_passing_rank] = before_last_passing_rank
            t_instance[data_name.before_popular] = before_popular
            t_instance[data_name.burden_weight] = burden_weight_score
            t_instance[data_name.dist_kind] = cd.distKind()
            t_instance[data_name.dist_kind_count] = dist_kind_count
            t_instance[data_name.flame_evaluation_one] = flame_evaluation_one
            t_instance[data_name.flame_evaluation_two] = flame_evaluation_two
            t_instance[data_name.flame_evaluation_three] = flame_evaluation_three
            t_instance[data_name.foot_used_best] = self.race_type.best_foot_used( cd, pd )
            t_instance[data_name.predict_first_passing_rank] = predict_first_passing_rank
            t_instance[data_name.predict_first_passing_rank_index] = predict_first_passing_rank_index
            t_instance[data_name.predict_first_passing_rank_stand] = predict_first_passing_rank_stand
            t_instance[data_name.horce_num] = cd.horceNumber()
            t_instance[data_name.horce_sex] = horce_sex
            t_instance[data_name.jockey_rank] = jockey_rank_score
            t_instance[data_name.predict_last_passing_rank] = predict_last_passing_rank
            t_instance[data_name.predict_last_passing_rank_index] = predict_last_passing_rank_index
            t_instance[data_name.predict_last_passing_rank_stand] = predict_last_passing_rank_stand
            t_instance[data_name.limb] = limb_math
            t_instance[data_name.my_limb_count] = current_race_data[data_name.my_limb_count][key_limb]
            t_instance[data_name.money] = money_score
            t_instance[data_name.place] = place_num
            t_instance[data_name.race_interval] = race_interval_score
            t_instance[data_name.high_level_score] = high_level_score
            t_instance[data_name.speed_index] = current_race_data[data_name.speed_index][count]
            t_instance[data_name.speed_index_index] = \
              current_race_data[data_name.speed_index_index].index( current_race_data[data_name.speed_index][count] )
            t_instance[data_name.speed_index_stand] = current_race_data[data_name.speed_index_stand][count]
            t_instance[data_name.weight] = weight_score
            t_instance[data_name.waku_three_rate] = waku_three_rate
            t_instance[data_name.weather] = cd.weather()
            t_instance[data_name.diff_load_weight] = diff_load_weight
            t_instance[data_name.predict_up3] = predict_up3
            #t_instance[data_name.predict_up3_index] = predict_up3_index
            t_instance[data_name.predict_up3_stand] = predict_up3_stand
            t_instance[data_name.predict_netkeiba_pace] = predict_netkeiba_pace
            t_instance[data_name.predict_netkeiba_deployment] = predict_netkeiba_deployment
            #t_instance[data_name.predict_popular] = predict_popular
            #t_instance[data_name.predict_popular_index] = predict_popular_index
            #t_instance[data_name.predict_popular_stand] = predict_popular_stand

            #for rk in win_rate_data.keys():
            #    t_instance[rk] = win_rate_data[rk]
            
            str_index = "_index"
            for data_key in current_race_data.keys():
                if len( current_race_data[data_key] ) == 0 or \
                  data_key in t_instance:
                    continue

                if str_index in data_key:
                    name = data_key.replace( str_index, "" )

                    if name in current_race_data:
                        t_instance[data_key] = current_race_data[data_key].index( current_race_data[name][count] )
                else:
                    t_instance[data_key] = current_race_data[data_key][count]

            t_list = self.data_list_create( t_instance )

            if year in lib.test_years:
                key_dist_kind = str( int( cd.distKind() ) )
                key_popular = str( int( cd.popular() ) )
                popular_win_rate = { "one": 0, "two": 0, "three": 0 }
                
                try:
                    popular_win_rate = copy.deepcopy( self.popular_kind_win_rate_data[key_place][key_dist_kind][key_kind][key_popular] )
                except:
                    pass

                lib.dicAppend( self.simu_data, race_id, {} )
                self.simu_data[race_id][horce_id] = {}
                self.simu_data[race_id][horce_id]["data"] = t_list
                self.simu_data[race_id][horce_id]["answer"] = { "rank": cd.rank(),
                                                               "odds": cd.odds(),
                                                               "popular": cd.popular(),
                                                               "horce_num": cd.horceNumber(),
                                                               "race_kind": cd.raceKind(),
                                                               "popular_win_rate": popular_win_rate,
                                                               "new": new_check }

            rank = cd.rank()
            answer_data.append( rank )
            teacher_data.append( t_list )
            diff_data.append( cd.diff() )
            popular_data.append( cd.popular() )

        if not len( answer_data ) == 0:
            self.result["answer"].append( answer_data )
            self.result["teacher"].append( teacher_data )
            self.result["year"].append( year )
            self.result["level"].append( [ current_high_level ] )
            self.result["query"].append( { "q": len( answer_data ), "year": year } )
            self.result["diff"].append( diff_data )
            self.result["popular"].append( popular_data )
