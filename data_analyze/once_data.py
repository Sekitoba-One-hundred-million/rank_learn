import math
import copy
import sklearn
from tqdm import tqdm
from mpi4py import MPI

import sekitoba_library as lib
import sekitoba_data_manage as dm

from sekitoba_data_create.time_index_get import TimeIndexGet
#from sekitoba_data_create.up_score import UpScore
from sekitoba_data_create.train_index_get import TrainIndexGet
#from sekitoba_data_create.pace_time_score import PaceTimeScore
from sekitoba_data_create.jockey_data_get import JockeyData
from sekitoba_data_create.trainer_data_get import TrainerData
from sekitoba_data_create.high_level_data_get import RaceHighLevel
from sekitoba_data_create.race_type import RaceType
from sekitoba_data_create.before_data import BeforeData
from sekitoba_data_create.before_race_score_get import BeforeRaceScore
#from sekitoba_data_create import parent_data_get

from common.name import Name

data_name = Name()

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )
dm.dl.file_set( "parent_id_data.pickle" )
dm.dl.file_set( "race_day.pickle" )
dm.dl.file_set( "horce_sex_data.pickle" )
dm.dl.file_set( "race_jockey_id_data.pickle" )
dm.dl.file_set( "race_trainer_id_data.pickle" )
dm.dl.file_set( "true_skill_data.pickle" )
dm.dl.file_set( "race_money_data.pickle" )
dm.dl.file_set( "waku_three_rate_data.pickle" )
dm.dl.file_set( "corner_true_skill_data.pickle" )
dm.dl.file_set( "wrap_data.pickle" )
dm.dl.file_set( "predict_first_passing_rank.pickle" )
dm.dl.file_set( "predict_last_passing_rank.pickle" )
dm.dl.file_set( "first_corner_rank.pickle" )
dm.dl.file_set( "up3_true_skill_data.pickle" )
dm.dl.file_set( "predict_train_score.pickle" )
dm.dl.file_set( "predict_up3.pickle" )
dm.dl.file_set( "popular_kind_win_rate_data.pickle" )

class OnceData:
    def __init__( self ):
        self.race_data = dm.dl.data_get( "race_data.pickle" )
        self.race_info = dm.dl.data_get( "race_info_data.pickle" )
        self.horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
        self.baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
        self.parent_id_data = dm.dl.data_get( "parent_id_data.pickle" )
        self.race_day = dm.dl.data_get( "race_day.pickle" )
        self.horce_sex_data = dm.dl.data_get( "horce_sex_data.pickle" )
        self.race_jockey_id_data = dm.dl.data_get( "race_jockey_id_data.pickle" )
        self.race_trainer_id_data = dm.dl.data_get( "race_trainer_id_data.pickle" )
        self.true_skill_data = dm.dl.data_get( "true_skill_data.pickle" )
        self.up3_true_skill_data = dm.dl.data_get( "up3_true_skill_data.pickle" )
        self.race_money_data = dm.dl.data_get( "race_money_data.pickle" )
        self.waku_three_rate_data = dm.dl.data_get( "waku_three_rate_data.pickle" )
        self.corner_true_skill_data = dm.dl.data_get( "corner_true_skill_data.pickle" )
        self.wrap_data = dm.dl.data_get( "wrap_data.pickle" )
        self.predict_first_passing_rank = dm.dl.data_get( "predict_first_passing_rank.pickle" )
        self.predict_last_passing_rank = dm.dl.data_get( "predict_last_passing_rank.pickle" )
        self.predict_train_score = dm.dl.data_get( "predict_train_score.pickle" )
        self.first_corner_rank = dm.dl.data_get( "first_corner_rank.pickle" )
        self.predict_up3 = dm.dl.data_get( "predict_up3.pickle" )
        self.popular_kind_win_rate_data = dm.dl.data_get( "popular_kind_win_rate_data.pickle" )
        
        self.race_high_level = RaceHighLevel()
        self.race_type = RaceType()
        self.time_index = TimeIndexGet()
        self.trainer_data = TrainerData()
        self.jockey_data = JockeyData()
        self.before_data = BeforeData()
        self.train_index = TrainIndexGet()
        self.before_race_score = BeforeRaceScore()

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

    def score_write( self ):
        f = open( "common/rank_score_data.txt", "w" )

        for data_name in self.write_data_list:
            f.write( data_name + "\n" )

        f.close()

    def data_list_create( self, data_dict ):
        result = []
        write_instance = []
        
        for data_name in self.data_name_list:
            try:
                result.append( data_dict[data_name] )
                write_instance.append( data_name )
            except:
                continue

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( write_instance )

        return result

    def division( self, score, d ):
        if score < 0:
            score *= -1
            score /= d
            score *= -1
        else:
            score /= d

        return int( score )

    def match_rank_score( self, cd: lib.current_data, target_id ):
        try:
            target_data = self.horce_data[target_id]
        except:
            target_data = []
                
        target_pd = lib.past_data( target_data, [] )
        count = 0
        score = 0
            
        for target_cd in target_pd.past_cd_list():
            c = 0
                
            if target_cd.place() == cd.place():
                c += 1
                
            if target_cd.baba_status() == cd.baba_status():
                c += 1

            if lib.dist_check( target_cd.dist() * 1000 ) == lib.dist_check( cd.dist() * 1000 ):
                c += 1

            count += c
            score += target_cd.rank() * c

        if not count == 0:
            score /= count
                
        return int( score )

    def clear( self ):
        dm.dl.data_clear()
    
    def create( self, k ):
        race_id = lib.id_get( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( self.race_info[race_id]["place"] )
        key_dist = str( self.race_info[race_id]["dist"] )
        key_kind = str( self.race_info[race_id]["kind"] )      
        key_baba = str( self.race_info[race_id]["baba"] )
        ymd = { "y": int( year ), "m": self.race_day[race_id]["month"], "d": self.race_day[race_id]["day"] }
        #ri_list = [ key_place + ":place", key_dist + ":dist", key_kind + ":kind", key_baba + ":baba" ]        
        #info_key_dist = key_dist

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return

        if not race_id in self.race_money_data:
            return

        if not race_id in self.first_corner_rank:
            return
            
        key_race_money_class = str( int( lib.money_class_get( self.race_money_data[race_id] ) ) )
        current_high_level = self.race_high_level.current_high_level( race_id )
        teacher_data = []
        answer_data = []
        popular_data = []
        diff_data = []

        count = 0
        race_limb = {}
        current_race_data = {}
        current_race_data[data_name.horce_true_skill] = []
        current_race_data[data_name.jockey_true_skill] = []
        current_race_data[data_name.trainer_true_skill] = []
        current_race_data[data_name.up3_horce_true_skill] = []
        current_race_data[data_name.corner_diff_rank_ave] = []
        current_race_data[data_name.corner_true_skill] = []
        current_race_data[data_name.match_rank] = []
        current_race_data[data_name.speed_index] = []
        current_race_data[data_name.up_rate] = []
        current_race_data[data_name.my_limb_count] = { "-1": -1 }
        current_race_data[data_name.burden_weight] = []
        current_race_data[data_name.age] = []
        current_race_data[data_name.level_score] = []
        current_race_data[data_name.train_score] = []
        current_race_data[data_name.foot_used] = []
        
        for horce_id in self.race_data[k].keys():
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            limb_math = lib.limb_search( pd )

            if not limb_math == -1:
                key_limb = str( int( limb_math ) )
                lib.dic_append( current_race_data[data_name.my_limb_count], key_limb, 0 )
                current_race_data[data_name.my_limb_count][key_limb] += 1

            jockey_id = ""
            trainer_id = ""

            try:
                jockey_id = self.race_jockey_id_data[race_id][horce_id]
            except:
                pass

            try:
                trainer_id = self.race_trainer_id_data[race_id][horce_id]
            except:
                pass

            race_limb[horce_id] = limb_math
            horce_true_skill = 25
            jockey_true_skill = 25
            trainer_true_skill = 25
            corner_true_skill = 25
            up3_horce_true_skill = 25

            if race_id in self.true_skill_data["horce"] and \
              horce_id in self.true_skill_data["horce"][race_id]:
                horce_true_skill = self.true_skill_data["horce"][race_id][horce_id]

            if race_id in self.true_skill_data["jockey"] and \
              jockey_id in self.true_skill_data["jockey"][race_id]:
                jockey_true_skill = self.true_skill_data["jockey"][race_id][jockey_id]

            if race_id in self.true_skill_data["trainer"] and \
              trainer_id in self.true_skill_data["trainer"][race_id]:
                trainer_true_skill = self.true_skill_data["trainer"][race_id][trainer_id]

            if race_id in self.corner_true_skill_data["horce"] and \
              horce_id in self.corner_true_skill_data["horce"][race_id]:
                corner_true_skill = self.corner_true_skill_data["horce"][race_id][horce_id]

            if race_id in self.up3_true_skill_data["horce"] and \
              horce_id in self.up3_true_skill_data["horce"][race_id]:
                up3_horce_true_skill = self.up3_true_skill_data["horce"][race_id][horce_id]

            train_score = -10000

            if race_id in self.predict_train_score and horce_id in self.predict_train_score[race_id]:
                train_score = self.predict_train_score[race_id][horce_id]

            current_year = cd.year()
            horce_birth_day = int( horce_id[0:4] )
            age = current_year - horce_birth_day
            current_time_index = self.time_index.main( horce_id, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.baba_index_data[horce_id] )
            corner_diff_rank_ave = pd.corner_diff_rank()
            current_race_data[data_name.horce_true_skill].append( horce_true_skill )
            current_race_data[data_name.jockey_true_skill].append( jockey_true_skill )
            current_race_data[data_name.trainer_true_skill].append( trainer_true_skill )
            current_race_data[data_name.corner_true_skill].append( corner_true_skill )
            current_race_data[data_name.up3_horce_true_skill].append( up3_horce_true_skill )            
            current_race_data[data_name.corner_diff_rank_ave].append( corner_diff_rank_ave )
            current_race_data[data_name.speed_index].append( lib.max_check( speed ) + current_time_index["max"] )
            current_race_data[data_name.match_rank].append( pd.match_rank() )
            current_race_data[data_name.up_rate].append( pd.up_rate( key_race_money_class ) )
            current_race_data[data_name.burden_weight].append( cd.burden_weight() )
            current_race_data[data_name.age].append( age )
            current_race_data[data_name.level_score].append( pd.level_score() )
            current_race_data[data_name.train_score].append( train_score )
            current_race_data[data_name.foot_used].append( self.race_type.foot_used_score_get( cd, pd ) )

        if len( current_race_data[data_name.burden_weight] ) == 0:
            return

        sort_race_data: dict[ str, list ] = {}
        ave_burden_weight = sum( current_race_data[data_name.burden_weight] ) / len( current_race_data[data_name.burden_weight] )
        #ave_age = sum( current_race_data[data_name.age] ) / len( current_race_data[data_name.age] )
        sort_race_data[data_name.speed_index_index] = sorted( current_race_data[data_name.speed_index], reverse = True )
        sort_race_data[data_name.horce_true_skill_index] = sorted( current_race_data[data_name.horce_true_skill], reverse = True )
        sort_race_data[data_name.jockey_true_skill_index] = sorted( current_race_data[data_name.jockey_true_skill], reverse = True )
        sort_race_data[data_name.trainer_true_skill_index] = sorted( current_race_data[data_name.trainer_true_skill], reverse = True )
        sort_race_data[data_name.corner_true_skill_index] = sorted( current_race_data[data_name.corner_true_skill], reverse = True )
        sort_race_data[data_name.up3_horce_true_skill_index] = sorted( current_race_data[data_name.up3_horce_true_skill], reverse = True )
        sort_race_data[data_name.corner_diff_rank_ave_index] = sorted( current_race_data[data_name.corner_diff_rank_ave], reverse = True )
        sort_race_data[data_name.match_rank_index] = sorted( current_race_data[data_name.match_rank], reverse = True )
        sort_race_data[data_name.up_rate_index] = sorted( current_race_data[data_name.up_rate], reverse = True )
        sort_race_data[data_name.level_score_index] = sorted( current_race_data[data_name.level_score], reverse = True )
        sort_race_data[data_name.train_score_index] = sorted( current_race_data[data_name.train_score], reverse = True )
        sort_race_data[data_name.foot_used_index] = sorted( current_race_data[data_name.foot_used], reverse = True )

        speed_index_stand = lib.standardization( current_race_data[data_name.speed_index] )
        horce_true_skill_stand = lib.standardization( current_race_data[data_name.horce_true_skill] )
        jockey_true_skill_stand = lib.standardization( current_race_data[data_name.jockey_true_skill] )
        trainer_true_skill_stand = lib.standardization( current_race_data[data_name.trainer_true_skill] )
        corner_true_skill_stand = lib.standardization( current_race_data[data_name.corner_true_skill] )
        up3_horce_true_skill_stand = lib.standardization( current_race_data[data_name.up3_horce_true_skill] )
        corner_diff_rank_ave_stand = lib.standardization( current_race_data[data_name.corner_diff_rank_ave] )
        match_rank_stand = lib.standardization( current_race_data[data_name.match_rank] )
        up_rate_stand = lib.standardization( current_race_data[data_name.up_rate] )
        level_score_stand = lib.standardization( current_race_data[data_name.level_score] )
        train_score_stand = lib.standardization( current_race_data[data_name.train_score] )
        foot_used_stand = lib.standardization( current_race_data[data_name.foot_used] )

        for kk in self.race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue
            
            before_cd = pd.before_cd()
            place_num = int( race_place_num )
            horce_num = int( cd.horce_number() )

            before_speed_score = -1
            before_diff_score = 1000
            before_id_weight_score = 1000
            before_popular = -1
            before_passing_list = [ -1, -1, -1, -1 ]
            before_rank = -1
            diff_load_weight = -1000
            before_pace_up_diff = -1000

            if not before_cd == None:
                before_speed_score = before_cd.speed()
                before_diff_score = max( before_cd.diff(), 0 ) * 10
                before_id_weight_score = self.division( min( max( before_cd.id_weight(), -10 ), 10 ), 2 )
                before_popular = before_cd.popular()
                before_passing_list = before_cd.passing_rank().split( "-" )
                before_rank = before_cd.rank()
                up3 = before_cd.up_time()
                p1, p2 = before_cd.pace()
                diff_load_weight = cd.burden_weight() - before_cd.burden_weight()

            predict_first_passing_rank = -1
            predict_first_passing_rank_index = -1
            predict_first_passing_rank_stand = 0
            predict_last_passing_rank = -1
            predict_last_passing_rank_index = -1
            predict_last_passing_rank_stand = 0
            predict_up3 = -1
            predict_up3_index = -1
            predict_up3_stand = 0

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

            before_year = int( year ) - 1
            key_before_year = str( int( before_year ) )
            father_id = self.parent_id_data[horce_id]["father"]
            mother_id = self.parent_id_data[horce_id]["mother"]

            father_match_rank = self.match_rank_score( cd, father_id )
            mother_match_rank = self.match_rank_score( cd, mother_id )
            #stright_slope_score = self.race_type.stright_slope( cd, pd )
            high_level_score = self.race_high_level.data_get( cd, pd, ymd )
            limb_math = race_limb[kk]#lib.limb_search( pd )
            key_limb = str( int( limb_math ) )            
            race_interval_score = min( max( pd.race_interval(), 0 ), 20 )
            weight_score = cd.weight() / 10
            trainer_rank_score = self.trainer_data.rank( race_id, horce_id )
            jockey_rank_score = self.jockey_data.rank( race_id, horce_id )
            #popular_rank = abs( before_cd.rank() - before_cd.popular() )
            #limb_horce_number = int( limb_math * 100 + int( cd.horce_number() / 2 ) )
            before_race_score = self.before_race_score.score_get( before_cd, limb_math, horce_id )
            
            base_key = {}
            kind_key_data = {}
            kind_key_data["place"] = key_place
            kind_key_data["dist"] = key_dist
            kind_key_data["baba"] = key_baba
            kind_key_data["kind"] = key_kind
            kind_key_data["limb"] = key_limb

            waku = -1

            if cd.horce_number() < cd.all_horce_num() / 2:
                waku = 1
            else:
                waku = 2

            base_key[data_name.waku_three_rate] = str( int( waku ) )
            base_key[data_name.limb_score] = key_limb
            waku_three_rate = lib.kind_score_get( self.waku_three_rate_data, self.kind_score_key_list[data_name.waku_three_rate], kind_key_data, base_key[data_name.waku_three_rate] )

            my_limb_count_score = current_race_data[data_name.my_limb_count][key_limb]
            horce_true_skill = current_race_data[data_name.horce_true_skill][count]
            jockey_true_skill = current_race_data[data_name.jockey_true_skill][count]
            trainer_true_skill = current_race_data[data_name.trainer_true_skill][count]
            corner_true_skill = current_race_data[data_name.corner_true_skill][count]
            up3_horce_true_skill = current_race_data[data_name.up3_horce_true_skill][count]
            corner_diff_rank_ave = current_race_data[data_name.corner_diff_rank_ave][count]
            speed_index = current_race_data[data_name.speed_index][count]
            match_rank = current_race_data[data_name.match_rank][count]
            up_rate = current_race_data[data_name.up_rate][count]
            age = current_race_data[data_name.age][count]
            level_score = min( current_race_data[data_name.level_score][count], 3 )
            train_score = current_race_data[data_name.train_score][count]
            foot_used = current_race_data[data_name.foot_used][count]
            
            horce_true_skill_index = sort_race_data[data_name.horce_true_skill_index].index( horce_true_skill )
            jockey_true_skill_index = sort_race_data[data_name.jockey_true_skill_index].index( jockey_true_skill )
            trainer_true_skill_index = sort_race_data[data_name.trainer_true_skill_index].index( trainer_true_skill )
            corner_true_skill_index = sort_race_data[data_name.corner_true_skill_index].index( corner_true_skill )
            up3_horce_true_skill_index = sort_race_data[data_name.up3_horce_true_skill_index].index( up3_horce_true_skill )
            corner_diff_rank_ave_index = sort_race_data[data_name.corner_diff_rank_ave_index].index( corner_diff_rank_ave )
            speed_index_index = sort_race_data[data_name.speed_index_index].index( speed_index )
            match_rank_index = sort_race_data[data_name.match_rank_index].index( match_rank )
            up_rate_index = sort_race_data[data_name.up_rate_index].index( up_rate )
            level_score_index = sort_race_data[data_name.level_score_index].index( level_score )
            train_score_index = sort_race_data[data_name.train_score_index].index( train_score )
            foot_used_index = sort_race_data[data_name.foot_used_index].index( foot_used )

            ave_burden_weight_diff = ave_burden_weight - cd.burden_weight()
            #ave_age_diff = ave_age - age
            money_score = pd.get_money()
            
            if not money_score == 0:
                money_score += 100
                
            burden_weight_score = cd.burden_weight()
            before_continue_not_three_rank = pd.before_continue_not_three_rank()
            horce_sex = self.horce_sex_data[horce_id]
            dist_kind_count = pd.dist_kind_count()
            
            try:
                before_last_passing_rank = int( before_passing_list[-1] )
            except:
                before_last_passing_rank = 0

            try:
                before_first_passing_rank = int( before_passing_list[0] )
            except:
                before_first_passing_rank = 0

            jockey_year_rank_score = self.jockey_data.year_rank( race_id, horce_id, key_before_year )
            baba = cd.baba_status()
            three_rate_score = -1
            three_rate_score_index = -1
            three_rate_score_stand = -1

            t_instance = {}
            #t_instance[data_name.pace] = pace
            t_instance[data_name.all_horce_num] = cd.all_horce_num()
            t_instance[data_name.ave_burden_weight_diff] = ave_burden_weight_diff
            t_instance[data_name.baba] = cd.baba_status()
            t_instance[data_name.before_continue_not_three_rank] = before_continue_not_three_rank
            t_instance[data_name.before_diff] = before_diff_score
            t_instance[data_name.before_first_passing_rank] = before_first_passing_rank
            t_instance[data_name.before_id_weight] = before_id_weight_score
            t_instance[data_name.before_last_passing_rank] = before_last_passing_rank
            t_instance[data_name.before_popular] = before_popular
            t_instance[data_name.before_rank] = before_rank
            t_instance[data_name.before_race_score] = before_race_score
            t_instance[data_name.before_speed] = before_speed_score
            t_instance[data_name.burden_weight] = burden_weight_score
            t_instance[data_name.corner_diff_rank_ave] = corner_diff_rank_ave
            t_instance[data_name.corner_diff_rank_ave_index] = corner_diff_rank_ave_index
            t_instance[data_name.corner_diff_rank_ave_stand] = corner_diff_rank_ave_stand[count]
            t_instance[data_name.corner_true_skill] = corner_true_skill
            t_instance[data_name.corner_true_skill_index] = corner_true_skill_index
            t_instance[data_name.corner_true_skill_stand] = corner_true_skill_stand[count]
            t_instance[data_name.dist_kind] = cd.dist_kind()
            t_instance[data_name.dist_kind_count] = dist_kind_count
            t_instance[data_name.father_rank] = father_match_rank
            t_instance[data_name.foot_used] = foot_used
            t_instance[data_name.foot_used_index] = foot_used_index
            t_instance[data_name.foot_used_stand] = foot_used_stand[count]
            t_instance[data_name.foot_used_best] = self.race_type.best_foot_used( cd, pd )
            t_instance[data_name.predict_first_passing_rank] = predict_first_passing_rank
            t_instance[data_name.predict_first_passing_rank_index] = predict_first_passing_rank_index
            t_instance[data_name.predict_first_passing_rank_stand] = predict_first_passing_rank_stand
            t_instance[data_name.horce_num] = cd.horce_number()
            t_instance[data_name.horce_sex] = horce_sex
            t_instance[data_name.horce_true_skill] = horce_true_skill
            t_instance[data_name.horce_true_skill_index] = horce_true_skill_index
            t_instance[data_name.horce_true_skill_stand] = horce_true_skill_stand[count]
            t_instance[data_name.jockey_rank] = jockey_rank_score
            t_instance[data_name.jockey_true_skill] = jockey_true_skill
            t_instance[data_name.jockey_true_skill_index] = jockey_true_skill_index
            t_instance[data_name.jockey_true_skill_stand] = jockey_true_skill_stand[count]
            t_instance[data_name.jockey_year_rank] = jockey_year_rank_score
            t_instance[data_name.trainer_true_skill] = trainer_true_skill
            t_instance[data_name.trainer_true_skill_index] = trainer_true_skill_index
            t_instance[data_name.trainer_true_skill_stand] = trainer_true_skill_stand[count]
            t_instance[data_name.level_score] = level_score
            t_instance[data_name.level_score_index] = level_score_index
            t_instance[data_name.level_score_stand] = level_score_stand[count]
            t_instance[data_name.predict_last_passing_rank] = predict_last_passing_rank
            t_instance[data_name.predict_last_passing_rank_index] = predict_last_passing_rank_index
            t_instance[data_name.predict_last_passing_rank_stand] = predict_last_passing_rank_stand
            t_instance[data_name.limb] = limb_math
            t_instance[data_name.match_rank] = match_rank
            t_instance[data_name.match_rank_index] = match_rank_index
            t_instance[data_name.match_rank_stand] = match_rank_stand[count]
            t_instance[data_name.money] = money_score
            t_instance[data_name.mother_rank] = mother_match_rank
            t_instance[data_name.my_limb_count] = my_limb_count_score
            t_instance[data_name.place] = place_num
            t_instance[data_name.race_interval] = race_interval_score
            t_instance[data_name.race_level_check] = high_level_score
            t_instance[data_name.speed_index] = speed_index
            t_instance[data_name.speed_index_index] = speed_index_index
            t_instance[data_name.speed_index_stand] = speed_index_stand[count]
            t_instance[data_name.train_score] = train_score
            t_instance[data_name.train_score_index] = train_score_index
            t_instance[data_name.train_score_stand] = train_score_stand[count]
            t_instance[data_name.up3_horce_true_skill] = up3_horce_true_skill
            t_instance[data_name.up3_horce_true_skill_index] = up3_horce_true_skill_index
            t_instance[data_name.up3_horce_true_skill_stand] = up3_horce_true_skill_stand[count]
            t_instance[data_name.up_rate] = up_rate
            t_instance[data_name.up_rate_index] = up_rate_index
            t_instance[data_name.up_rate_stand] = up_rate_stand[count]
            t_instance[data_name.weight] = weight_score
            t_instance[data_name.waku_three_rate] = waku_three_rate
            t_instance[data_name.weather] = cd.weather()
            t_instance[data_name.diff_load_weight] = diff_load_weight
            t_instance[data_name.predict_up3] = predict_up3
            t_instance[data_name.predict_up3_index] = predict_up3_index
            t_instance[data_name.predict_up3_stand] = predict_up3_stand

            count += 1
            t_list = self.data_list_create( t_instance )

            if year in lib.test_years:
                key_dist_kind = str( int( cd.dist_kind() ) )
                key_popular = str( int( cd.popular() ) )
                popular_win_rate = { "one": 0, "two": 0, "three": 0 }
                
                try:
                    popular_win_rate = copy.deepcopy( self.popular_kind_win_rate_data[key_place][key_dist_kind][key_kind][key_popular] )
                except:
                    pass

                lib.dic_append( self.simu_data, race_id, {} )
                self.simu_data[race_id][horce_id] = {}
                self.simu_data[race_id][horce_id]["data"] = t_list
                self.simu_data[race_id][horce_id]["answer"] = { "rank": cd.rank(),
                                                               "odds": cd.odds(),
                                                               "popular": cd.popular(),
                                                               "horce_num": cd.horce_number(),
                                                               "race_kind": cd.race_kind(),
                                                               "popular_win_rate": popular_win_rate }

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
