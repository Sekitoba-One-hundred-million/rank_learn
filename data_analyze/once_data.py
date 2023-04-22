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
#from sekitoba_data_create import parent_data_get

from common.name import Name

data_name = Name()

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )
dm.dl.file_set( "parent_id_data.pickle" )
dm.dl.file_set( "omega_index_data.pickle" )
dm.dl.file_set( "race_day.pickle" )
dm.dl.file_set( "parent_id_data.pickle" )
dm.dl.file_set( "horce_sex_data.pickle" )
dm.dl.file_set( "race_jockey_id_data.pickle" )
dm.dl.file_set( "horce_jockey_true_skill_data.pickle" )

class OnceData:
    def __init__( self ):
        self.race_data = dm.dl.data_get( "race_data.pickle" )
        self.race_info = dm.dl.data_get( "race_info_data.pickle" )
        self.horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
        self.baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
        self.parent_id_data = dm.dl.data_get( "parent_id_data.pickle" )
        self.omega_index_data = dm.dl.data_get( "omega_index_data.pickle" )
        self.race_day = dm.dl.data_get( "race_day.pickle" )
        self.parent_id_data = dm.dl.data_get( "parent_id_data.pickle" )
        self.horce_sex_data = dm.dl.data_get( "horce_sex_data.pickle" )
        self.race_jockey_id_data = dm.dl.data_get( "race_jockey_id_data.pickle" )
        self.horce_jockey_true_skill_data = dm.dl.data_get( "horce_jockey_true_skill_data.pickle" )
        
        self.race_high_level = RaceHighLevel()
        self.race_type = RaceType()
        self.time_index = TimeIndexGet()
        self.trainer_data = TrainerData()
        self.jockey_data = JockeyData()
        self.before_data = BeforeData()
        self.train_index = TrainIndexGet()

        self.data_name_list = []
        self.write_data_list = []
        self.simu_data = {}
        self.result = { "answer": [], "teacher": [], "query": [], "year": [], "level": [] }
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

        current_high_level = self.race_high_level.current_high_level( race_id )
        teacher_data = []
        answer_data = []
        year_data = []

        count = 0
        race_limb = {}
        current_race_data = {}
        current_race_data[data_name.speed_index] = []
        current_race_data[data_name.horce_true_skill_index] = []
        current_race_data[data_name.jockey_true_skill_index] = []
        current_race_data[data_name.my_limb_count] = {}
        
        for kk in self.race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            limb_math = lib.limb_search( pd )
            key_limb = str( int( limb_math ) )
            lib.dic_append( current_race_data[data_name.my_limb_count], key_limb, 0 )
            current_race_data[data_name.my_limb_count][key_limb] += 1
            race_limb[kk] = limb_math

            try:
                horce_true_skill = self.horce_jockey_true_skill_data["horce"][race_id][horce_id]
            except:
                horce_true_skill = 25

            try:
                jockey_id = self.race_jockey_id_data[race_id][horce_id]
                jockey_true_skill = self.horce_jockey_true_skill_data["jockey"][race_id][jockey_id]
            except:
                jockey_true_skill = 25

            current_time_index = self.time_index.main( kk, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.baba_index_data[horce_id] )
            current_race_data[data_name.speed_index].append( lib.max_check( speed ) + current_time_index["max"] )
            current_race_data[data_name.horce_true_skill_index].append( horce_true_skill )
            current_race_data[data_name.jockey_true_skill_index].append( jockey_true_skill )

        sort_speed_index = sorted( current_race_data[data_name.speed_index], reverse = True )
        sort_horce_true_skill_index = sorted( current_race_data[data_name.horce_true_skill_index], reverse = True )
        sort_jockey_true_skill_index = sorted( current_race_data[data_name.jockey_true_skill_index], reverse = True )

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
            current_year = cd.year()
            horce_birth_day = int( horce_id[0:4] )
            horce_num = int( cd.horce_number() )

            try:
                omega_index_score = self.omega_index_data[race_id][horce_num-1]
            except:
                omega_index_score = 0

            if before_cd == None:
                continue

            before_speed_score = before_cd.speed()
            before_diff_score = max( before_cd.diff(), 0 ) * 10
            before_id_weight_score = self.division( min( max( before_cd.id_weight(), -10 ), 10 ), 2 )
            before_popular = before_cd.popular()
            before_passing_list = before_cd.passing_rank().split( "-" )
            before_rank = before_cd.rank()
            up3 = before_cd.up_time()
            p1, p2 = before_cd.pace()
            up3_standard_value = max( min( ( up3 - p2 ) * 5, 15 ), -10 )

            before_year = int( year ) - 1
            key_before_year = str( int( before_year ) )
            father_id = self.parent_id_data[horce_id]["father"]
            mother_id = self.parent_id_data[horce_id]["mother"]
            
            father_score = self.match_rank_score( cd, father_id )
            mother_score = self.match_rank_score( cd, mother_id )
            #stright_slope_score = self.race_type.stright_slope( cd, pd )
            #foot_used_score = self.race_type.foot_used_score_get( cd, pd )
            high_level_score = self.race_high_level.data_get( cd, pd, ymd )
            limb_math = race_limb[kk]#lib.limb_search( pd )
            key_limb = str( int( limb_math ) )
            my_limb_count_score = current_race_data[data_name.my_limb_count][key_limb]
            age = current_year - horce_birth_day
            speed_index_score = sort_speed_index.index( current_race_data[data_name.speed_index][count] )
            race_interval_score = min( max( pd.race_interval(), 0 ), 20 )
            weight_score = cd.weight() / 10
            omega_index_score = omega_index_score
            trainer_rank_score = self.trainer_data.rank( race_id, horce_id )
            jockey_rank_score = self.jockey_data.rank( race_id, horce_id )
            #popular_rank = abs( before_cd.rank() - before_cd.popular() )
            #limb_horce_number = int( limb_math * 100 + int( cd.horce_number() / 2 ) )
            horce_true_skill = current_race_data[data_name.horce_true_skill_index][count]
            jockey_true_skill = current_race_data[data_name.jockey_true_skill_index][count]
            horce_true_skill_index = sort_horce_true_skill_index.index( horce_true_skill )
            jockey_true_skill_index = sort_jockey_true_skill_index.index( jockey_true_skill )
            diff_load_weight = cd.burden_weight() - before_cd.burden_weight()

            macth_rank_score = pd.match_rank()
            money_score = pd.get_money()
            
            if not money_score == 0:
                money_score += 100
                
            burden_weight_score = cd.burden_weight()
            #before_up3_rank = self.before_data.up3_rank( before_cd )
            before_continue_not_three_rank = pd.before_continue_not_three_rank()
            #limb_place_score = int( cd.place() * 10 + limb_math )
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
            train_score = self.train_index.score_get( race_id, horce_num )
            count += 1

            t_instance = {}
            t_instance[data_name.age] = age
            t_instance[data_name.all_horce_num] = cd.all_horce_num()
            t_instance[data_name.baba] = cd.baba_status()
            t_instance[data_name.before_continue_not_three_rank] = before_continue_not_three_rank
            t_instance[data_name.before_diff] = before_diff_score
            t_instance[data_name.before_first_passing_rank] = before_first_passing_rank
            t_instance[data_name.before_id_weight] = before_id_weight_score
            t_instance[data_name.before_last_passing_rank] = before_last_passing_rank
            t_instance[data_name.before_popular] = before_popular
            t_instance[data_name.before_rank] = before_rank
            t_instance[data_name.before_speed] = before_speed_score
            t_instance[data_name.burden_weight] = burden_weight_score
            t_instance[data_name.dist_kind] = cd.dist_kind()
            t_instance[data_name.dist_kind_count] = dist_kind_count
            t_instance[data_name.father_rank] = father_score
            t_instance[data_name.horce_sex] = horce_sex
            t_instance[data_name.horce_true_skill] = horce_true_skill
            t_instance[data_name.horce_true_skill_index] = horce_true_skill_index
            t_instance[data_name.jockey_rank] = jockey_rank_score
            t_instance[data_name.jockey_true_skill] = jockey_true_skill
            t_instance[data_name.jockey_true_skill_index] = jockey_true_skill_index
            t_instance[data_name.jockey_year_rank] = jockey_year_rank_score
            t_instance[data_name.limb] = limb_math
            t_instance[data_name.match_rank] = macth_rank_score
            t_instance[data_name.money] = money_score
            t_instance[data_name.mother_rank] = mother_score
            t_instance[data_name.my_limb_count] = my_limb_count_score
            t_instance[data_name.omega] = omega_index_score
            t_instance[data_name.place_num] = place_num
            t_instance[data_name.race_interval] = race_interval_score
            t_instance[data_name.race_level_check] = high_level_score
            t_instance[data_name.speed_index] = speed_index_score
            t_instance[data_name.train_score] = train_score
            t_instance[data_name.up3_standard_value] = up3_standard_value
            t_instance[data_name.weight] = weight_score
            t_instance[data_name.weather] = cd.weather()
            t_instance[data_name.diff_load_weight] = diff_load_weight
            
            t_list = self.data_list_create( t_instance )

            if year in lib.test_years:
                lib.dic_append( self.simu_data, race_id, {} )
                self.simu_data[race_id][horce_id] = {}
                self.simu_data[race_id][horce_id]["data"] = t_list
                self.simu_data[race_id][horce_id]["answer"] = { "rank": cd.rank(),
                                                               "odds": cd.odds(),
                                                               "popular": cd.popular(),
                                                               "horce_num": cd.horce_number() }

            rank = cd.rank()
            answer_data.append( rank )
            teacher_data.append( t_list )

        if not len( answer_data ) == 0:
            self.result["answer"].append( answer_data )
            self.result["teacher"].append( teacher_data )
            self.result["year"].append( year )
            self.result["level"].append( [ current_high_level ] )
            self.result["query"].append( { "q": len( answer_data ), "year": year } )
