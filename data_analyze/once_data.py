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
from SekitobaDataCreate.get_horce_data import GetHorceData
from SekitobaDataCreate.kinetic_energy import KineticEnergy
from SekitobaDataCreate.blood_type_score import BloodTypeScore

from common.name import Name

data_name = Name()

dm.dl.file_set( "predict_first_passing_rank.pickle" )
dm.dl.file_set( "predict_last_passing_rank.pickle" )
dm.dl.file_set( "predict_up3.pickle" )
dm.dl.file_set( "predict_first_up3.pickle" )
dm.dl.file_set( "predict_diff.pickle" )
dm.dl.file_set( "predict_first_up3.pickle" )
dm.dl.file_set( "predict_time_index.pickle" )
dm.dl.file_set( "predict_test.pickle" )
dm.dl.file_set( "predict_race_time.pickle" )

class OnceData:
    def __init__( self ):
        self.predict_data = {}
        self.predict_data[data_name.predict_first_passing_rank] = dm.dl.data_get( "predict_first_passing_rank.pickle" )
        self.predict_data[data_name.predict_last_passing_rank] = dm.dl.data_get( "predict_last_passing_rank.pickle" )
        self.predict_data[data_name.predict_first_up3] = dm.dl.data_get( "predict_first_up3.pickle" )
        self.predict_data[data_name.predict_up3] = dm.dl.data_get( "predict_up3.pickle" )
        self.predict_data[data_name.predict_diff] = dm.dl.data_get( "predict_diff.pickle" )
        self.predict_data[data_name.predict_race_time] = dm.dl.data_get( "predict_race_time.pickle" )
        self.predict_data[data_name.predict_time_index] = dm.dl.data_get( "predict_time_index.pickle" )
        #self.predict_data[data_name.predict_test] = dm.dl.data_get( "predict_test.pickle" )

        self.race_data = ps.RaceData()
        self.race_horce_data = ps.RaceHorceData()
        self.horce_data = ps.HorceData()
        self.trainer_data = ps.TrainerData()
        self.jockey_data = ps.JockeyData()

        self.kinetic_energy = KineticEnergy( self.race_data )
        self.stride_ablity = StrideAblity( self.race_data )
        self.race_high_level = RaceHighLevel()
        self.time_index = TimeIndexGet( self.horce_data )
        self.trainer_analyze = TrainerAnalyze( self.race_data, self.race_horce_data, self.trainer_data )
        self.jockey_analyze = JockeyAnalyze( self.race_data, self.race_horce_data, self.jockey_data )
        self.race_type = RaceType()
        self.before_race_score = BeforeRaceScore( self.race_data )
        self.blood_type_score = BloodTypeScore( self.race_data, self.horce_data )

        self.data_name_list = []
        self.write_data_list = []
        self.simu_data = {}
        self.kind_score_key_list = {}
        self.kind_score_key_list[data_name.waku_three_rate] = [ "place", "dist", "limb", "baba", "kind" ]
        self.kind_score_key_list[data_name.limb_score] = [ "place", "dist", "baba", "kind" ]
        self.result = { "answer": [], "teacher": [], "query": [], "year": [], \
                        "level": [], "diff": [], "popular": [], "category": {} }
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
        
        for data_name in name_list:
            if data_dict[data_name] == lib.escapeValue:
                result.append( math.nan )
            else:
                result.append( round( data_dict[data_name], 3 ) )

        if len( self.write_data_list ) == 0:
            self.write_data_list = copy.deepcopy( name_list )

        return result

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

        str_year = race_id[0:4]
        category_data = []
        key_race_money_class = str( int( lib.money_class_get( self.race_data.data["money"] ) ) )
        current_high_level = self.race_high_level.current_high_level( race_id )
        teacher_data = []
        answer_data = []
        popular_data = []
        diff_data = []
        horce_id_list = []
        race_limb = {}
        current_race_data = {}
        current_race_data[data_name.my_limb_count] = { str(lib.escapeValue): lib.escapeValue }
        
        for count, horce_id in enumerate( self.race_horce_data.horce_id_list ):
            current_data, past_data = lib.race_check( self.horce_data.data[horce_id]["past_data"], ymd )
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data, self.race_data )

            if not cd.race_check():
                continue

            cd.setting_odds( self.race_data.data["dev_odds_popular"][horce_id]["odds"] )
            cd.setting_popular( self.race_data.data["dev_odds_popular"][horce_id]["popular"] )
            place_num = int( key_place )
            horce_num = int( cd.horce_number() )

            t_instance = {}
            for name in self.predict_data.keys():
                if race_id in self.predict_data[name] and horce_id in self.predict_data[name][race_id]:
                    t_instance[name] = self.predict_data[name][race_id][horce_id]["score"]
                    t_instance[name+"_index"] = self.predict_data[name][race_id][horce_id]["index"]
                    t_instance[name+"_stand"] = self.predict_data[name][race_id][horce_id]["stand"]
                else:
                    t_instance[name] = lib.escapeValue
                    t_instance[name+"_index"] = lib.escapeValue
                    t_instance[name+"_stand"] = lib.escapeValue
                                    
            t_instance.update( lib.horce_teacher_analyze( current_race_data, t_instance, count ) )
            
            t_list = self.data_list_create( t_instance )

            if str_year in lib.test_years:
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
                                                               "popular_win_rate": popular_win_rate,
                                                               "new": cd.new_check() }

            answer_data.append( cd.rank() )
            teacher_data.append( t_list )
            diff_data.append( cd.diff() )
            popular_data.append( cd.popular() )

        if not len( answer_data ) == 0:
            self.result["answer"].append( answer_data )
            self.result["teacher"].append( teacher_data )
            self.result["year"].append( str_year )
            self.result["level"].append( [ current_high_level ] )
            self.result["query"].append( { "q": len( answer_data ), "year": str_year } )
            self.result["diff"].append( diff_data )
            self.result["popular"].append( popular_data )

        if len( self.result["category"] ) == 0:
            self.result["category"] = category_data
