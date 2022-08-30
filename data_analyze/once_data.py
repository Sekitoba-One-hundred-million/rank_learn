import math
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
        
        self.race_high_level = RaceHighLevel()
        self.race_type = RaceType()
        self.time_index = TimeIndexGet()
        self.trainer_data = TrainerData()
        self.jockey_data = JockeyData()
        self.before_data = BeforeData()
        #self.up_score_get = UpScore()
        self.train_index = TrainIndexGet()
        #self.pace_time_score = PaceTimeScore()

        self.data_name_list = []
        self.simu_data = {}
        self.result = { "answer": [], "teacher": [], "query": [], "year": [] }
        self.data_name_read()

    def data_name_read( self ):
        f = open( "learn_data.txt", "r" )
        str_data_list = f.readlines()

        for str_data in str_data_list:
            self.data_name_list.append( str_data.replace( "\n", "" ) )

    def data_list_create( self, data_dict ):
        result = []
        
        for data_name in self.data_name_list:
            try:
                result.append( data_dict[data_name] )
            except:
                continue

        return result

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
        ri_list = [ key_place + ":place", key_dist + ":dist", key_kind + ":kind", key_baba + ":baba" ]        
        info_key_dist = key_dist

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            return
        
        teacher_data = []
        answer_data = []
        year_data = []
        
        for kk in self.race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue
            
            limb_math = lib.limb_search( pd )
            horce_num = int( cd.horce_number() )
            
            try:
                omega_index_score = self.omega_index_data[race_id][horce_num-1]
            except:
                continue

            current_time_index = self.time_index.main( kk, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.baba_index_data[horce_id] )
            train_score = self.train_index.score_get( race_id, horce_num )
            
            t_instance = {}
            t_instance[data_name.id_weight] = cd.id_weight()
            t_instance[data_name.burden_weight] = cd.burden_weight()
            t_instance[data_name.horce_number] = horce_num
            #t_instance[data_name.stright_dist] = rci_dist[-1]
            t_instance[data_name.speed_index] = lib.max_check( speed )
            t_instance[data_name.time_index] = current_time_index["max"]
            t_instance[data_name.three_ave_rank] = pd.three_average()
            t_instance[data_name.one_rate] = pd.one_rate()
            t_instance[data_name.two_rate] = pd.two_rate()
            t_instance[data_name.three_rate] = pd.three_rate()
            t_instance[data_name.money] = pd.get_money()
            t_instance[data_name.race_interval] = pd.race_interval()
            t_instance[data_name.train_score] = train_score
            t_instance[data_name.limb] = limb_math

            t_list = self.data_list_create( t_instance )

            if year in lib.test_years:
                lib.dic_append( self.simu_data, race_id, {} )
                self.simu_data[race_id][horce_id] = {}
                self.simu_data[race_id][horce_id]["answer"] = { "rank": cd.rank(), "odds": cd.odds(), "popular": cd.popular() }
                self.simu_data[race_id][horce_id]["data"] = t_list

            rank = cd.rank()
            answer_data.append( rank )
            teacher_data.append( t_list )
            year_data.append( year )

        if not len( answer_data ) == 0:
            self.result["answer"].extend( answer_data )
            self.result["teacher"].extend( teacher_data )
            self.result["year"].extend( year_data )
            self.result["query"].append( { "q": len( answer_data ), "year": year } )
