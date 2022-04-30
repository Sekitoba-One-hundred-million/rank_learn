import math
import sklearn
from tqdm import tqdm
from mpi4py import MPI

import sekitoba_library as lib
import sekitoba_data_manage as dm
import sekitoba_data_create as dc

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
#dm.dl.file_set( "race_cource_wrap.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "race_cource_info.pickle" )
dm.dl.file_set( "corner_horce_body.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )
dm.dl.file_set( "parent_id_data.pickle" )
dm.dl.file_set( "win_rate_data.pickle" )
dm.dl.file_set( "last_horce_body.pickle" )
dm.dl.file_set( "rank_learn_encoding.pickle" )
dm.dl.file_set( "one_rate_score.pickle" )
dm.dl.file_set( "diff_score.pickle" )
#dm.dl.file_set( "first_up3_halon.pickle" ) 

class OnceData:
    def __init__( self ):
        self.race_data = dm.dl.data_get( "race_data.pickle" )
        self.horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
        self.race_info = dm.dl.data_get( "race_info_data.pickle" )
        self.race_cource_info = dm.dl.data_get( "race_cource_info.pickle" )
        self.corner_horce_body = dm.dl.data_get( "corner_horce_body.pickle" )
        self.baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
        self.parent_id_data = dm.dl.data_get( "parent_id_data.pickle" )
        self.win_rate_data = dm.pickle_load( "win_rate_data.pickle" )
        self.last_horce_body_data = dm.dl.data_get( "last_horce_body.pickle" )
        self.rank_learn_encoding = dm.dl.data_get( "rank_learn_encoding.pickle" )
        self.one_rate_score = dm.dl.data_get( "one_rate_score.pickle" )
        self.diff_score = dm.dl.data_get( "diff_score.pickle" )
        self.train_index = dc.TrainIndexGet()
        self.time_index = dc.TimeIndexGet()
        self.jockey_data = dc.JockeyData()
        self.up_score = dc.UpScore()
        self.past_horce_body = dc.PastHorceBody()
        self.pace_time_score = dc.PaceTimeScore()
        self.slow_start = dc.SlowStart()
        self.simu_data = {}
        self.result = { "answer": [], "teacher": [], "query": [], "year": [] }

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
        
        if self.race_info[race_id]["out_side"]:
            info_key_dist += "外"

        try:
            rci_dist = self.race_cource_info[key_place][key_kind][info_key_dist]["dist"]
            rci_info = self.race_cource_info[key_place][key_kind][info_key_dist]["info"]
        except:
            return
        
        race_limb = [0] * 9
        popular_limb = -1
        data_list = {}
        data_list["speed"] = []
        data_list["up_speed"] = []
        data_list["pace_speed"] = []
        data_list["time_index"] = []
        data_list["average_speed"] = []

        for kk in self.race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )
            
            if not cd.race_check():
                continue

            current_time_index = self.time_index.main( kk, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( self.baba_index_data[horce_id] )
            data_list["speed"].append( lib.max_check( speed ) )
            data_list["up_speed"].append( lib.max_check( up_speed ) )
            data_list["pace_speed"].append( lib.max_check( pace_speed ) )      
            data_list["time_index"].append( current_time_index["max"] )
            data_list["average_speed"].append( pd.average_speed() )
            
            try:
                limb_math = lib.limb_search( pd )
            except:
                limb_math = 0
                
            race_limb[limb_math] += 1

        data_list["stand_speed"] = lib.speed_standardization( data_list["speed"] )
        data_list["stand_up_speed"] = lib.speed_standardization( data_list["up_speed"] )
        data_list["stand_pace_speed"] = lib.speed_standardization( data_list["pace_speed"] )
        data_list["stand_time_index"] = lib.speed_standardization( data_list["time_index"] )
        data_list["stand_average_speed"] = lib.speed_standardization( data_list["average_speed"] )
        count = -1
        co = 0
        
        for kk in self.race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( self.horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue
            
            current_jockey = self.jockey_data.data_get( horce_id, cd.birthday(), cd.race_num() )
            before_horce_body = 0
            count += 1

            stand_speed = data_list["stand_speed"][count]
            stand_up_speed = data_list["stand_up_speed"][count]
            stand_pace_speed = data_list["stand_pace_speed"][count]
            stand_time_index = data_list["stand_time_index"][count]            
            stand_average_speed = data_list["stand_average_speed"][count]
            
            t_instance = []
            
            try:
                limb_math = lib.limb_search( pd )
            except:
                limb_math = 0

            key_horce_num = str( int( cd.horce_number() ) )
            key_limb = str( int( limb_math ) )
            key_burden = str( int( cd.burden_weight() ))
            """
            if not year == lib.test_year:
                try:
                    last_horce_body = self.corner_horce_body[race_id]["4"][key_horce_num]
                except:
                    co += 1
                    continue
            else:
                try:
                    last_horce_body = self.last_horce_body_data[race_id][key_horce_num]
                except:
                    co += 1
                    continue
            """
            father_id = self.parent_id_data[horce_id]["father"]
            mother_id = self.parent_id_data[horce_id]["mother"]
            father_data = dc.parent_data_get.main( self.horce_data, father_id, self.baba_index_data )
            mother_data = dc.parent_data_get.main( self.horce_data, mother_id, self.baba_index_data )
            current_train = self.train_index.main( race_id, key_horce_num )
            #dm.dn.append( t_instance, race_limb[0], "その他の馬の数" )
            #dm.dn.append( t_instance, race_limb[1], "逃げaの馬の数" )
            #dm.dn.append( t_instance, race_limb[2], "逃げbの馬の数" )
            dm.dn.append( t_instance, race_limb[3], "先行aの馬の数" )
            dm.dn.append( t_instance, race_limb[4], "先行bの馬の数" )
            #dm.dn.append( t_instance, race_limb[5], "差しaの馬の数" )
            #dm.dn.append( t_instance, race_limb[6], "差しbの馬の数" )
            dm.dn.append( t_instance, race_limb[7], "追いの馬の数" )
            #dm.dn.append( t_instance, race_limb[8], "後方の馬の数" )
            #dm.dn.append( t_instance, self.rank_learn_encoding["place_limb"]["one"][key_place][key_limb], "場所one" )
            #dm.dn.append( t_instance, self.rank_learn_encoding["place_limb"]["two"][key_place][key_limb], "場所two" )
            #dm.dn.append( t_instance, self.rank_learn_encoding["place_limb"]["three"][key_place][key_limb], "場所three" )
            dm.dn.append( t_instance, self.rank_learn_encoding["dist_limb"]["one"][key_dist][key_limb], "距離one" )
            dm.dn.append( t_instance, self.rank_learn_encoding["dist_limb"]["two"][key_dist][key_limb], "距離two" )
            dm.dn.append( t_instance, self.rank_learn_encoding["dist_limb"]["three"][key_dist][key_limb], "距離three" )
            dm.dn.append( t_instance, self.rank_learn_encoding["kind_limb"]["one"][key_kind][key_limb], "芝かダートone" )
            dm.dn.append( t_instance, self.rank_learn_encoding["kind_limb"]["two"][key_kind][key_limb], "芝かダートtwo" )
            dm.dn.append( t_instance, self.rank_learn_encoding["kind_limb"]["three"][key_kind][key_limb], "芝かダートthree" )
            dm.dn.append( t_instance, self.rank_learn_encoding["baba_limb"]["one"][key_baba][key_limb], "馬場one" )
            dm.dn.append( t_instance, self.rank_learn_encoding["baba_limb"]["two"][key_baba][key_limb], "馬場two" )
            #dm.dn.append( t_instance, self.rank_learn_encoding["baba_limb"]["three"][key_baba][key_limb], "馬場three" )
            #dm.dn.append( t_instance, cd.id_weight(), "馬体重の増減" )
            #dm.dn.append( t_instance, self.rank_learn_encoding["burden"][key_burden], "斤量" )
            #dm.dn.append( t_instance, self.rank_learn_encoding["horce_number"][key_horce_num], "馬番" )
            #dm.dn.append( t_instance, float( key_dist ) - rci_dist[-1], "今まで走った距離" )
            dm.dn.append( t_instance, rci_dist[-1], "直線の距離" )
            dm.dn.append( t_instance, self.rank_learn_encoding["limb"]["one"][key_limb], "過去データからの予想脚質one" )
            #dm.dn.append( t_instance, self.rank_learn_encoding["limb"]["two"][key_limb], "過去データからの予想脚質two" )
            dm.dn.append( t_instance, self.rank_learn_encoding["limb"]["three"][key_limb], "過去データからの予想脚質three" )
            dm.dn.append( t_instance, data_list["speed"][count-1], "スピード指数" )
            #dm.dn.append( t_instance, data_list["speed"][count-1] - max( data_list["speed"] ), "max_diffスピード指数" )
            dm.dn.append( t_instance, stand_speed, "standスピード指数" )
            #dm.dn.append( t_instance, stand_speed - max( data_list["stand_speed"] ), "stand_max_diffスピード指数" )
            
            dm.dn.append( t_instance, data_list["up_speed"][count-1], "上り指数" )
            #dm.dn.append( t_instance, data_list["up_speed"][count-1] - max( data_list["up_speed"] ), "max_diff上り指数" )
            dm.dn.append( t_instance, stand_up_speed, "stand上り指数" )
            #dm.dn.append( t_instance, stand_up_speed - max( data_list["stand_up_speed"] ), "stand_max_diff上り指数" )
            
            dm.dn.append( t_instance, data_list["pace_speed"][count-1], "ペース指数" )
            #dm.dn.append( t_instance, data_list["pace_speed"][count-1] - max( data_list["pace_speed"] ), "max_diffペース指数" )
            dm.dn.append( t_instance, stand_pace_speed , "standペース指数" )
            #dm.dn.append( t_instance, stand_pace_speed - max( data_list["stand_pace_speed"] ), "stand_max_diffペース指数" )
            
            dm.dn.append( t_instance, data_list["time_index"][count], "タイム指数" )
            #dm.dn.append( t_instance, data_list["time_index"][count-1] - max( data_list["time_index"] ), "max_diffタイム指数" )
            dm.dn.append( t_instance, stand_time_index, "standタイム指数" )
            #dm.dn.append( t_instance, stand_time_index - max( data_list["stand_time_index"] ), "stand_max_diffタイム指数" )
            #dm.dn.append( t_instance, pd.three_average(), "過去3レースの平均順位" )
            #dm.dn.append( t_instance, pd.dist_rank_average(), "過去同じ距離の種類での平均順位" )
            #dm.dn.append( t_instance, pd.racekind_rank_average(), "過去同じレース状況での平均順位" )
            #dm.dn.append( t_instance, pd.baba_rank_average(), "過去同じ馬場状態での平均順位" )
            #dm.dn.append( t_instance, pd.jockey_rank_average(), "過去同じ騎手での平均順位" )
            #dm.dn.append( t_instance, pd.three_average(), "複勝率" )
            #dm.dn.append( t_instance, pd.two_rate(), "連対率" )
            #dm.dn.append( t_instance, pd.get_money(), "獲得賞金" )
            dm.dn.append( t_instance, pd.best_weight(), "ベスト体重と現在の体重の差" )
            dm.dn.append( t_instance, pd.race_interval(), "中週" )
            #dm.dn.append( t_instance, stand_average_speed, "平均速度" )
            #dm.dn.append( t_instance, pd.pace_up_check(), "ペースと上りの関係" )
            dm.dn.append( t_instance, current_train["score"] ,"調教score" )
            #dm.dn.append( t_instance, current_train["a"], "調教ペースの傾き" )
            #dm.dn.append( t_instance, current_train["b"], "調教ペースの切片" )
            dm.dn.append( t_instance, father_data["rank"], "父親の平均順位" )
            #dm.dn.append( t_instance, father_data["two_rate"], "父親の連対率" )
            #dm.dn.append( t_instance, father_data["three_rate"], "父親の副賞率" )
            dm.dn.append( t_instance, father_data["average_speed"], "父親の平均速度" )
            #dm.dn.append( t_instance, father_data["speed_index"], "父親の最大のスピード指数" )
            #dm.dn.append( t_instance, father_data["up_speed_index"], "父親の最大の上りスピード指数" )
            #dm.dn.append( t_instance, father_data["pace_speed_index"], "父親の最大のペース指数" )
            dm.dn.append( t_instance, father_data["limb"], "父親の脚質" )
            #dm.dn.append( t_instance, mother_data["rank"], "母親の平均順位" )
            dm.dn.append( t_instance, mother_data["two_rate"], "母親の連対率" )
            #dm.dn.append( t_instance, mother_data["three_rate"], "母親の副賞率" )
            #dm.dn.append( t_instance, mother_data["average_speed"], "母親の平均速度" )
            dm.dn.append( t_instance, mother_data["speed_index"], "母親の最大のスピード指数" )
            dm.dn.append( t_instance, mother_data["up_speed_index"], "母親の最大の上りスピード指数" )
            #dm.dn.append( t_instance, mother_data["pace_speed_index"], "母親の最大のペース指数" )
            #dm.dn.append( t_instance, mother_data["limb"], "母親の脚質" )
            dm.dn.append( t_instance, current_jockey["all"]["rank"], "騎手の過去の平均順位" )
            #dm.dn.append( t_instance, current_jockey["all"]["one"], "騎手の過去のone" )
            dm.dn.append( t_instance, current_jockey["all"]["two"], "騎手の過去のtwo" )
            dm.dn.append( t_instance, current_jockey["all"]["three"], "騎手の過去のthree" )
            #dm.dn.append( t_instance, current_jockey["all"]["slow"], "騎手の過去ののslow" )
            #dm.dn.append( t_instance, current_jockey["all"]["time"], "騎手の過去のタイム" )
            dm.dn.append( t_instance, current_jockey["all"]["up"], "騎手の過去の上り" )
            dm.dn.append( t_instance, current_jockey["100"]["rank"], "騎手の過去の100の平均順位" )
            #dm.dn.append( t_instance, current_jockey["100"]["one"], "騎手の過去の100のone" )
            #dm.dn.append( t_instance, current_jockey["100"]["two"], "騎手の過去の100のtwo" )
            #dm.dn.append( t_instance, current_jockey["100"]["three"], "騎手の過去の100のthree" )
            dm.dn.append( t_instance, current_jockey["100"]["slow"], "騎手の過去の100のslow" )
            dm.dn.append( t_instance, current_jockey["100"]["time"], "騎手の過去の100のタイム" )
            dm.dn.append( t_instance, current_jockey["100"]["up"], "騎手の過去の100の上り" )            
            dm.dn.append( t_instance, self.up_score.score_get( pd ), "up_score" )
            dm.dn.append( t_instance, self.pace_time_score.score_get( pd ), "pace_time_score" )
            dm.dn.append( t_instance, self.slow_start.main( horce_id, pd ), "slow" )
            #dm.dn.append( t_instance, last_horce_body, "前の馬身" )
            #dm.dn.append( t_instance, self.one_rate_score[race_id][key_horce_num]["score"], "one_rate_score" )
            #dm.dn.append( t_instance, self.diff_score[race_id][key_horce_num]["score"], "diff_score" )

            if year in lib.test_years:
                lib.dic_append( self.simu_data, race_id, {} )
                self.simu_data[race_id][key_horce_num] = {}
                self.simu_data[race_id][key_horce_num]["answer"] = { "rank": cd.rank(), "odds": cd.odds(), "popular": cd.popular() }
                self.simu_data[race_id][key_horce_num]["data"] = t_instance

            a_instance = [0] * 20
            rank = cd.rank()
            
            for r in range( 0, len( a_instance ) ):
                a_instance[r] = math.pow( 0.5, int( abs( rank - r ) ) ) * 2
            
            self.result["answer"].append( rank )
            self.result["teacher"].append( t_instance )
            self.result["year"].append( year )

        if not count + 1 == 0:
            self.result["query"].append( { "q": count - co + 1, "year": year } )
