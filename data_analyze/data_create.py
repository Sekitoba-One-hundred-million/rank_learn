import math
from tqdm import tqdm

import sekitoba_library as lib
import sekitoba_data_manage as dm
from data_analyze.train_index_get import train_index_get
from data_analyze.time_index_get import time_index_get
from data_analyze.jockey_data_get import JockeyData
from data_analyze import parent_data_get

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
#dm.dl.file_set( "first_up3_halon.pickle" ) 

def speed_standardization( data ):
    result = []
    ave = 0
    conv = 0
    count = 0

    for d in data:
        if d < 0:
            continue
        
        ave += d
        count += 1

    if count == 0:
        return [0] * len( data )

    ave /= count    

    for d in data:
        if d < 0:
            continue

        conv += math.pow( d - ave, 2 )

    conv /= count
    conv = math.sqrt( conv )

    if conv == 0:
        return [0] * len( data )
    
    for d in data:
        if d < 0:
            result.append( 0 )
        else:
            result.append( ( d - ave ) / conv )

    return result

def max_check( data ):
    try:
        return max( data )
    except:
        return -1    

def main( update = False ):
    result = None
    
    if not update:
        result = dm.pickle_load( "rank_learn_data.pickle" )
        simu_data = dm.pickle_load( "rank_simu_data.pickle" )

    if result == None:
        result = {}
        simu_data = {}        
    else:
        return result, simu_data

    result["answer"] = []
    result["answer_diff"] = []    
    result["answer_rank"] = []
    result["answer_list"] = []
    result["teacher"] = []
    result["year"] = []
    result["query"] = []
    min_horce_body = 10000
    max_horce_body = -1

    race_data = dm.dl.data_get( "race_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    #race_cource_wrap = dm.dl.data_get( "race_cource_wrap.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    race_cource_info = dm.dl.data_get( "race_cource_info.pickle" )
    corner_horce_body = dm.dl.data_get( "corner_horce_body.pickle" )
    baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
    parent_id_data = dm.dl.data_get( "parent_id_data.pickle" )
    win_rate_data = dm.pickle_load( "win_rate_data.pickle" )
    last_horce_body_data = dm.dl.data_get( "last_horce_body.pickle" )
    rank_learn_encoding = dm.dl.data_get( "rank_learn_encoding.pickle" )
    train_index = train_index_get()
    time_index = time_index_get()
    jockey_data = JockeyData()

    for k in tqdm( race_data.keys() ):
        race_id = lib.id_get( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]

        key_place = str( race_info[race_id]["place"] )
        key_dist = str( race_info[race_id]["dist"] )
        key_kind = str( race_info[race_id]["kind"] )        
        key_baba = str( race_info[race_id]["baba"] )
        ri_list = [ key_place + ":place", key_dist + ":dist", key_kind + ":kind", key_baba + ":baba" ]        
        info_key_dist = key_dist
        
        if race_info[race_id]["out_side"]:
            info_key_dist += "外"

        try:
            rci_dist = race_cource_info[key_place][key_kind][info_key_dist]["dist"]
            rci_info = race_cource_info[key_place][key_kind][info_key_dist]["info"]
        except:
            continue
        
        race_limb = [0] * 9
        popular_limb = -1
        data_list = {}
        data_list["speed"] = []
        data_list["up_speed"] = []
        data_list["pace_speed"] = []
        data_list["time_index"] = []
        data_list["average_speed"] = []
        train_index_list = train_index.main( race_data[k], horce_data, race_id )
        count = -1

        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )
            
            if not cd.race_check():
                continue

            current_time_index = time_index.main( kk, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( baba_index_data[horce_id] )
            data_list["speed"].append( max_check( speed ) )
            data_list["up_speed"].append( max_check( up_speed ) )
            data_list["pace_speed"].append( max_check( pace_speed ) )            
            data_list["time_index"].append( current_time_index["max"] )
            data_list["average_speed"].append( pd.average_speed() )
            
            try:
                limb_math = lib.limb_search( pd )
            except:
                limb_math = 0
                
            race_limb[limb_math] += 1

        data_list["stand_speed"] = speed_standardization( data_list["speed"] )
        data_list["stand_up_speed"] = speed_standardization( data_list["up_speed"] )
        data_list["stand_pace_speed"] = speed_standardization( data_list["pace_speed"] )
        data_list["stand_time_index"] = speed_standardization( data_list["time_index"] )
        data_list["stand_average_speed"] = speed_standardization( data_list["average_speed"] )
        
        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            #pad = parent_data.main( horce_name, cd )
            current_jockey = jockey_data.data_get( horce_id, cd.birthday(), cd.race_num() )
            before_horce_body = 0
            count += 1

            stand_speed = data_list["stand_speed"][count-1]
            stand_up_speed = data_list["stand_up_speed"][count-1]
            stand_pace_speed = data_list["stand_pace_speed"][count-1]
            stand_time_index = data_list["stand_time_index"][count-1]            
            stand_average_speed = data_list["stand_average_speed"][count-1]
            
            t_instance = []
            
            try:
                limb_math = lib.limb_search( pd )
            except:
                limb_math = 0

            key_horce_num = str( int( cd.horce_number() ) )
            key_limb = str( int( limb_math ) )
            key_burden = str( int( cd.burden_weight() ))

            try:
                last_horce_body = corner_horce_body[race_id]["4"][key_horce_num]
            except:
                last_horce_body = -1

            father_id = parent_id_data[horce_id]["father"]
            mother_id = parent_id_data[horce_id]["mother"]
            father_data = parent_data_get.main( horce_data, father_id, baba_index_data )
            mother_data = parent_data_get.main( horce_data, mother_id, baba_index_data )
            #dm.dn.append( t_instance, race_limb[0], "その他の馬の数" )
            dm.dn.append( t_instance, race_limb[1], "逃げaの馬の数" )
            #dm.dn.append( t_instance, race_limb[2], "逃げbの馬の数" )
            #dm.dn.append( t_instance, race_limb[3], "先行aの馬の数" )
            dm.dn.append( t_instance, race_limb[4], "先行bの馬の数" )
            dm.dn.append( t_instance, race_limb[5], "差しaの馬の数" )
            #dm.dn.append( t_instance, race_limb[6], "差しbの馬の数" )
            #dm.dn.append( t_instance, race_limb[7], "追いの馬の数" )
            #dm.dn.append( t_instance, race_limb[8], "後方の馬の数" )
            dm.dn.append( t_instance, rank_learn_encoding["place_limb"][key_place][key_limb], "場所" )
            dm.dn.append( t_instance, rank_learn_encoding["dist_limb"][key_dist][key_limb] , "距離" )
            #dm.dn.append( t_instance, rank_learn_encoding["kind_limb"][key_kind][key_limb], "芝かダート" )
            #dm.dn.append( t_instance, rank_learn_encoding["baba_limb"][key_baba][key_limb], "馬場" )
            #dm.dn.append( t_instance, cd.id_weight(), "馬体重の増減" )
            dm.dn.append( t_instance, rank_learn_encoding["burden"][key_burden], "斤量" )
            dm.dn.append( t_instance, rank_learn_encoding["horce_number"][key_horce_num], "馬番" )
            #dm.dn.append( t_instance, float( key_dist ) - rci_dist[-1], "今まで走った距離" )
            #dm.dn.append( t_instance, rci_dist[-1], "直線の距離" )
            #dm.dn.append( t_instance, rank_learn_encoding["limb"][key_limb], "過去データからの予想脚質" )

            #dm.dn.append( t_instance, data_list["speed"][count-1], "スピード指数" )
            #dm.dn.append( t_instance, data_list["speed"][count-1] - max( data_list["speed"] ), "max_diffスピード指数" )
            dm.dn.append( t_instance, stand_speed, "standスピード指数" )
            #dm.dn.append( t_instance, stand_speed - max( data_list["stand_speed"] ), "stand_max_diffスピード指数" )
            
            #m.dn.append( t_instance, data_list["up_speed"][count-1], "上り指数" )
            #m.dn.append( t_instance, data_list["up_speed"][count-1] - max( data_list["up_speed"] ), "max_diff上り指数" )
            dm.dn.append( t_instance, stand_up_speed, "stand上り指数" )
            dm.dn.append( t_instance, stand_up_speed - max( data_list["stand_up_speed"] ), "stand_max_diff上り指数" )
            
            #dm.dn.append( t_instance, data_list["pace_speed"][count-1], "ペース指数" )
            dm.dn.append( t_instance, data_list["pace_speed"][count-1] - max( data_list["pace_speed"] ), "max_diffペース指数" )
            #dm.dn.append( t_instance, stand_pace_speed , "standペース指数" )
            dm.dn.append( t_instance, stand_pace_speed - max( data_list["stand_pace_speed"] ), "stand_max_diffペース指数" )
            
            #dm.dn.append( t_instance, data_list["time_index"][count-1], "タイム指数" )
            #dm.dn.append( t_instance, data_list["time_index"][count-1] - max( data_list["time_index"] ), "max_diffタイム指数" )
            #dm.dn.append( t_instance, stand_time_index, "standタイム指数" )
            dm.dn.append( t_instance, stand_time_index - max( data_list["stand_time_index"] ), "stand_max_diffタイム指数" )
            #dm.dn.append( t_instance, pd.three_average(), "過去3レースの平均順位" )
            #dm.dn.append( t_instance, pd.dist_rank_average(), "過去同じ距離の種類での平均順位" )
            #dm.dn.append( t_instance, pd.racekind_rank_average(), "過去同じレース状況での平均順位" )
            #dm.dn.append( t_instance, pd.baba_rank_average(), "過去同じ馬場状態での平均順位" )
            #dm.dn.append( t_instance, pd.jockey_rank_average(), "過去同じ騎手での平均順位" )
            #dm.dn.append( t_instance, pd.three_average(), "複勝率" )
            #dm.dn.append( t_instance, pd.two_rate(), "連対率" )
            #dm.dn.append( t_instance, pd.get_money(), "獲得賞金" )
            dm.dn.append( t_instance, pd.best_weight(), "ベスト体重と現在の体重の差" )
            #dm.dn.append( t_instance, pd.race_interval(), "中週" )
            #dm.dn.append( t_instance, stand_average_speed, "平均速度" )
            dm.dn.append( t_instance, pd.pace_up_check(), "ペースと上りの関係" )
            #dm.dn.append( t_instance, train_index_list[key_horce_num]["a"], "調教ペースの傾き" )
            dm.dn.append( t_instance, train_index_list[key_horce_num]["b"], "調教ペースの切片" )
            #dm.dn.append( t_instance, train_index_list[count]["time"], "調教ペースの指数タイム" )
            #dm.dn.append( t_instance, father_data["rank"], "父親の平均順位" )
            #dm.dn.append( t_instance, father_data["two_rate"], "父親の連対率" )
            #dm.dn.append( t_instance, father_data["three_rate"], "父親の副賞率" )
            dm.dn.append( t_instance, father_data["average_speed"], "父親の平均速度" )
            #dm.dn.append( t_instance, father_data["speed_index"], "父親の最大のスピード指数" )
            #dm.dn.append( t_instance, father_data["up_speed_index"], "父親の最大の上りスピード指数" )
            dm.dn.append( t_instance, father_data["pace_speed_index"], "父親の最大のペース指数" )
            #dm.dn.append( t_instance, father_data["limb"], "父親の脚質" )
            #dm.dn.append( t_instance, mother_data["rank"], "母親の平均順位" )
            #dm.dn.append( t_instance, mother_data["two_rate"], "母親の連対率" )
            #dm.dn.append( t_instance, mother_data["three_rate"], "母親の副賞率" )
            #dm.dn.append( t_instance, mother_data["average_speed"], "母親の平均速度" )
            #dm.dn.append( t_instance, mother_data["speed_index"], "母親の最大のスピード指数" )
            #dm.dn.append( t_instance, mother_data["up_speed_index"], "母親の最大の上りスピード指数" )
            dm.dn.append( t_instance, mother_data["pace_speed_index"], "母親の最大のペース指数" )
            #dm.dn.append( t_instance, mother_data["limb"], "母親の脚質" )
            #dm.dn.append( t_instance, current_jockey["all"]["rank"], "騎手の過去の平均順位" )
            #dm.dn.append( t_instance, current_jockey["all"]["one"], "騎手の過去のone" )
            #dm.dn.append( t_instance, current_jockey["all"]["two"], "騎手の過去のtwo" )
            #dm.dn.append( t_instance, current_jockey["all"]["three"], "騎手の過去のthree" )
            #dm.dn.append( t_instance, current_jockey["all"]["time"], "騎手の過去のタイム" )
            #dm.dn.append( t_instance, current_jockey["all"]["up"], "騎手の過去の上り" )
            #dm.dn.append( t_instance, current_jockey["100"]["rank"], "騎手の過去の100の平均順位" )
            #dm.dn.append( t_instance, current_jockey["100"]["one"], "騎手の過去の100のone" )
            #dm.dn.append( t_instance, current_jockey["100"]["two"], "騎手の過去の100のtwo" )
            #dm.dn.append( t_instance, current_jockey["100"]["three"], "騎手の過去の100のthree" )
            #dm.dn.append( t_instance, current_jockey["100"]["time"], "騎手の過去の100のタイム" )
            dm.dn.append( t_instance, current_jockey["100"]["up"], "騎手の過去の100の上り" )
            
            #win_rate_append( t_instance, win_rate_data, ri_list, key_data )
            
            """
            if not year == lib.test_year:
                #dm.dn.append( t_instance, first_horce_body, "最初の馬身" )
                dm.dn.append( t_instance, last_horce_body, "前の馬身" )
            else:
                #dm.dn.append( t_instance, first_horce_body, "最初の馬身" )
                dm.dn.append( t_instance, last_horce_body_data[race_id][key_horce_num], "前の馬身" )
            """
            
            if year == lib.test_year:
                lib.dic_append( simu_data, race_id, {} )
                simu_data[race_id][key_horce_num] = {}
                simu_data[race_id][key_horce_num]["answer"] = { "rank": cd.rank(), "odds": cd.odds() }
                simu_data[race_id][key_horce_num]["data"] = t_instance
                #simu_data[race_id][key_horce_num]["change"] = change_data

            a_instance = [0] * 20
            rank = cd.rank()
            
            for r in range( 0, len( a_instance ) ):
                a_instance[r] = math.pow( 0.5, int( abs( rank - r ) ) ) * 2
            
            result["answer"].append( rank )
            result["answer_diff"].append( cd.diff() )
            result["answer_list"].append( a_instance )
            result["answer_rank"].append( rank )
            result["teacher"].append( t_instance )
            result["year"].append( year )

        if not count + 1 == 0:
            result["query"].append( { "q": count + 1, "year": year } )
            
    for i in range( 0, len( result["answer"] ) ):
        result["answer"][i] = min( max( int( result["answer"][i] ), 0 ), 20 )

    print( len( result["answer"] ) , len( result["teacher"] ) )
    dm.dn.write( "last_staight_momo.txt" )
    dm.pickle_upload( "rank_learn_data.pickle", result )
    dm.pickle_upload( "rank_simu_data.pickle", simu_data )
    dm.dl.data_clear()
    
    return result, simu_data

