import os
import numpy as np
from tqdm import tqdm

import sekitoba_library as lib
import sekitoba_data_manage as dm
from sekitoba_data_create.time_index_get import TimeIndexGet

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "baba_index_data.pickle" )

name = "speed_index"
RANK = "rank"
COUNT = "count"
    
def main():
    result = {}
    data_storage = []
    race_data = dm.dl.data_get( "race_data.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    baba_index_data = dm.dl.data_get( "baba_index_data.pickle" )
    time_index = TimeIndexGet()
    
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

        #if year in lib.test_years:
        #    continue

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            continue

        count = 0
        speed_index_list = []
        up_speed_index_list = []
        pace_speed_index_list = []
        
        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            current_time_index = time_index.main( horce_id, pd.past_day_list() )
            speed, up_speed, pace_speed = pd.speed_index( baba_index_data[horce_id] )
            #speed_index_list.append( lib.max_check( speed ) + lib.max_check( up_speed ) + lib.max_check( pace_speed ) + current_time_index["max"] )
            speed_index_list.append( lib.max_check( speed ) + current_time_index["max"] )
            #up_speed_index_list.append( lib.max_check( up_speed ) )
            #pace_speed_index_list.append( lib.max_check( pace_speed ) )

        sort_speed_index = sorted( speed_index_list, reverse = True )
        
        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            score = sort_speed_index.index( speed_index_list[count] )
            key = str( int( score ) )
            
            lib.dic_append( result, year, {} )
            lib.dic_append( result[year], key, { RANK: 0, COUNT: 0 } )
            
            result[year][key][COUNT] += 1
            result[year][key][RANK] += cd.rank()

    for year in result.keys():
        for k in result[year].keys():
            result[year][k][RANK] /= result[year][k][COUNT]
            result[year][k][RANK] = round( result[year][k][RANK], 2 )

    lib.write_rank_csv( result, name + ".csv" )
 

if __name__ == "__main__":
    main()
        