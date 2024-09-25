import SekitobaLibrary as lib
import SekitobaDataManage as dm
from SekitobaDataCreate.high_level_data_get import RaceHighLevel

from tqdm import tqdm
import matplotlib.pyplot as plt

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "race_level_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "race_level_split_data.pickle" )

name = "race_level_check"
RANK = "rank"
COUNT = "count"

def main():
    result = {}
    race_high_level = RaceHighLevel()
    race_data = dm.dl.data_get( "race_data.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    race_day = dm.dl.data_get( "race_day.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )

    for k in tqdm( race_data.keys() ):
        race_id = lib.idGet( k )
        year = race_id[0:4]
        race_place_num = race_id[4:6]
        day = race_id[9]
        num = race_id[7]
        ymd = { "y": int( year ), "m": race_day[race_id]["month"], "d": race_day[race_id]["day"] }
        
        key_place = str( race_info[race_id]["place"] )
        key_dist = str( race_info[race_id]["dist"] )
        key_kind = str( race_info[race_id]["kind"] )        
        key_baba = str( race_info[race_id]["baba"] )

        #if year in lib.test_years:
        #    continue

        #芝かダートのみ
        if key_kind == "0" or key_kind == "3":
            continue

        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.raceCheck( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.CurrentData( current_data )
            pd = lib.PastData( past_data, current_data )

            if not cd.raceCheck():
                continue

            score = race_high_level.data_get( cd, pd, ymd )

            if score == 1000:
                continue
            
            score = int( score )
            key = str( int( score ) )
            
            lib.dicAppend( result, year, {} )
            lib.dicAppend( result[year], key, { RANK: 0, COUNT: 0 } )
            
            result[year][key][COUNT] += 1
            result[year][key][RANK] += cd.rank()

    for year in result.keys():
        for k in result[year].keys():
            result[year][k][RANK] /= result[year][k][COUNT]
            result[year][k][RANK] = round( result[year][k][RANK], 2 )

    lib.write_rank_csv( result, name + ".csv" )
    
if __name__ == "__main__":
    main()
        
