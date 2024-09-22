import SekitobaLibrary as lib
import SekitobaDataManage as dm

from tqdm import tqdm
import matplotlib.pyplot as plt

dm.dl.file_set( "race_data.pickle" )
dm.dl.file_set( "race_info_data.pickle" )
dm.dl.file_set( "horce_data_storage.pickle" )
dm.dl.file_set( "track_bias_data.pickle" )

name = "age"
RANK = "rank"
COUNT = "count"

def main():
    result = { "one": 0, "two": 0, "three": 0, "count": 0 }
    race_data = dm.dl.data_get( "race_data.pickle" )
    race_info = dm.dl.data_get( "race_info_data.pickle" )
    horce_data = dm.dl.data_get( "horce_data_storage.pickle" )
    track_bias_data = dm.dl.data_get( "track_bias_data.pickle" )
    split_horce_num = 5
    
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

        if not race_id in track_bias_data:
            continue
        
        data_list = []

        for kk in race_data[k].keys():
            horce_id = kk
            current_data, past_data = lib.race_check( horce_data[horce_id],
                                                     year, day, num, race_place_num )#今回と過去のデータに分ける
            cd = lib.current_data( current_data )
            pd = lib.past_data( past_data, current_data )

            if not cd.race_check():
                continue

            position_key = min( int( cd.horce_number() / split_horce_num ), 2 )
            max_time_point = pd.max_time_point()
            max_time_point += track_bias_data[race_id][position_key]["one"] + track_bias_data[race_id][position_key]["two"] + track_bias_data[race_id][position_key]["three"]
            max_time_point += track_bias_data[race_id][position_key]["popular_rank"]
            data_list.append( { "max_time_point": max_time_point, "rank": cd.rank() } )

        if len( data_list ) == 0:
            continue

        data_list = sorted( data_list, key=lambda x:x["max_time_point"], reverse = True )
        result["count"] += 1

        if data_list[0]["rank"] == 1:
            result["one"] += 1
            result["two"] += 1
            result["three"] += 1
        elif data_list[0]["rank"] == 2:
            result["two"] += 1
            result["three"] += 1
        elif data_list[0]["rank"] == 3:
            result["three"] += 1

    result["one"] /= result["count"]
    result["two"] /= result["count"]
    result["three"] /= result["count"]

    print( "one: {}".format( result["one"] * 100 ) )
    print( "two: {}".format( result["two"] * 100 ) )
    print( "three: {}".format( result["three"] * 100 ) )
            
if __name__ == "__main__":
    main()
        
