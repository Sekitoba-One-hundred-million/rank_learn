import SekitobaDataManage as dm
import SekitobaLibrary as lib
import SekitobaPsql as ps

from SekitobaDataCreate.last_wrap import LastWrap

race_id = "202406030111"
race_data = ps.RaceData()
horce_data = ps.HorceData()
race_horce_data = ps.RaceHorceData()
race_data.get_all_data( race_id )
race_horce_data.get_all_data( race_id )
horce_data.get_multi_data( race_horce_data.horce_id_list )
last_wrap = LastWrap( race_data, horce_data, race_horce_data )
last_wrap.create_score()
print( last_wrap.horce_last_wrap )
print( last_wrap.horce_four_corner_to_goal_time )
exit( 0 )
#race_data.get_all_data( "202406030302" )
wrap_data = race_data.data["wrap"]

wrap_list = []
ave_wrap = 0
all_wrap = 0

for key in wrap_data.keys():
    wrap = wrap_data[key]
    
    if key == '100':
        wrap *= 2

    ave_wrap += wrap
    all_wrap += wrap

ave_wrap /= len( wrap_data )
before_wrap = ave_wrap
w = 0

for key in wrap_data.keys():
    if key == '100':
        wrap_list.append( wrap_data[key] )
        before_wrap = wrap_data[key] * 2
        continue

    current_wrap = wrap_data[key]
    a = ( before_wrap - current_wrap ) / -200
    b = before_wrap
    middle_wrap = a * 100 + b
    wrap_list.append( middle_wrap / 2 )
    wrap_list.append( current_wrap / 2 )
    before_wrap = current_wrap

ymd = { "year": race_data.data["year"], \
       "month": race_data.data["month"], \
       "day": race_data.data["day"] }

key_place = str( race_data.data["place"] )
key_baba = str( race_data.data["baba"] )
key_dist = str( race_data.data["dist"] )
race_cource_info = dm.pickle_load( "race_cource_info.pickle" )
four_corner_dist = race_cource_info[key_place][key_baba][key_dist]["dist"][-1] + race_cource_info[key_place][key_baba][key_dist]["dist"][-2]
dist_one_index = int( ( race_data.data["dist"] - four_corner_dist ) / 100 )
dist_two_index = int( ( race_data.data["dist"] - four_corner_dist + 100 ) / 100 )

a = ( sum( wrap_list[0:dist_one_index] ) - sum( wrap_list[0:dist_two_index] ) ) / ( dist_one_index * 100 - dist_two_index * 100 )
b = sum( wrap_list[0:dist_one_index] ) - a * ( dist_one_index * 100 )
four_corner_to_goal_time = a * four_corner_dist + b
race_time = 100000
last_three_wrap = wrap_list[int(len(wrap_list)-6):len(wrap_list)]
horce_last_wrap = {}
race_up3 = sum( last_three_wrap )

for horce_id in race_horce_data.horce_id_list:
    current_data, past_data = lib.race_check( horce_data.data[horce_id]["past_data"], ymd )
    cd = lib.current_data( current_data )
    race_time = min( race_time, cd.race_time() )
    horce_last_wrap[horce_id] = []

    for i in range( 0, len( last_three_wrap ) ):
        horce_last_wrap[horce_id].append( last_three_wrap[i] * ( cd.up_time() / race_up3 ) )

horce_four_corner_to_goal_time = {}

for horce_id in race_horce_data.horce_id_list:
    current_data, past_data = lib.race_check( horce_data.data[horce_id]["past_data"], ymd )
    cd = lib.current_data( current_data )
    key_horce_num = str( int( cd.horce_number() ) )
    horce_four_corner_to_goal_time[horce_id] = four_corner_to_goal_time + ( cd.race_time() - race_time )
