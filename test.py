import sekitoba_data_manage as dm

simu_data = dm.pickle_load( "rank_simu_data.pickle" )
update_race_id_list = dm.pickle_load( "update_race_id_list.pickle" )

race_id = update_race_id_list[105]
print( "race_id: {}".format( race_id ) )

horce_id = list( simu_data[race_id].keys() )[2]
print( "horce_id: {}".format( horce_id ) )

f = open( "./common/rank_score_data.txt" )
all_data = f.readlines()

for i in range( 0, len( all_data ) ):
    value = simu_data[race_id][horce_id]["data"][i]
    str_data = all_data[i].replace( "\n", "" )
    print( "{}: {}".format( str_data, value ) )
