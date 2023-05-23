import sekitoba_data_manage as dm
data1 = dm.pickle_load( "rank_simu_data.pickle" )
data2 = dm.pickle_load( "rank_simu_data.pickle.backup-1684513162" )

TEACHER = "teacher"

for race_id in data1.keys():
    check = True

    for horce_id in data1[race_id].keys():
        for i in range( 0, len( data1[race_id][horce_id]["data"] ) ):
            if not data1[race_id][horce_id]["data"][i] == data2[race_id][horce_id]["data"][i]:
                print( horce_id, i, data1[race_id][horce_id]["data"][i], data2[race_id][horce_id]["data"][i] )
    break
