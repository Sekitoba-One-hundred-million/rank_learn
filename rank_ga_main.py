import copy
from argparse import ArgumentParser

import genetic_algorithm 
import sekitoba_data_manage as dm
import sekitoba_library as lib
from data_analyze import data_create
from learn import learn

lib.name.set_name( "rank" )

def data_remove( data, simu_data, remove_data ):
    data_result = []
    
    for i in range( 0, len( data["teacher"] ) ):
        t_instance = []        
        for r in range( 0, len( remove_data ) ):
            if remove_data[r] == 1:
                t_instance.append( data["teacher"][i][r] )

        data_result.append( t_instance )

    data["teacher"] = data_result
    
    for k in simu_data.keys():
        for kk in simu_data[k].keys():
            t_instance = []
            
            for i in range( 0, len( remove_data ) ):
                if remove_data[i] == 1:
                    t_instance.append( simu_data[k][kk]["data"][i] )

            simu_data[k][kk]["data"] = t_instance

def score_create( data, win_rate ):
    score = 0    
    
    if data < 80:
        return score    

    score += win_rate - 20
    score = data - 80

    if win_rate > 25:
        score += 5

    if 90 < data:
        score += 5

    if 100 < data:
        score += 10

    return score# * win_rate

def main():
    #lib.log.set_write( False )
    lib.log.set_name( "ga_rank_learn" )
    download_data = data_create.main( update = False )

    data = copy.deepcopy( download_data["data"] )
    simu_data = copy.deepcopy( download_data["simu"] )
    download_data.clear()
    
    p = 10
    e = len( data["teacher"][0] )
    ga = genetic_algorithm.GA( e, p )
    lib.log.write( "ga_rank_learn" )
    max_recovery_rate = 0
    max_win_rate = 0

    for i in range( 0, 100 ): 
        score_result = []
        parent = ga.get_parent()
        
        for r in range( 0, p ):
            instance_data = copy.deepcopy( data )
            instance_simu_data = copy.deepcopy( simu_data )
            data_remove( instance_data, instance_simu_data, parent[r] )
            simu_result = learn.main( instance_data, instance_simu_data )
            recovery_rate = simu_result["recovery"]
            win_rate = simu_result["win"]
            max_recovery_rate = max( max_recovery_rate, recovery_rate )
            score_result.append( score_create( recovery_rate, win_rate ) )
        
        ga.scores_set( score_result )
        ga.next_genetic()
        best_population, best_score = ga.get_best()
        print( best_score )
        print( best_population )
        print( max_recovery_rate )
        lib.log.write( str( i + 1 ) + " " + str( max( score_result ) ) + "%" )
        lib.log.write( "-----------------" )

    best_population, best_score = ga.get_best()
    print( best_score )
    print( best_population )
    print( max_recovery_rate )
    lib.log.write( str( best_population ) )
    lib.log.write( str( best_score ) )
    lib.log.write( str( max_recovery_rate ) + "%" )

if __name__ == "__main__":
    main()
