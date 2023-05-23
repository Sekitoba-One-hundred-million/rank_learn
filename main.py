def data_score_read():
    result = []
    f = open( "./common/rank_score_data.txt", "r" )
    all_data = f.readlines()

    for i in range( 0, len( all_data ) ):
        split_data = all_data[i].replace( "\n", "" ).split( " " )

        if len( split_data ) == 2:
            result.append( i )
            
    f.close()
    result = sorted( result, reverse = True )
    return result

def data_remove( data: list, delete_data: list ):
    for i in range( 0, len( delete_data ) ):
        data.pop( delete_data[i] )

    return data

def main():
    from argparse import ArgumentParser
    import matplotlib.pyplot as plt
    import numpy as np
    from mpi4py import MPI
    from tqdm import tqdm

    import sekitoba_data_manage as dm
    import sekitoba_library as lib
    from data_analyze import data_create
    from learn import learn
    from learn import rate_learn
    from simulation import buy_simulation

    lib.name.set_name( "rank" )

    lib.log.set_write( False )
    parser = ArgumentParser()
    parser.add_argument( "-u", type=bool, default = False, help = "optional" )
    parser.add_argument( "-s", type=bool, default = False, help = "optional" )
    parser.add_argument( "-l", type=bool, default = False, help = "optional" )

    u_check = parser.parse_args().u
    s_check = parser.parse_args().s
    l_check = parser.parse_args().l
    
    data = data_create.main( update = u_check )

    if not data  == None:
        simu_data = data["simu"]
        learn_data = data["data"]
        remove_list = data_score_read()

        for k in simu_data.keys():
            for kk in simu_data[k].keys():
                simu_data[k][kk]["data"] = data_remove( simu_data[k][kk]["data"], remove_list )

        for i in range( 0, len( learn_data["teacher"] ) ):
            for r in range( 0, len( learn_data["teacher"][i] ) ):
                learn_data["teacher"][i][r] = data_remove( learn_data["teacher"][i][r], remove_list )

        models = dm.pickle_load( "test_model.pickle" )

        if models == None or l_check or u_check:
            models = {}
            #models.update( rate_learn.main( data["data"] ) )
            models.update( learn.main( data["data"] ) )
            
        buy_simulation.main( models, simu_data )
        #tree_model_data_create()
        dm.pickle_upload( "test_model.pickle", models )
        
    MPI.Finalize()        
    
if __name__ == "__main__":
    #from simulation import buy_simulation
    #buy_simulation.main()
    main()
