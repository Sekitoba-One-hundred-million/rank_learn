from argparse import ArgumentParser
import numpy as np
from mpi4py import MPI

import sekitoba_data_manage as dm
import sekitoba_library as lib
from data_analyze import data_create
from learn import learn

lib.name.set_name( "rank" )

def tree_model_data_create():
    result = {}
    model = dm.pickle_load( lib.name.model_name() )
    simu_data = dm.pickle_load( lib.name.simu_name() )
    
    for k in simu_data.keys():
        data = []
        result[k] = {}
        
        for kk in simu_data[k].keys():
            pah = model.predict( np.array( [ simu_data[k][kk]["data"] ] ) )[0]            
            result[k][kk] = {}
            result[k][kk]["score"] = pah
            result[k][kk]["answer"] = simu_data[k][kk]["answer"]

    dm.pickle_upload( lib.name.score_name(), result )

def main():
    lib.log.set_write( False )
    parser = ArgumentParser()
    parser.add_argument( "-g", type=bool, default = False, help = "optional" )
    parser.add_argument( "-u", type=bool, default = False, help = "optional" )
    parser.add_argument( "-s", type=bool, default = False, help = "optional" )
    parser.add_argument( "-r", type=bool, default = False, help = "optional" )

    g_check = parser.parse_args().g
    u_check = parser.parse_args().u
    s_check = parser.parse_args().s
    r_check = parser.parse_args().r
    lib.log.write( "g_check:" + str( g_check ) )
    lib.log.write( "u_check:" + str( u_check ) )
    lib.log.write( "s_check:" + str( s_check ) )
    lib.log.write( "r_check:" + str( r_check ) )
    
    data = data_create.main( update = u_check )
    
    if not data  == None:        
        learn.main( data["data"], data["simu"] )
        tree_model_data_create()
        
    MPI.Finalize()        
    
if __name__ == "__main__":
    main()
