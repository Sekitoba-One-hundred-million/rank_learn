from argparse import ArgumentParser

import sekitoba_data_manage as dm
import sekitoba_library as lib
from data_analyze import data_create
from machine_learn_torch import learn
from machine_learn_torch import nn
import simulation
from rank_learn import rank_learn

def simu( model = None, simu_data = None, units = None ):

    if units == None:
        units = dm.pickle_load( "last_staight_units.pickle" )

    if model == None:
        model = nn.LastStrightNN( units["n"], units["a"] )
        model = dm.model_load( "last_straight_model.pth", model )

    if simu_data == None:
        simu_data = dm.pickle_load( "last_straight_simu_data.pickle" )
        
    model.eval()
    simulation.main( model, simu_data )
    
def main():
    #lib.log.set_name( "nn_simulation_3.log" )
    lib.log.set_name( "rank_learn_9.log" )
    
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
    
    if s_check:
        simu()
        return
        
    data, simu_data = data_create.main( update = u_check )
    
    if r_check:
        lib.log.write( "rank learn" )
        rank_model = rank_learn.main( data, simu_data )
        return
    
    model, units = learn.main( data )
    simu( model = model, simu_data = simu_data, units = units )
    
if __name__ == "__main__":
    main()
