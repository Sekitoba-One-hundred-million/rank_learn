from argparse import ArgumentParser

import sekitoba_data_manage as dm
import sekitoba_library as lib
from data_analyze import data_create
from machine_learn_torch import learn
from machine_learn_torch import nn
from machine_learn_torch import model_test
import simulation

def simu( model = None, simu_data = None, units = None ):

    if units == None:
        units = dm.pickle_load( "nn_rank_units.pickle" )

    if model == None:
        model = nn.LastStrightNN( units["n"], units["a"] )
        model = dm.model_load( "nn_rank_model.pth", model )

    if simu_data == None:
        simu_data = dm.pickle_load( "rank_simu_data.pickle" )
        
    model.eval()
    simulation.main( model, simu_data )
    
def main():
    lib.log.set_name( "nn_rank_learn_2.log" )
    
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
    model, units = learn.main( data )
    model.eval()
    model_test.main( model, simu_data )
    #simu( model = model, simu_data = simu_data, units = units )
    
if __name__ == "__main__":
    main()