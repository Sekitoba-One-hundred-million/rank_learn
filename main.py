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

    lib.name.set_name( "rank" )

    lib.log.set_write( False )
    parser = ArgumentParser()
    parser.add_argument( "-u", type=bool, default = False, help = "optional" )
    parser.add_argument( "-s", type=bool, default = False, help = "optional" )
    parser.add_argument( "-r", type=bool, default = False, help = "optional" )

    u_check = parser.parse_args().u
    s_check = parser.parse_args().s
    r_check = parser.parse_args().r
    
    data = data_create.main( update = u_check )
    
    if not data  == None:
        #rate_learn.main( data["data"], data["simu"] )
        learn.main( data["data"], data["simu"] )
        #tree_model_data_create()
        
    MPI.Finalize()        
    
if __name__ == "__main__":
    #from simulation import buy_simulation
    #buy_simulation.main()
    main()
