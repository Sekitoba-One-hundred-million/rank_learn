import copy
import random
import numpy as np
from tqdm import tqdm

import SekitobaLibrary as lib
import SekitobaDataManage as dm
from machine_learn_torch import nn

def data_check( data, models ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    ma = -1
    mi = 100
    
    for i in tqdm( range( 0, len( data["answer"] ) ) ):
        ma = max( data["answer"][i], ma )
        mi = min( data["answer"][i], mi )
        use_teacher = copy.copy( data["teacher"][i] )
        rank_score = models["rank"].predict( np.array( [ use_teacher ] ) )[0]
        answer = 0

        if data["answer"][i] == 1:
            answer = 1        
            
        use_teacher.append( rank_score )

        if data["year"][i] == "2020":
            result["test_teacher"].append( use_teacher )
            result["test_answer"].append( answer )
        else:
            result["teacher"].append( use_teacher )
            result["answer"].append( answer )

    print( ma )
    #ma = len( data["answer_list"][0] )
    ma = 2
    return result, ma

def main( data, GPU = False ):
    units = {}
    models = {}
    models["rank"] = dm.pickle_load( "rank_model.pickle" )
    learn_data, a_units = data_check( data, models )
    n_units = len( learn_data["teacher"][0] )
    print( n_units, a_units )
    units["n"] = n_units
    units["a"] = a_units

    dm.pickle_upload( "nn_rank_units.pickle", units )
    model = nn.LastStrightNN( n_units, a_units )
    model = nn.main( learn_data, model, GPU )
    dm.model_upload( "nn_rank_model.pth", model )
    
    return model, units
