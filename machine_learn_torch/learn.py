import random

import sekitoba_library as lib
import sekitoba_data_manage as dm
from machine_learn_torch import nn

def data_check( data ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    ma = -1
    mi = 100
    
    for i in range( 0, len( data["answer"] ) ):
        ma = max( data["answer"][i], ma )
        mi = min( data["answer"][i], mi )
        
        if data["year"][i] == "2020":
            result["test_teacher"].append( data["teacher"][i] )
            result["test_answer"].append( data["answer"][i] )
        else:
            result["teacher"].append( data["teacher"][i] )
            result["answer"].append( data["answer"][i] )

    print( ma )
    ma = len( data["answer_list"][0] )
    return result, ma

def batch_normalization( data ):
    average_list = [0] * len( data[0] )
    dev_list = [0] * len( data[0] )

    for i in range( 0, len( data[0] ) ):
        for r in range( 0, len( data ) ):
            average_list[i] += data[r][i]

    for i in range( 0, len( average_list ) ):
        average_list[i] /= len( data )

    for i in range( 0, len( data[0] ) ):
        for r in range( 0, len( data ) ):
            dev_list[i] += pow( data[r][i] - average_list[i], 2 )

    for i in range( 0, len( dev_list ) ):
        dev_list[i] /= len( data )

    for i in range( 0, len( data[0] ) ):
        if dev_list[i] == 0:
            continue
        
        for r in range( 0, len( data ) ):
            data[r][i] = ( data[r][i] - average_list[i] ) / dev_list[i]

    return data

def batch_data_check( data ):
    lib.log.write( "create batch data" )    
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []

    count = 0
    for i in range( 0, len( data["query"] ) ):
        year = data["query"][i]["year"]
        q = data["query"][i]["q"]

        if year == "2020":
            result["test_teacher"].extend( batch_normalization( data["teacher"][count:count+q] ) )
            result["test_answer"].extend( data["answer"][count:count+q] )
        else:
            result["teacher"].extend( batch_normalization( data["teacher"][count:count+q] ) )
            result["answer"].extend( data["answer_list"][count:count+q] )

        count += q
        
    ma = len( data["answer_list"][0] )
    return result, ma

def main( data, GPU = False ):
    units = {}
    learn_data, a_units = data_check( data )
    #learn_data, a_units = batch_data_check( data )
    n_units = len( data["teacher"][0] )
    print( n_units, a_units )
    units["n"] = n_units
    units["a"] = a_units

    dm.pickle_upload( "last_staight_units.pickle", units )
    model = nn.LastStrightNN( n_units, a_units )
    model = nn.main( learn_data, model, GPU )
    dm.model_upload( "last_straight_model.pth", model )
    
    return model, units
