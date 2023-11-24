import math
import numpy as np

import sekitoba_library as lib
import sekitoba_data_manage as dm

def data_check( data, test_years = lib.test_years ):
    result = {}
    result["teacher"] = []
    result["test_teacher"] = []
    result["answer"] = []
    result["test_answer"] = []
    result["query"] = []
    result["test_query"] = []

    count = 0

    for i in range( 0, len( data["query"] ) ):
        q = data["query"][i]["q"]
        year = data["query"][i]["year"]

        if q < 3:
            continue

        current_data = list( data["teacher"][i] )
        current_answer = list( data["answer"][i] )
        current_level = list( data["level"][i] )
        current_diff = list( data["diff"][i] )
        current_popular = list( data["popular"][i] )

        if 1 not in current_answer and year in lib.test_years:
            continue
        
        if year in lib.test_years:
            if year in test_years:
                result["test_query"].append( q )
        else:
            result["query"].append( q )
        
        for r in range( 0, len( current_data ) ):
            answer_rank = current_answer[r]

            if answer_rank == 1:
                answer_rank = 10
            elif answer_rank == 2:
                answer_rank = 5
            elif answer_rank == 3:
                answer_rank = 3
            else:
                answer_rank = 0

            if not answer_rank == 0:
                answer_rank += current_level[0]
                answer_rank += int( current_diff[r] )

            if year in lib.test_years:
                if year in test_years:
                    result["test_teacher"].append( current_data[r] )
                    result["test_answer"].append( float( answer_rank ) )
            else:
                result["teacher"].append( current_data[r] )
                result["answer"].append( float( answer_rank ) )

    return result
