import math
import numpy as np

import sekitoba_library as lib
import sekitoba_data_manage as dm

def data_check( data, state = "test" ):
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
        data_check = lib.test_year_check( data["year"][i], state )

        #if 1 not in current_answer and year in lib.test_years:
        #    continue

        if data_check == "test":
            result["test_query"].append( q )
        elif data_check == "teacher":
            result["query"].append( q )

        min_diff = min( current_diff )

        for r in range( 0, len( current_data ) ):
            answer_score = int( ( 1 / ( current_diff[r] - min_diff + 1 ) ) * len( current_answer ) )
            answer_rank = current_answer[r]

            if answer_rank == 1:
                answer_score += 10
            elif answer_rank == 2:
                answer_score += 7
            elif answer_rank == 3:
                answer_score += 5

            #print( answer_rank, answer_score )
            if data_check == "test":
                result["test_teacher"].append( current_data[r] )
                result["test_answer"].append( float( answer_score ) )
            elif data_check == "teacher":
                result["teacher"].append( current_data[r] )
                result["answer"].append( float( answer_score ) )

    return result
