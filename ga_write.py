import sekitoba_library as lib

lib.name.set_name( "rank" )

def main():
    check_list = [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0]
    rf = open( lib.name.memo_name(), "r" )
    wf = open( "ga_" + lib.name.memo_name(), "w" )

    str_data_list = rf.readlines()

    for i in range( 0, len( str_data_list ) ):
        str_data = str_data_list[i].replace( "\n", "" ) + " " + str( check_list[i] ) + "\n"
        wf.write( str_data )

    rf.close()
    wf.close()

if __name__ == "__main__":
    main()
