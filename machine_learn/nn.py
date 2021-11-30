import numpy as np
import math
import random
import chainer
from chainer import Link, Chain, ChainList, optimizers, Variable, cuda, serializers
import chainer.functions as F
import chainer.links as L
from tqdm import tqdm

import sekitoba_data_manage as dm
import sekitoba_library as lib

class StraightHorceBodyNN( Chain ):

    def __init__( self, n, a ):
        super( StraightHorceBodyNN, self ).__init__(
            l1 = L.Linear( n, n ),
            l2 = L.Linear( n, n ),
            l3 = L.Linear( n, n ),
            l4 = L.Linear( n, n ),
            l5 = L.Linear( n, n ),
            l6 = L.Linear( n, n ),
            l7 = L.Linear( n, n ),
            l8 = L.Linear( n, n ),
            l9 = L.Linear( n, n ),
            l10 = L.Linear( n, n ),
            l11 = L.Linear( n, n ),
            l12 = L.Linear( n, n ),
            l13 = L.Linear( n, n ),
            l14 = L.Linear( n, n ),
            l15 = L.Linear( n, n ),
            l16 = L.Linear( n, n ),
            l17 = L.Linear( n, n ),
            l18 = L.Linear( n, n ),
            l30 = L.Linear( n, a ),
            
            b1 = L.BatchNormalization( n ),
            b2 = L.BatchNormalization( n ),
            b3 = L.BatchNormalization( n ),
            b4 = L.BatchNormalization( n ),
            b5 = L.BatchNormalization( n ),
            b6 = L.BatchNormalization( n ),
            b7 = L.BatchNormalization( n ),
            b8 = L.BatchNormalization( n ),
            b9 = L.BatchNormalization( n ),
            b10 = L.BatchNormalization( n ),
            b11 = L.BatchNormalization( n ),
            b12 = L.BatchNormalization( n ),
            b13 = L.BatchNormalization( n ),
            b14 = L.BatchNormalization( n ),
            b15 = L.BatchNormalization( n ),
            b16 = L.BatchNormalization( n ),
            b17 = L.BatchNormalization( n ),
        )

    def forward( self, x ):
        h1 = F.relu( self.l1( x ) )
        h2 = F.relu( self.b1( ( self.l2( h1 ) ) ) )
        h3 = F.relu( self.b2( self.l3( h2 ) ) )
        h4 = F.relu( self.b3( self.l4( h3 + h2 ) ) )
        h5 = F.relu( self.b4( self.l5( h4 ) ) )
        h6 = F.relu( self.b5( self.l6( h5 + h4 ) ) )
        h7 = F.relu( self.b6( self.l7( h6 ) ) )
        h8 = F.relu( self.b7( self.l8( h7 + h6 ) ) )
        h9 = F.relu( self.b8( self.l9( h8 ) ) )
        h10 = F.relu( self.b9( self.l10( h9 + h8 ) ) )
        h11 = F.relu( self.b10( self.l11( h10 ) ) )
        h12 = F.relu( self.b11( self.l12( h11 + h10 ) ) )
        h13 = F.relu( self.b12( self.l13( h12 ) ) )
        h14 = F.relu( self.b13( self.l14( h13 + h12 ) ) )
        h15 = F.relu( self.b14( self.l15( h14 ) ) )
        h16 = F.relu( self.b15( self.l16( h15 + h14 ) ) )
        h17 = F.relu( self.b16( self.l17( h16 ) ) )
        h18 = F.relu( self.b17( self.l18( h17 + h16 ) ) )
        h30 = self.l30( h18 )

        return h30

def test( test_teacher, test_answer, model ):
    predict_answer = model.forward( Variable( np.array( test_teacher, dtype = np.float32 ) ) ).data
    pa = np.argmax( np.array( predict_answer ), axis = 1 )
    count = 0
    diff_check = 0

    for i in range( 0, len( test_answer ) ):
        diff_check += abs( pa[i] - test_answer[i] )
        
        if pa[i] == test_answer[i]:
            count += 1

    diff_check /= len( test_teacher )
    diff_check /= 2

    return count / len( test_teacher ) * 100, diff_check
    
    
def main( data, model, GPU = False ):
    teacher_data = data["teacher"]
    answer_data = data["answer"]
    test_teacher = data["test_teacher"]
    test_answer = data["test_answer"]
    
    optimizer = optimizers.Adam()
    optimizer.setup( model )
    gpu_device = 0
    
    if GPU:
        print( "GPU使用" )
        cuda.get_device( gpu_device ).use()
        model.to_gpu( gpu_device )
        xp = cuda.cupy
    else:
        xp = np

    N = len( teacher_data )
    epoch = 500
    batch_size = 2048
    print( N )
    teacher_data = Variable( xp.array( teacher_data, dtype = xp.float32 ) )
    answer_data = Variable( xp.array( answer_data, dtype = xp.int32 ) )
    
    for e in range( 0, epoch ):
        all_loss = 0
        data_list = list( range( 0, N ) )
        random.shuffle( data_list )
        
        for i in range( 0, int( N / batch_size ) ):
            b = i * batch_size
            model.zerograds()
            y = model.forward( teacher_data[data_list[b:b+batch_size]] )
            #loss = F.mean_squared_error( y, answer_data[data_list[b:b+batch_size]] )
            loss = F.softmax_cross_entropy( y, answer_data[data_list[b:b+batch_size]] )
            all_loss += loss.data
            loss.backward()
            optimizer.update()

        answer_rate, diff_minute = test( test_teacher, test_answer, model )
        print( "学習:{}回 正答率:{}% 誤差:{}馬身 loss:{}".format( e + 1, answer_rate, diff_minute, all_loss / int( N / batch_size ) ) )

        model.to_cpu()

        if GPU:
            model.to_gpu( gpu_device )    

    dm.model_upload( "suzuka_test_model.pickle", model )
    
    return model
