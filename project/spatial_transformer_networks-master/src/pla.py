import numpy as np
import time
import matplotlib.pyplot as plt
import random
import sys
import threading
def read_input(path):
    buf_x = []
    buf_y = []
    with open(path,'r') as f:
        lines = f.readlines()
    for l in lines:
        _ = l.strip().replace('\t',' ').split(' ')
        #add bias -> x0
        x = [1] + _[0:-1]
        assert len(x) == 5
        y = _[-1]
        buf_x.append(x)
        buf_y.append(y)
    X = np.vstack(buf_x).astype(np.float32) 
    Y = np.vstack(buf_y).astype(np.int32) 
    return X,Y
def pla(X,Y,w,lr=1.,indice = None):
    if indice == None:
        indice = xrange(X.shape[0])
    update_count = 0
    for i in indice:
        x = X[i,:]
        y = Y[i]

        wx = 1 if np.dot(x,w) > 0 else -1
        #encounter a error
        if wx != y:
            w += y*x*lr
            #print 'error at ',x,wx,y
            update_count += 1

    #no error -> 0
    no_error = True if update_count == 0 else False
    return update_count,no_error
def pocket_pla(X,Y,w,lr=1.):
    update_count = 0
    error = False
    while not error:
        i = random.randint(0,X.shape[0]-1) 
        x = X[i,:]
        y = Y[i]

        wx = 1 if np.dot(x,w) > 0 else -1
        #encounter a error
        if wx != y:
            w += y*x*lr
            error = True
def verify(X,Y,w):
    error_num = 0.
    
    w2 = w.reshape(5,1)
    Z = np.matmul(X,w2)

    for i in xrange(X.shape[0]):
        z = Z[i,0]
        y = Y[i]

        z = 1 if z > 0 else -1
        #encounter a error
        if z != y:
            error_num += 1
    return error_num

def shuffle_indice(index_range,reset_seed=True):
    if reset_seed:
        random.seed(time.time())
    
    indice = range(index_range)
    random.shuffle(indice)
    return indice

def hw1_15():
    input_path = './hw1_8_train.dat.txt'
    X,Y = read_input(input_path)
    
    w = np.array([0]*5,dtype=np.float32)
    c = 0
    no_error = False
    while not no_error:
        update_count,no_error = pla(X,Y,w)
        c += update_count
    print c,w

#fixed, pre-determined random cycles
def hw1_16(num):
    input_path = './hw1_8_train.dat.txt'
    X,Y = read_input(input_path)
    
    cycle_list = [] 
    for _ in xrange(num):

        indice = shuffle_indice(X.shape[0])
        #init w 
        w = np.array([0]*5,dtype=np.float32)
        #init counter
        c = 0
        end = False
        
        no_error = False
        while not no_error:
            update_count,no_error = pla(X,Y,w,indice = indice)
            c += update_count
        cycle_list.append(c)
    print np.mean(cycle_list),np.var(cycle_list)

#fixed, pre-determined random cycles with learn rate
def hw1_17(num,lr):
    input_path = './hw1_8_train.dat.txt'
    X,Y = read_input(input_path)
    ll = []
    cycle_list = [] 
    for _ in xrange(num):

        indice = shuffle_indice(X.shape[0])
        #init w 
        w = np.array([0]*5,dtype=np.float32)
        #init counter
        c = 0
        end = False
        
        no_error = False
        while not no_error:
            update_count,no_error = pla(X,Y,w,lr=lr,indice = indice)
            c += update_count
        cycle_list.append(c)
    print np.mean(cycle_list),np.var(cycle_list)
#fixed, pre-determined random cycles with pocket algo
#purely random
def hw1_18(num,pocket_num):
    input_path = './hw1_18_train.dat.txt'
    X,Y = read_input(input_path)
    test_path = './hw1_18_test.dat.txt'
    test_X,test_Y = read_input(input_path)
    error_list = [] 
    thread_list = []
    for _ in xrange(num):
        t = threading.Thread(target = hw1_18_mul, args = (X,Y,test_X,test_Y,pocket_num,error_list))
        thread_list.append(t)
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()

    print np.mean(error_list),np.var(error_list)
def hw1_18_mul(X,Y,test_X,test_Y,update_time,error_list):

    #init w 
    w = np.array([0]*5,dtype=np.float32)
    #init counter
    p = w.copy()       #pocket ,best w 

    for _ in xrange(update_time):
        #
        pocket_pla(X,Y,w)
        w_error = verify(X,Y,w)
        p_error = verify(X,Y,p)
        if w_error < p_error :
            np.copyto(p,w)
        if p_error == 0:
            break
        #


    e = verify(test_X,test_Y,p)  / test_X.shape[0] #Eout acc
    error_list.append(e)
#turly random per cycle, output the w after fixed, pocket_num update
#purely random
def hw1_19(num,pocket_num):
    input_path = './hw1_18_train.dat.txt'
    X,Y = read_input(input_path)
    test_path = './hw1_18_test.dat.txt'
    test_X,test_Y = read_input(input_path)
    error_list = [] 
    thread_list = []
    for _ in xrange(num):
        t = threading.Thread(target = hw1_19_mul, args = (X,Y,test_X,test_Y,pocket_num,error_list))
        thread_list.append(t)
    for t in thread_list:
        t.start()
    for t in thread_list:
        t.join()

    print np.mean(error_list),np.var(error_list)
def hw1_19_mul(X,Y,test_X,test_Y,update_time,error_list):

    #init w 
    w = np.array([0]*5,dtype=np.float32)
    #init counter

    for _ in xrange(update_time):
        #
        pocket_pla(X,Y,w)
        #


    e = verify(test_X,test_Y,w)  / test_X.shape[0] #Eout acc
    error_list.append(e)

#fixed, pre-determined random cycles
def hw1_8_histogram(num):
    input_path = './hw1_8_train.dat.txt'
    X,Y = read_input(input_path)
    
    cycle_list = [] 
    for _ in xrange(num):

        indice = shuffle_indice(X.shape[0])
        #init w 
        w = np.array([0]*5,dtype=np.float32)
        #init counter
        c = 0
        no_error = False 
        while not no_error:
            update_count,no_error = pla(X,Y,w,indice = indice)
            c += update_count
        cycle_list.append(c)
    #histogram
    a = np.array(cycle_list)
    plt.hist(a)
    plt.show()
    print np.mean(cycle_list),np.var(cycle_list)

def main(argv):

    if argv == '15':
        hw1_15()         #45 cycle
    elif argv == '16':
        hw1_16(2000)    #40 cycle
    elif argv == '17':
        hw1_17(2000,lr=0.5)    #40 cycle
    elif argv == '18':
        hw1_18(2000,50)    #error rate : 0.125382
    elif argv == '19':
        hw1_19(2000,50)     #error rate : 0.351831
    elif argv == '20':
        hw1_18(2000,100)    #error rate : 0.1079
    elif argv.find('hist') != -1:
        hw1_8_histogram(2000)
    else:
        usage()
def usage():
    print 'usage    :   python ./pla.py <problem id>'
    print 'example  :   python ./pla.py 15'
    print 'output   :   <mean>  <variance>'
    print 'note!'
    print 'the dataset must be the same directory : ./  '

if __name__ == '__main__':
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        usage()
