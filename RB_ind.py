import numpy as np
import scipy as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
import cmath
import numpy.random as random
from numpy.random import choice, sample, randint
from scipy import optimize
from scipy.optimize import curve_fit
import statistics
import os
import pandas as pd
import qinfer as qi
import time

start_time = time.clock()


Zero = np.matrix('1.0; 0.0')
One = np.matrix('0.0; 1.0')


def Id(n):
    return np.eye(n)

def infid(A,B):
    A = np.matrix(A)
    B = np.matrix(B)
    F = np.dot(A.getH(),B)
    F = np.trace(F)
    r = 1-F**2/4
    return r


def NormalizeState(state):
    return state/sp.linalg.norm(state)

def Nkron(*args):
    result = np.matrix('1.0')
    for op in args:
        result = np.kron(result, op)
    return result

def fit_1st(x,A,B,p):
    return A*p**x+B

pi = math.pi


#Operators
P0 = np.dot(Zero,Zero.T)
P1 = np.dot(One,One.T)
X = np.matrix('0. 1.; 1. 0.')
Y = np.matrix('0 -1j; 1j 0')
Z = np.matrix('1. 0.; 0. -1.')
axis = ([1,0,0],[0,1,0],[0,0,1],
        [1,0,1],[1,0,-1],[1,1,0],[1,-1,0],[0,1,1],[0,1,-1],
        [1,1,1],[1,1,-1],[1,-1,1],[-1,1,1],[0,0,0])
angle_1 = [pi,-pi/2,pi/2]
angle_2 = [pi]
angle_3 = [2*pi/3,4*pi/3]


#Clifford Elements

cliff = np.load('./cliff.npy')

axis = ([1,0,0,pi],[1,0,0,-pi/2],[1,0,0,pi/2],[0,1,0,pi],[0,1,0,-pi/2],[0,1,0,pi/2],[0,0,1,pi],[0,0,1,-pi/2],[0,0,1,pi/2],[1,0,1,pi],[1,0,-1,pi],[1,1,0,pi],[1,-1,0,pi],[0,1,1,pi],[0,1,-1,pi],[1,1,1,2*pi/3],[1,1,1,4*pi/3],[1,1,-1,2*pi/3],[1,1,-1,4*pi/3],[1,-1,1,2*pi/3],[1,-1,1,4*pi/3],[-1,1,1,2*pi/3],[-1,1,1,4*pi/3],[0,0,0,0])

####################################################################


#Propagators

def id_rot(a,b,c,d):
    m = [a,b,c]
    if(a==b==c==0):
        return Id(2)
    else:
        return la.expm(-1j*(a*X+b*Y+c*Z)*d/(2*la.norm(m)))


def H_x(a):
    return (a)*X
def H_z(b,c):
    return (b-c)*Z
def H_y(a):
    return a*Y

def Rot(angle,phase):

    T = angle/(2*W)
    Omega = np.random.normal(W,1e-2*W)
    H_tot = math.cos(phase)*H_x(Omega)+math.sin(phase)*H_y(Omega)
    U = la.expm(-1j*H_tot*T)

    return U


def U3(theta,phi,lamda):

    A = np.matmul(R_X(pi/2),R_Z(lamda-pi/2))
    A = np.matmul(R_Z(pi-theta),A)
    A = np.matmul(R_X(pi/2),A)
    A = np.matmul(R_Z(phi-pi/2),A)

    return A


def voss(nrows, ncols=16):
    """Generates pink noise using the Voss-McCartney algorithm.                    
                                                                                   
    nrows: number of values to generate                                            
    rcols: number of random sources to add                                         
                                                                                   
    returns: NumPy array                                                           
    """
    array = np.empty((nrows, ncols))
    array.fill(np.nan)
    array[0, :] = np.random.random(ncols)
    array[:, 0] = np.random.random(nrows)

    # the total number of changes is nrows                                         
    n = nrows
    cols = np.random.geometric(0.5, n)
    cols[cols >= ncols] = 0
    rows = np.random.randint(nrows, size=n)
    array[rows, cols] = np.random.random(n)

    df = pd.DataFrame(array)
    df.fillna(method='ffill', axis=0, inplace=True)
    total = df.sum(axis=1)
    x = total.values
    max = np.amax(x)
    min = np.amin(x)
    A = (max-min)/2
    x = (x-A)/A

    return x

sign = [1,-1]

def rot(a,b,c,phi,sigma):

    if(a==b==c==0):
        return Id(2)
    else:
        pass

    if(a==0):
        Omega1 = 0
        W1 = 0
    else:
        Omega1 = np.random.normal(W,sigma)
        W1 = W

    if(b==0):
        Omega2 = 0
        W2 = 0
    else:
        Omega2 = W
        W2 = W


    if(a==b==0)and(c!=0):
        omega = 0
        w = 0
    elif(c==0):
        omega = w_0
        w = w_0
    else:
        omega = w_0-W
        w = w_0-W

    phase1 = 0
    phase2 = pi/2
    v = [W1*(math.cos(phase1)+math.cos(phase2)),W2*(math.sin(phase1)+math.sin(phase2)),w_0-w]
    T = phi/(2*la.norm(v))
    H_tot = math.cos(phase1)*H_x(a*Omega1)+math.sin(phase1)*H_y(b*Omega1)+math.cos(phase2)*H_x(a*Omega2)+math.sin(phase2)*H_y(b*Omega2)+c*(w_0-omega)*Z
    U = la.expm(-1j*H_tot*T)


    return U


def cliff_inv(A):
    error = []

    for s in np.arange(0,len(cliff)):
        D = cliff[s]
        r = infid(A,D)
        error.append(r)
        
        min_value = np.amin(error)
        
    for s in np.arange(0,len(error)):
        number = error[s]
        if(number == min_value):
            N = s
                

    if(N==23):
        C = axis[N]
    elif(N<=8):
        if(N%3==0):
            C = axis[N]
        elif(N%3==1):
            C = axis[N+1]
        else:
            C = axis[N-1]
    elif(9<=N)and(N<=15):
        C = axis[N]
    elif(16<=N)and(N<=22):
        if(N%2==0):
            C = axis[N-1]
        else:
            C = axis[N+1]
    return C



def rand_axis():
    n = randint(0,13)
    u = axis[n]

    if(la.norm(u) == 1):
        phi = random.choice(angle_1)
    else:
        pass
    if(la.norm(u) > 1 and  la.norm(u) < 1.5):
        phi = pi
    else:
        pass
    if(la.norm(u) > 1.5):
        phi = random.choice(angle_3)
    else:
        pass
    u = [u[0],u[1],u[2],phi]

    return u


w_0 = 5e9
W = 600e3
w = 5e9


#Numbers of measurement for each sequence
N_m = 4
n_m = float(N_m)
#Numbers of sequence for each length 
N_e = 4
n_e = float(N_e)
#Maximum length (2^M gates)
M = 11
m = float(M)


print("Enable preparation error?")
prep = str(input())
if(prep == "y"):
    prep = 1
    prep_error = 1e-2
else:
    prep = 0

print("Enable measurement error?")
meas = str(input())
if(meas == "y"):
    meas = 1
    meas_error = 1e-2
else:
    meas = 0
print("Minimum noise")
n_min = float(input())
print("Maximum noise")
n_max = float(input())



p = []
x = []
for i in np.arange(0,M+1,10):
    x.append(int(i))

np_x = np.array(x)

if(prep == 1):
    in_state = NormalizeState(Zero+prep_error*One)
else:
    in_state = Zero
z = 0
for m in np.arange(n_min,n_max,0.5):
    noise = 10**(m)
    print("Noise = 10^%f" %m)
    x = []
    L = M-z
    print("M = %d" %L)
    for i in np.arange(1,L+1):
        x.append(2**i)
        
    np_x = np.array(x)

    y = []
    e = []
    mu = []
    sigma = []

    z = z+1
    print(L)
    for i in np.arange(1,L+1):


        data = []
        for v in np.arange(1,N_e+1):
            Count = 0.
            seq = Id(2)
            seq2 = Id(2)
            NewState = in_state


            for k in np.arange(2**i):
                n = np.random.randint(0,24)
                C = axis[n]
                B = rot(C[0],C[1],C[2],C[3],noise)
                seq = np.dot(B,seq)
                NewState = np.dot(B,NewState)

            inv = cliff_inv(seq)
            seq_inv = rot(inv[0],inv[1],inv[2],inv[3],noise)
            ResultState = np.dot(seq_inv, NewState)
            RhoResult = np.dot(ResultState, ResultState.getH())
            Prob0 = np.trace(np.dot(P0,RhoResult))
            Prob1 = np.trace(np.dot(P1,RhoResult))
#            print("Prob measuring 0: %f, Prob measuring 1: %f" %(Prob0, Prob1))

            for j in np.arange(1,N_m+1):
                a = np.random.rand()
                if(a < math.fabs(Prob0)):
                    Result = 0
                    Count += 1
                else:
                    Result = 1
#                print("Result of mesurement #%d, exp. #%d (sequence length %d): %d" %(j,v,i, Result))
                                            
            r = Count

            data.append(r)
            
        print("Finished(sequence length %d)" %2**i)
        mu = int(statistics.mean(data))
        sigma = statistics.stdev(data)
        y.append(mu)
        e.append(sigma)
    
    np_y = np.array(y)
    np_e = np.array(e)

    print("Sequence Length")
    print(np_x)
    print("Counts of Zeros: ")
    print(np_y)
    print("Standard Deviation:")
    print(np_e)
        
    if(prep==1):
        np.savetxt('./data/ind_noise/RB_1e'+str(m)+'_pe.txt',np.transpose([np_x,np_y,np_e]))
    else:
        np.savetxt('./data/ind_noise/RB_1e'+str(m)+'.txt',np.transpose([np_x,np_y,np_e]))

     
#    data = np.column_stack([np_y*N_m,np_x,np.ones_like(np_y)*N_m])
#    mean, cov, extra = qi.simple_est_rb(data, return_all=True,n_particles=500000, p_min=0.0,p_max=1.0)

#    print("[p, A, B]")
#    print(mean)
#    r = (2-1)*(1-mean[0])/2
#    print("Infidelity r:")
#    print(r)

#    p.append(mean[0])




#plt.axis([0,M+10,0,1.1])
#plt.title("Randomized Benchmarking Simulation")
#plt.xlabel("Seqeuence Length m")
#plt.ylabel("Fidelity")
#l1 = plt.plot(np_x, np_y,'r.', markersize=5, label = "Simulated Data Points")
#l2 = plt.plot(np_x, fit_1st(np_x,params[0],params[1],params[2]),'b',label = "Fitting Curve")
#plt.errorbar(np_x,np_y, np_e, linestyle='None', capsize=3)


print("Runtime: ", time.clock()-start_time, "seconds")


#np.savetxt('./data/fit_par_1.txt',(params))
np_p = np.array(p)
if(prep==1):
    np.savetxt('./data/ind_noise/RB_fid_ind_pe.txt',(np_p))
else:
    np.savetxt('./data/ind_noise/RB_fid_ind.txt',(np_p))

#plt.savefig('plot/RB_sim_1.png')

#plt.show()




