#Uses python3

import numpy as np
import random as rnd
import time
import pandas as pd
import sys
rnd.seed(int(sys.argv[1]))
np.random.seed(int(sys.argv[1]))

#====================== all functions ======================
def kron(i,j):
    """Kroneker's symbol"""
    if i==j: return 1
    else: return 0

def dec_to_base4(N, base=4):
    """переводит из 10-чной в q-чную систему счисления"""
    if not hasattr(dec_to_base4, 'table'):        
        dec_to_base4.table = '0123456789ABCDEF'       
    x, y = divmod(N, base)        
    return dec_to_base4(x, base) + dec_to_base4.table[y] if x else dec_to_base4.table[y]
 
def gen_e0(n):
    """генерирует все возможные последовательности из 5 спинов"""
    n_4 = dec_to_base4(n)
    n_arr = [0]*(5-len(n_4)) + [int(x) for x in n_4]
    return n_arr

def calc_e_dict(arr):
    """вычисляет энергию конкретной конфигурации"""
    e =(kron(arr[0],arr[1]) + 
        kron(arr[0],arr[2]) +
        kron(arr[0],arr[3]) + 
        kron(arr[0],arr[4]))
    return -e

def gen_e_dict():
    """генерируем и заполняем словарь всех энергий"""
    d = dict()
    for i in range(0,1024):
        conf = np.array(gen_e0(i))
        d[str(conf)] = calc_e_dict(conf)
    return d

def coord(i):
    """get coordinate i of vector"""
    x = i // L
    y = i - x*L
    return (x,y)

def get(i):
    """fixin' boundary"""
    if i<0: return i
    else: return i % L
    
def get_neigh():
    """get neighbour's arr"""
    s = np.arange(L**2).reshape(L,L)
    nei = [[]]*L*L
    for n in range(len(nei)):
        i,j = coord(n)
        nei[n] = [s[get(i),get(j)],s[get(i-1),get(j)],s[get(i),get(j+1)],s[get(i+1),get(j)],s[get(i),get(j-1)]]
    return nei

#################################################################

def gen_state():
    """generate random start state with lenght L*L and q components"""
    state = np.random.randint(0, q, L*L)
    return state

def mc_choice(dE,T):
    """принимаем или не принимаем переворот спина?"""
    if dE <= 0:
        return True
    elif rnd.uniform(1,0) <= np.exp(-dE/T):
        return True
    else:
        return False
    
def step(s,edict,nei,T):
    """крутим 1 спин"""
    i = rnd.randint(0, L*L-1)      ### выбираем случайный спин
    new_val = rnd.choice([_ for _ in range(q)])
    
    arr = s[nei[i]]           ### формируем конфигурацию "вокруг" него
    arr_before = arr.copy()       ### конфигурация ДО переворота спина
    arr[0] = new_val    ### конфигурация ПОСЛЕ переворота спина
    dE = edict[str(arr)]-edict[str(arr_before)]
    if mc_choice(dE,T):
        s[i] = new_val
    return s

def mc_step(s, edict, nei, T):
    """perform L*L flips for 1 MC step"""
    for _ in range(L*L):
        s = step(s,edict,nei,T)
    return s

################################################################################

def calc_e(state):
    s = state.reshape(L,L)
    e = 0
    for i in range(-1,L-1):
        for j in range(-1,L-1):
            e += kron(s[i,j], s[i+1,j]) # right neighbour
            e += kron(s[i,j], s[i,j+1]) # down neighbour
    return -e     # e = -J*qi*qj

def calc_m(state):
    s = state.reshape(L,L)
    m_vect = np.array([np.count_nonzero(s == i) for i in range(q)])
    return (max(m_vect)*q/L**2-1)/(q-1)  #Numerical revision of the ... two-dimensional 4-state Potts model (15)

################################################################################

def model_p4(T,N_avg=10,N_mc=10,Relax=10):
    """Моделируем АТ"""
    E, M = [], []
    edict = gen_e_dict()
    state = gen_state()
    nei = get_neigh()
    
    #relax $Relax times be4 AVG
    for __ in range(Relax):
            state = mc_step(state, edict, nei, T)
    #AVG every $N_mc steps
    for _ in range(N_avg):
        for __ in range(N_mc):
            state = mc_step(state, edict, nei, T)
        E += [calc_e(state)]
        M += [calc_m(state)]
    
    return E, M


if __name__=="__main__":
    global L, q, J
    L = 10
    q = 4      # components
    J = 1      # interaction energy
    N_avg = 10
    N_mc = 1
    Relax = 10

    tc = 1/(np.log(1+4**0.5)) # 0.9102392266268373
    t_ = np.array([0.002, 0.005])
    t_low = np.round(-t_*tc+tc, 3)  #low
    t_high = np.round(t_*tc+tc, 3)
    t = np.concatenate((t_low, t_high), axis=None)
    t.sort()
        
    df_e,df_m =[pd.DataFrame() for i in range(2)]
    st = time.time()
    for ind,T in enumerate(t):
        e,m = model_p4(T,N_avg,N_mc,Relax)
        df_e.insert(ind,T,e, True)
        df_m.insert(ind,T,m, True)
    title = 'potts4_L'+str(L)+'_avg'+str(N_avg)+'_mc'+str(N_mc)+'_relax'+str(Relax)+'mc_'
    df_e.to_csv('export/e_'+title+'seed'+str(sys.argv[1])+'.csv', index = None, header=True)
    df_m.to_csv('export/m_'+title+'seed'+str(sys.argv[1])+'.csv', index = None, header=True)
    print('im done in ',time.time()-st)