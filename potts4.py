#Uses python3

import numpy as np
import time
import pandas as pd
import sys
# np.random.seed(int(sys.argv[1]))

#====================== all functions ======================
def kron(i,j):
    """Kroneker's symbol"""
    if i==j: return 1
    else: return 0

def coord(site):
    """get coordinate i of vector"""
    x = site // L
    y = site - x*L
    return (x,y)

def get(i):
    """fixin' boundary"""
    if i<0: return i
    else: return i % L
    
def get_neigh():
    """get neighbour's arr"""
    s = np.arange(L**2).reshape(L,L)
    nei = []
    for site in range(L*L):
        i,j = coord(site)
        nei += [s[get(i-1),get(j)],s[get(i),get(j+1)],s[get(i+1),get(j)],s[get(i),get(j-1)]]
    return np.array(nei).reshape(L*L,4)

#################################################################

def gen_state():
    """generate random start state with lenght L*L and q components"""
    state = np.random.randint(0, q, L*L)
    return state

def mc_choice(dE,T):
    """принимаем или не принимаем переворот спина?"""
    if dE <= 0:
        return True
    elif np.random.uniform() <= np.exp(-dE/T):
        return True
    else:
        return False

def calc_dE(old_val, new_val, neigh):
    """calculate dE"""
    e1 = np.count_nonzero(neigh == old_val)
    e0 = np.count_nonzero(neigh == new_val)
    return e1-e0
    
def step(s,nei,T):
    """крутим 1 спин"""
    i = np.random.randint(0, L*L)      ### выбираем случайный спин
    new_val = np.random.randint(q)     ### выбираем случайное значение
    
    neigh = s[nei[i,:]]                ### формируем конфигурацию соседей
    dE = calc_dE(s[i], new_val, neigh)
    if mc_choice(dE,T):
        s[i] = new_val
    return s

def mc_step(s,nei,T):
    """perform L*L flips for 1 MC step"""
    for _ in range(L*L):
        s = step(s,nei,T)
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

    state = gen_state()
    nei = get_neigh()
    
    #relax $Relax times be4 AVG
    for __ in range(Relax):
            state = mc_step(state, nei, T)
    #AVG every $N_mc steps
    for _ in range(N_avg):
        for __ in range(N_mc):
            state = mc_step(state, nei, T)
        E += [calc_e(state)]
        M += [calc_m(state)]
    
    return E, M


if __name__=="__main__":
    np.random.seed(1)
    global L, q, J
    L = 10
    q = 4      # components
    J = 1      # interaction energy
    N_avg = 10
    N_mc = 1
    Relax = 10

    tc = 1/(np.log(1+4**0.5)) # 0.9102392266268373
    t_ = np.array([0.002])
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
    # title = 'potts4_L'+str(L)+'_avg'+str(N_avg)+'_mc'+str(N_mc)+'_relax'+str(Relax)+'mc_'
    # df_e.to_csv('export/e_'+title+'seed'+str(sys.argv[1])+'.csv', index = None, header=True)
    # df_m.to_csv('export/m_'+title+'seed'+str(sys.argv[1])+'.csv', index = None, header=True)
    print(df_e[0.908].values.sum())    # просто численная проверка
    print('im done in ',time.time()-st)