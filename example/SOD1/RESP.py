"""
@author: Shawn Hsueh (PhD candidate UBC Physics)
"""
import numpy as np
import re
from scipy.spatial.distance import cdist

def assign_identical(A, B, Identical_charges):
    A_tmp = A
    B_tmp = B
    a_tmp = a
    q0_tmp= q0
    redun_list = []
    for i in range(len(Identical_charges)): #each group of identical charges
        for j in range(1, len(Identical_charges[i])): #each atom in the group
            kept =  Identical_charges[i][0]
            redun = Identical_charges[i][j]
            A[kept,:] = A_tmp[kept,:]+A_tmp[redun,:]
            A[:,kept] = A_tmp[:,kept]+A_tmp[:,redun]
            B[kept] = B_tmp[kept]+B_tmp[redun]
            redun_list += [redun]
    
    redun_list.sort(reverse=True)
    for i in range(len(redun_list)):
        redun = redun_list[i]
        A = np.delete(A, redun, axis=0)
        A = np.delete(A, redun, axis=1)
        B = np.delete(B, redun, axis=0)
    return A, B
        
def build_ordered_q(q, N_atom, Identical_charges):
    q_correct = np.zeros(N_atom)
    Identical_charges_flat = [item for sublist in Identical_charges for item in sublist]
    #print(Identical_charges_flat)
    counter = 0
    is_continue = 0
    for i in range(N_atom):
        if q_correct[i] != 0:
            continue
        if i in Identical_charges_flat:
            #Find the list that i belong to in Identical_charges
            for j in range(len(Identical_charges)): #each group of identical charges
                for k in range(len(Identical_charges[j])): #each atom in the group
                    if Identical_charges[j][k]==i:
                        subgroup = Identical_charges[j]
            for index in subgroup:
                q_correct[index] = q[counter]
            counter += 1
        else:
            q_correct[i] = q[counter]
            counter += 1
    return q_correct

file_ESP_grid = 'ESP_grid.txt'
file_ESP = 'ESP.txt'
MaxIter = 10

#assign identical charges
Identical_charges = [[24,32],[26,34],[27,35],[28,36],[29,37],[30,38],[31,39],[25,33],[66,62],[14,51],[12,49],[13,50],[9,46],[10,47],[11,48],[61,60],[15,52],[16,53], [41,42,63]] # same index as in gaussian input
for i in range(len(Identical_charges)): #each group of identical charges
    Identical_charges[i].sort()
    for j in range(len(Identical_charges[i])): #each atom in the group
        Identical_charges[i][j] -= 1


with open(file_ESP_grid, 'r+') as f:
    ESP_grid = f.readlines()

R = np.zeros((len(ESP_grid),3))
N_atom = 0
for l in range(len(ESP_grid)):
    toks = re.split(r'[\s,]+', ESP_grid[l])
    toks = [ x for x in toks if x!="" ]
    if toks[0] == 'Atomic':
        N_atom += 1
    R[l] = [float(i) for i in toks[-3:]]


a = np.ones(N_atom)*0.001 #strength of the restraint
q0 = np.zeros(N_atom)     #bias center
r_grid = R[N_atom:] # position at each spatial grid
r_atom = R[:N_atom] # position at each atomic center

with open(file_ESP, 'r+') as f:
    ESP = f.readlines()
V = np.zeros(len(ESP))
for l in range(len(ESP)):
    toks = re.split(r'[\s,]+', ESP[l])
    toks = [ x for x in toks if x!="" ]
    V[l] = float(toks[-1])
V_grid = V[N_atom:]  # potential at each spatial grid


B = np.zeros((N_atom, 1))
A = np.zeros((N_atom, N_atom))
r = cdist(r_grid, r_atom)
r = r/0.529 # 1 hartree length unit=0.529A
for j in range(N_atom):
    B[j] = np.sum(V_grid/r[:, j])
    for k in range(N_atom):
        A[j,k] = np.sum(1/r[:,j] * 1/r[:,k])
        

q = np.linalg.solve(A, B) # solve Aq=B
Vhat_grid = np.sum(q.T/r, axis=1)
RRMS = (np.sum((V_grid-Vhat_grid)**2)/np.sum(V_grid**2))**0.5
print('ESP charges:')
print('RRMS=', RRMS)
print('q (ESP): ', q.flatten())


#A, B = assign_identical(A, B, Identical_charges)
#q = np.linalg.solve(A, B) # solve Aq=B
#q = build_ordered_q(q, N_atom, Identical_charges)
#print('ESP charges:')
#print('RRMS=', RRMS)
#print(q)

B = np.zeros((N_atom, 1))
A = np.zeros((N_atom, N_atom))
RRMS_prev = -1
for t in range(MaxIter):
    B = np.zeros((N_atom, 1))
    A = np.zeros((N_atom, N_atom))
    for j in range(N_atom):
        drestr_dq = 2*a[j]*(q0[j]-q[j])
        B[j] = np.sum(V_grid/r[:, j]) + a[j]*q0[j]
        for k in range(N_atom):
            A[j,k] = np.sum(1/r[:,j] * 1/r[:,k]) + (j==k)*a[j]
    A, B = assign_identical(A, B, Identical_charges)
    q = np.linalg.solve(A, B) # solve Aq=B
    q = build_ordered_q(q, N_atom, Identical_charges)
    Vhat_grid = np.sum(q.T/r, axis=1)
    RRMS = (np.sum((V_grid-Vhat_grid)**2)/np.sum(V_grid**2))**0.5
    if abs(RRMS - RRMS_prev)<0.00001:
        print('RESP charges:')
        print('RRMS=', RRMS)
        print('q (RESP): ', q)
        break
    RRMS_prev = RRMS


