#!/usr/bin/env python
# coding: utf-8

# In[2]:


#-------------------------------------------------- Data Extraction ------------------------------------------------------------

from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time


clinic_data_df = pd.read_excel("Data.xlsx", sheet_name='ClinicData',index_col=0)
demands_df = pd.read_excel("Data.xlsx", sheet_name='Demands',index_col=0)
locations_df = pd.read_excel("Data.xlsx", sheet_name='Locations', index_col=0, usecols=[0, 1])
distance_df = pd.read_excel("Data.xlsx", sheet_name='Distance_between_all_nodes (km)',index_col=0)

#Sets
S_indx = ["s1","s2","s3"]
K_indx = ["k1","k2","k3","k4","k5","k6"]

K_s_indx = {
    "s1": ["k2","k3","k4","k5","k6"],
    "s2": ["k2","k3","k4","k5","k6"],
    "s3": ["k1","k3","k4","k5","k6"]
}

T_indx = list(range(1, 29))
W_indx = list(range(1, 5))

J_indx = demands_df.iloc[0:14, 0].tolist()
L_indx = locations_df.iloc[:, 0].tolist()
D_indx = clinic_data_df.iloc[0:7, 4].tolist()
union_J_and_D = list(set(D_indx).union(set(J_indx)))

#Parameters

#d_ij
d_ij = distance_df.loc[L_indx, union_J_and_D]

#l_ij
l_js = demands_df.loc[1:14, S_indx]
l_js.index = J_indx

#w_js
columns_name = ['s1_demand', 's2_demand', 's3_demand']
w_js = demands_df.loc[1:14, columns_name]
new_column_names = ['s1', 's2', 's3']
w_js.columns = new_column_names
w_js.index = J_indx

#e_js
columns_name = ['e_s1', 'e_s2', 'e_s3']
e_js = demands_df.loc[1:14, columns_name]
new_column_names = ['s1', 's2', 's3']
e_js.columns = new_column_names
e_js.index = J_indx

#a_ijs
index = pd.MultiIndex.from_product([L_indx, J_indx, S_indx], names=['Location', 'Demand', 'Service'])
a_ijs = pd.DataFrame(0, index=index, columns=['Value'])
for loc in L_indx:
    for demand in J_indx:
        for service in S_indx:
                    if d_ij.loc[loc, demand] <= l_js.loc[demand, service]:
                        a_ijs.loc[(loc, demand, service), 'Value'] = 1
                        
#o_k
o_k = clinic_data_df.loc[0:6, 'DepotLocation']
o_k.index = K_indx

#v_k
v_k = clinic_data_df.loc[1:6, 'maxNumberOfWorkingDaysInAWeek']
v_k.index = K_indx

#t_w
t_w = [1,8,15,22]

#c
c = 100

#a_s
a_s = pd.DataFrame([0.2,0.2,0.2], index=['s1', 's2', 's3'])
a_s.columns = ["minServiceLevelPerServiceType"]

print("Perod: (T_indx) \n" , T_indx)
print("------------------------------------------")
print("Service Type: (S_indx)\n" , S_indx)
print("------------------------------------------")
print("Eligibale Mobile Clinics: (K_s_indx) \n" , K_s_indx)
print("------------------------------------------")
print("Mobile Clinics: (K_indx) \n" , K_indx)
print("------------------------------------------")
print("Weeks: (W_indx)\n" , W_indx)
print("------------------------------------------")
print("Demand Points: (J_indx) \n" , J_indx)
print("------------------------------------------")
print("Potential Visit Locations: (L_indx)\n" , L_indx)
print("------------------------------------------")
print("Depot Locations: (D_indx)\n" , D_indx)
print("------------------------------------------")
print("o_k: (D_indx)\n" , o_k)


# In[3]:


#-------------------------------------------------- Mathematical Model ---------------------------------------------------------
time_limit = 600 
results = []
total_start_time = time.time()

for epsilon_2 in range(1,7,1):
    for epsilon_3 in range(25,401,25):
                loop_start_time = time.time()
                model = ConcreteModel()

                #Variables
                model.X = Var(L_indx,K_indx,T_indx, within = Binary)
                X = model.X
                model.Z = Var(J_indx,S_indx, within = Binary)
                Z = model.Z
                model.F = Var(J_indx,S_indx,T_indx, within = Binary)
                F = model.F
                model.V = Var(K_indx,T_indx, within = Binary)
                V = model.V
                model.U = Var(K_indx , within = Binary)
                U = model.U

                #Constraint1
                model.con1 = ConstraintList()
                for t in T_indx:
                    for k in K_indx:
                        for i in L_indx:
                            model.con1.add(d_ij.at[i, o_k.loc[k]]*X[i,k,t] <= c)

                #Constraint2
                model.con2 = ConstraintList()   
                for t in T_indx:
                    for k in K_indx:
                        model.con2.add(sum(X[i,k,t] for i in L_indx) <= V[k,t])

                #Constraint3
                model.con3 = ConstraintList() 
                for i in L_indx:
                    for j in J_indx:
                        for s in S_indx:
                            if a_ijs.loc[(i, j, s), 'Value'] == 1:
                                    for k in K_s_indx[s]:
                                        for t in T_indx:
                                               model.con3.add(a_ijs.loc[(i, j, s), 'Value']*X[i, k, t] <= F[j, s, t])

                #Constraint4
                model.con4 = ConstraintList() 
                for t in T_indx:
                    for s in S_indx:
                        for j in J_indx:
                            model.con4.add(F[j,s,t] <= sum(a_ijs.loc[(i, j, s), 'Value']*X[i,k,t] 
                                                           for k in K_s_indx[s] for i in L_indx))

                #Constraint5
                model.con5 = ConstraintList()
                for j in J_indx:
                    for s in S_indx:
                        for t in T_indx:
                            if s == "s1":
                                t_end = t + 20 - 1
                            elif s == "s2":
                                t_end = t + 10 - 1
                            else: 
                                t_end = t + 10 - 1
                            if t_end <= max(T_indx):
                                model.con5.add(sum(F[j, s, t_prime] for t_prime in range(t, t_end + 1)) >= Z[j, s])

                #Constraint7
                model.con7 = ConstraintList()
                for k in K_indx:
                    for w in W_indx:
                        model.con7.add(sum(V[k,t_prime] for t_prime in range(t_w[w-1],t_w[w-1]+7)) <= v_k.loc[k])

                #Constraint8
                model.con8 = ConstraintList()
                for k in K_indx:
                    for t in T_indx:
                        model.con8.add(V[k,t] <= U[k])

                #Additional Constraint1        
                model.additional_con1 = Constraint(expr = sum(U[k] for k in K_indx) <= epsilon_2)

                #Additional Constraint2
                model.additional_con2 = Constraint(expr = sum( 
                    2*d_ij.loc[i, o_k.at[k]]*X[i, k, t] 
                     for k in K_indx for i in L_indx for t in T_indx) <= epsilon_3)        

                #Objective Function
                model.obj = Objective(expr = sum(w_js.loc[j,s]*Z[j,s] for j in J_indx for s in S_indx), sense = maximize)

                #Solve
                solver = SolverFactory('gurobi')
                solver.solve(model, timelimit=time_limit)
                
                loop_end_time = time.time() 
                loop_duration = loop_end_time - loop_start_time
                
                #model.pprint()
                obj_value = value(model.obj)
                con1_value = value(sum(value(U[k]) for k in K_indx))
                con2_value = sum( 
                    2*d_ij.loc[i, o_k.at[k]]*value(X[i, k, t]) 
                     for k in K_indx for i in L_indx for t in T_indx)
                results.append((obj_value, con1_value, con2_value))
                
                print("epsilon_2: " , epsilon_2 , "epsilon_3" , epsilon_3)
                print("obj_value" , obj_value)
                print("con1_value" , con1_value)
                print("con2_value" , con2_value)
                print(f"Loop runtime: {loop_duration:.2f} seconds")
                print("-------------------------------------- \n")


total_end_time = time.time() 
total_duration = total_end_time - total_start_time
print(f"Total runtime for all loops: {total_duration:.2f} seconds")


results_df = pd.DataFrame(results, columns=['obj', 'additional_con1', 'additional_con2'])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
trisurf = ax.plot_trisurf(results_df['obj'], results_df['additional_con1'], 
                          results_df['additional_con2'], cmap='viridis', edgecolor='none')
ax.set_xlabel('Objective1', fontsize=10)
ax.set_ylabel('Objective2', fontsize=10)
ax.set_zlabel('Objective3', fontsize=10)
fig.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
plt.tight_layout(pad=5)
cbar = fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Color Intensity', fontsize=12)

plt.show()


# In[ ]:




