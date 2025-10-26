import numpy as np
from utils import *
import argparse
import random
import matplotlib.pyplot as plt
import networkx as nx

P_IN = 0.2
P_OUT = 0.2

RES_NUM = G.number_of_nodes() # ez valojaban G-bol jon
res_m = 3
fitness_distribution = 'uniform'

INPUT_NUM = 85
output_num = INPUT_NUM

ACT_TYPE = 9
SEED = 2023
CNAME = "f_v0"

random.seed(SEED)
np.random.seed(SEED)

RES_DIRECTION = "directed"#/"directed"
FF = True

if RES_DIRECTION == "undirected" and FF == True:
    print("FF cannot be undirected")
    exit()

DRAW = False#True

###################################
# Creating the network dictionary #
###################################

shared_bias_groups = []
shared_weight_groups = []

graph = {}
activations = {}


# inputs
for j in range(INPUT_NUM):    
    activations = add_input(activations, neuron_id = j + 1, input_id = 1, act_type = 0, modifiable = 1)
    graph = add_neuron(graph, neuron_id = j + 1)
    
# reservoir
for j in range(RES_NUM):
    activations = add_input(activations, neuron_id = j + 1 + INPUT_NUM, input_id = 1, act_type = ACT_TYPE, modifiable = 1)
    graph = add_neuron(graph, neuron_id = j + 1 + INPUT_NUM)
        

# outputs
for j in range(output_num):
    activations = add_input(activations, neuron_id = j + 1 + INPUT_NUM + RES_NUM, input_id = 1, act_type = 0, modifiable = 1)
    graph = add_neuron(graph, neuron_id = j + 1 + INPUT_NUM + RES_NUM)
       
print(f"Number of nodes in G graph: {G.number_of_nodes()}")
edges = list(G.edges)


           
# res -> res
for edge in edges:
    si = edge[0] + INPUT_NUM + 1
    ei = edge[1] + INPUT_NUM + 1
    
    if RES_DIRECTION == "directed":
        r = random.random()
        if FF == False:
            if r > 0.5:
                graph = add_neighbor(graph,neuron_id = si, neighbor_n_id = ei, neighbor_i_id=1, modifiable=1)
            else:
                graph = add_neighbor(graph,neuron_id = ei, neighbor_n_id = si, neighbor_i_id=1, modifiable=1)
        else:
            if ei > si:
                graph = add_neighbor(graph,neuron_id = si, neighbor_n_id = ei, neighbor_i_id=1, modifiable=1)
            else:
                graph = add_neighbor(graph,neuron_id = ei, neighbor_n_id = si, neighbor_i_id=1, modifiable=1)
            
    if RES_DIRECTION == "undirected":
        shared_group = []            

        graph = add_neighbor(graph,neuron_id = si, neighbor_n_id = ei, neighbor_i_id=1, modifiable=1)
        graph = add_neighbor(graph,neuron_id = ei, neighbor_n_id = si, neighbor_i_id=1, modifiable=1)
    
        shared_group.append((si, ei, 1))
        shared_group.append((ei, si, 1))
                  
        if len(shared_group)>0:
            shared_weight_groups.append(shared_group)
    
# res -> output
for j in range(RES_NUM):
    for k in range(output_num):

        r = random.random()
        if r < P_OUT:
            si = INPUT_NUM + j + 1
            ei = k + 1 + INPUT_NUM + RES_NUM
            graph = add_neighbor(graph,neuron_id = si, neighbor_n_id = ei, neighbor_i_id = 1, modifiable = 1)
    

# inp -> res edges
for i in range(INPUT_NUM):   
    for j in range(RES_NUM):

        
        r = random.random()
        if r < P_IN:
        
            si = i + 1
            ei = INPUT_NUM + j + 1
            
            graph = add_neighbor(graph,neuron_id = si, neighbor_n_id = ei, neighbor_i_id = 1, modifiable = 1)


######################
# Saving the network #
######################

# Creating the shared weights file
print(f"Shared weight groups: {len(shared_weight_groups)}")
to_file_shared_w = []
for group in shared_weight_groups:
    line = str(len(group))+" ### "
    for weight in group:
        line += f"{weight[0]} {weight[1]} {weight[2]}; "
    to_file_shared_w.append(line)

# Creating the shared bias file
print(f"Shared bias groups: {len(shared_bias_groups)}")
to_file_shared_b = []
for group in shared_bias_groups:
    line = str(len(group))+" ### "
    for bias in group:
        line += f"{bias[0]} {bias[1]}; "
    to_file_shared_b.append(line)

# Converting to list
to_file_graph = []
to_file_logic = []
to_file_fixwb = []

for line_ind in sorted(graph):
    neighbors = graph[line_ind]
    line_graph = str(len(neighbors))+ " ### "
    line_fixwb = ""
    line_logic = ""
    
    # neighbors
    for neighbor in neighbors:
        line_graph += f"{neighbor[0]} {neighbor[1]}; "
        logic_switch = int(neighbor[2])
        line_logic += f"{logic_switch} "
        
        if (neighbor[2]==0):
            line_fixwb += f"{neighbor[3]} "
            
    # activations
    line_graph += "### "
    line_logic += "### "
    line_fixwb += "### "
    activations_neuron = activations[line_ind]
    
    line_graph += str(len(activations_neuron)) + " ### "
    for activation_id in sorted(activations_neuron):
        activation = activations_neuron[activation_id]
        line_graph += f"{activation[0]} "
        
        logic_switch = int(activation[1])
        line_logic += f"{logic_switch} "
        
        if (activation[1]==False):
            line_fixwb += f"{activation[2]} "
    
    to_file_graph.append(line_graph)
    to_file_logic.append(line_logic)
    to_file_fixwb.append(line_fixwb)
    
neuron_num = len(graph)
print(f"Input num: {INPUT_NUM}")
print(f"Output num: {output_num}")
print(f"Neuron num: {neuron_num}")

# Parameter reduction caused by the weight sharing (tied weights)
trainable_parameters = 0
for line in to_file_logic:
    trainable_parameters += line.count('1')

for group in shared_bias_groups:
    
    neuron_id = group[0][0]
    input_id = group[0][1]
    logic = activations[neuron_id][input_id][1]
    if logic == 1:
        trainable_parameters -= len(group) - 1
        
for group in shared_weight_groups:
    

    neuron_id = group[0][0]
    neighbor_id = group[0][1]
    neighbor_input_id = group[0][2]
    
    find = 0
    i = 0
    logic = 0
    while find == 0 and i < len(graph[neuron_id]):
        neighbor_id_temp = graph[neuron_id][i][0]
        neighbor_input_id_temp = graph[neuron_id][i][1]
        logic = graph[neuron_id][i][2]
        if neighbor_id_temp == neighbor_id and neighbor_input_id_temp == neighbor_input_id:
            find = 1
        i += 1
    
    if logic == 1:
        trainable_parameters -= len(group) - 1

print(f"Trainable parameters: {trainable_parameters}")

# Saving to files
graph_datas = f"graph_{CNAME}_FF{FF}_R{RES_NUM}_.dat"
f = open(graph_datas,"w")
for line in to_file_graph:
    f.write(line+"\n")
f.close()

logic_datas = f"logic_{CNAME}_FF{FF}_R{RES_NUM}_.dat"
f = open(logic_datas,"w")
for line in to_file_logic:
    f.write(line+"\n")
f.close()

fixwb_datas = f"fixwb_{CNAME}_FF{FF}_R{RES_NUM}_.dat"
f = open(fixwb_datas,"w")
for line in to_file_fixwb:
    f.write(line+"\n")
f.close()

shared_w_datas = f"shared_w_{CNAME}_FF{FF}_R{RES_NUM}_.dat"
f = open(shared_w_datas,"w")
for line in to_file_shared_w:
    f.write(line+"\n")
f.close()

shared_b_datas = f"shared_b_{CNAME}_FF{FF}_R{RES_NUM}_.dat"
f = open(shared_b_datas,"w")
for line in to_file_shared_b:
    f.write(line+"\n")
f.close()

print(f"Id: *_{CNAME}_FF{FF}_R{RES_NUM}_.dat")

##############
# Properties #
##############



print(160*"-"+"\n")
##############
# Plot       #
##############

if DRAW:
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)  # Rugalmas elrendezés a vizualizációhoz
    nx.draw(G, pos, with_labels=False, node_size=30, alpha=0.7, edge_color='gray')
    plt.title("Barabasi gráf")
    plt.show();

#input("Press the Enter key to continue: ")


