import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

def add_neighbor(graph_dict,neuron_id,neighbor_n_id,neighbor_i_id,modifiable=1,weight=1.0):
    """
    Adding a neighbor to the graph dictionary
    
    graph_dict - graph dictionary, the keys are the neurons and the values are lists
    neuron_id - from this neuron
    neighbor_n_id - to this neuron
    neighbor_i_id - to this input
    weight = fix weight
    modifiable = if 0, then immutable
    """
    if neuron_id in graph_dict.keys():
        neighbor = None
        if modifiable > 0:
            neighbor = (neighbor_n_id,neighbor_i_id,modifiable)
        else:
            neighbor = (neighbor_n_id,neighbor_i_id,0,weight)
        
        graph_dict[neuron_id].append(neighbor)
    else:
        neighbor = None
        if modifiable > 0:
            neighbor = (neighbor_n_id,neighbor_i_id,modifiable)
        else:
            neighbor = (neighbor_n_id,neighbor_i_id,0,weight)
        
        graph_dict[neuron_id] = [ neighbor ]
    return graph_dict

def add_input(activations_dict,neuron_id,input_id,act_type,modifiable=1,weight=0.0):
    """
    Adding an input to the activations dictionary
    
    activations_dict - activations dictionary, the keys are the neurons and the values are dictionaries with the inputs
    neuron_id - the neuron
    input_id - its input
    act_type - activation type (number), we convert it to string
    weight = fix weight
    modifiable = if 0, then immutable
    """
    act_type = str(act_type)
    if neuron_id in activations_dict.keys():
        if modifiable == 1:
            activations_dict[neuron_id][input_id] = [ act_type, 1 ]
        else:
            activations_dict[neuron_id][input_id] = [ act_type, 0, weight ]
    else:
        if modifiable == 1:
            activations_dict[neuron_id] = {input_id : [act_type, 1]}
        else:
            activations_dict[neuron_id] = {input_id : [act_type, 0, weight]}
    return activations_dict

def add_neuron(graph_dict,neuron_id):
    """
    Adding neuron to graph dictionary
    
    graph_dict - graph dictionary, the keys are the neurons and the values are lists
    neuron_id - the neuron
    
    """
    if neuron_id not in graph_dict.keys():
        graph_dict[neuron_id] = [ ]
    return graph_dict

def distance(v1,v2):
    """
    Calculates the euclidean distance of v1 and v2 vectors
    """
    return np.sqrt(np.sum((v1-v2)**2))


def generate_full_graph(G, INPUT_NUM, RES_DIRECTED, FF, P_IN, P_OUT, CNAME, ACT_TYPE, SEED, pos, DRAW=True):
    """
    Generate neural network topology files based on a NetworkX graph structure.

    Parameters:
    -----------
    G : networkx.Graph
        NetworkX graph representing the reservoir structure
    INPUT_NUM : int
        Number of input neurons
    RES_DIRECTED : str
        Direction type: "directed" or "undirected"
    FF : bool
        Feedforward flag (True/False)
    P_IN : float
        Probability of input-to-reservoir connections (0.0-1.0)
    P_OUT : float
        Probability of reservoir-to-output connections (0.0-1.0)
    CNAME : str
        Configuration name for output files
    ACT_TYPE : int
        Activation function type
    SEED : int
        Random seed for reproducibility
    pos : dict
        Node positions for graph visualization (from nx.spring_layout or similar)
    DRAW : bool, optional
        Whether to generate a graph visualization PNG file (default: True)

    Returns:
    --------
    dict
        Dictionary containing paths to generated files and statistics:
        - 'graph': Network connectivity structure file
        - 'logic': Trainable parameter flags file
        - 'fixwb': Fixed weights and biases file
        - 'shared_w': Weight sharing groups file
        - 'shared_b': Bias sharing groups file
        - 'graph_viz': Graph visualization PNG file (if DRAW=True)
        - 'trainable_parameters': Number of trainable parameters in the network
    """
    # Debug: Display input parameters
    print("\n" + "="*60)
    print("GENERATE_FULL_GRAPH - Input Parameters")
    print("="*60)
    print(f"G: NetworkX graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"INPUT_NUM: {INPUT_NUM}")
    print(f"RES_DIRECTED: {RES_DIRECTED}")
    print(f"FF: {FF}")
    print(f"P_IN: {P_IN}")
    print(f"P_OUT: {P_OUT}")
    print(f"CNAME: {CNAME}")
    print(f"ACT_TYPE: {ACT_TYPE}")
    print(f"SEED: {SEED}")
    print(f"DRAW: {DRAW}")
    print("="*60 + "\n")

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)

    # Validate parameters
    if RES_DIRECTED == "undirected" and FF == True:
        raise ValueError("FF cannot be True when RES_DIRECTED is 'undirected'")

    # Calculate derived parameters
    RES_NUM = G.number_of_nodes()
    output_num = INPUT_NUM

    # Initialize data structures
    shared_bias_groups = []
    shared_weight_groups = []
    graph = {}
    activations = {}

    ###################################
    # Creating the network dictionary #
    ###################################

    # Input layer
    for j in range(INPUT_NUM):
        activations = add_input(activations, neuron_id=j + 1, input_id=1, act_type=0, modifiable=1)
        graph = add_neuron(graph, neuron_id=j + 1)

    # Reservoir layer
    for j in range(RES_NUM):
        activations = add_input(activations, neuron_id=j + 1 + INPUT_NUM, input_id=1, act_type=ACT_TYPE, modifiable=1)
        graph = add_neuron(graph, neuron_id=j + 1 + INPUT_NUM)

    # Output layer
    for j in range(output_num):
        activations = add_input(activations, neuron_id=j + 1 + INPUT_NUM + RES_NUM, input_id=1, act_type=0, modifiable=1)
        graph = add_neuron(graph, neuron_id=j + 1 + INPUT_NUM + RES_NUM)

    print(f"Number of nodes in G graph: {G.number_of_nodes()}")
    edges = list(G.edges)

    # Reservoir to reservoir connections
    for edge in edges:
        si = edge[0] + INPUT_NUM + 1
        ei = edge[1] + INPUT_NUM + 1

        if RES_DIRECTED == "directed":
            r = random.random()
            if FF == False:
                if r > 0.5:
                    graph = add_neighbor(graph, neuron_id=si, neighbor_n_id=ei, neighbor_i_id=1, modifiable=1)
                else:
                    graph = add_neighbor(graph, neuron_id=ei, neighbor_n_id=si, neighbor_i_id=1, modifiable=1)
            else:
                if ei > si:
                    graph = add_neighbor(graph, neuron_id=si, neighbor_n_id=ei, neighbor_i_id=1, modifiable=1)
                else:
                    graph = add_neighbor(graph, neuron_id=ei, neighbor_n_id=si, neighbor_i_id=1, modifiable=1)

        if RES_DIRECTED == "undirected":
            shared_group = []

            graph = add_neighbor(graph, neuron_id=si, neighbor_n_id=ei, neighbor_i_id=1, modifiable=1)
            graph = add_neighbor(graph, neuron_id=ei, neighbor_n_id=si, neighbor_i_id=1, modifiable=1)

            shared_group.append((si, ei, 1))
            shared_group.append((ei, si, 1))

            if len(shared_group) > 0:
                shared_weight_groups.append(shared_group)

    # Reservoir to output connections
    for j in range(RES_NUM):
        for k in range(output_num):
            r = random.random()
            if r < P_OUT:
                si = INPUT_NUM + j + 1
                ei = k + 1 + INPUT_NUM + RES_NUM
                graph = add_neighbor(graph, neuron_id=si, neighbor_n_id=ei, neighbor_i_id=1, modifiable=1)

    # Input to reservoir connections
    for i in range(INPUT_NUM):
        for j in range(RES_NUM):
            r = random.random()
            if r < P_IN:
                si = i + 1
                ei = INPUT_NUM + j + 1
                graph = add_neighbor(graph, neuron_id=si, neighbor_n_id=ei, neighbor_i_id=1, modifiable=1)

    ######################
    # Saving the network #
    ######################

    # Creating the shared weights file
    print(f"Shared weight groups: {len(shared_weight_groups)}")
    to_file_shared_w = []
    for group in shared_weight_groups:
        line = str(len(group)) + " ### "
        for weight in group:
            line += f"{weight[0]} {weight[1]} {weight[2]}; "
        to_file_shared_w.append(line)

    # Creating the shared bias file
    print(f"Shared bias groups: {len(shared_bias_groups)}")
    to_file_shared_b = []
    for group in shared_bias_groups:
        line = str(len(group)) + " ### "
        for bias in group:
            line += f"{bias[0]} {bias[1]}; "
        to_file_shared_b.append(line)

    # Converting to list
    to_file_graph = []
    to_file_logic = []
    to_file_fixwb = []

    for line_ind in sorted(graph):
        neighbors = graph[line_ind]
        line_graph = str(len(neighbors)) + " ### "
        line_fixwb = ""
        line_logic = ""

        # neighbors
        for neighbor in neighbors:
            line_graph += f"{neighbor[0]} {neighbor[1]}; "
            logic_switch = int(neighbor[2])
            line_logic += f"{logic_switch} "

            if neighbor[2] == 0:
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

            if activation[1] == False:
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
    
    RD = "ud" if RES_DIRECTED == "undirected" else "di"

    # Saving to files
    graph_datas = f"graph_{CNAME}_FF{FF}_RD{RD}_R{RES_NUM}_I{INPUT_NUM}_PI{int(100*P_IN)}_PO{int(100*P_OUT)}.dat"
    f = open(graph_datas, "w")
    for line in to_file_graph:
        f.write(line + "\n")
    f.close()

    logic_datas = f"logic_{CNAME}_FF{FF}_RD{RD}_R{RES_NUM}_I{INPUT_NUM}_PI{int(100*P_IN)}_PO{int(100*P_OUT)}.dat"
    f = open(logic_datas, "w")
    for line in to_file_logic:
        f.write(line + "\n")
    f.close()

    fixwb_datas = f"fixwb_{CNAME}_FF{FF}_RD{RD}_R{RES_NUM}_I{INPUT_NUM}_PI{int(100*P_IN)}_PO{int(100*P_OUT)}.dat"
    f = open(fixwb_datas, "w")
    for line in to_file_fixwb:
        f.write(line + "\n")
    f.close()

    shared_w_datas = f"shared_w_{CNAME}_FF{FF}_RD{RD}_R{RES_NUM}_I{INPUT_NUM}_PI{int(100*P_IN)}_PO{int(100*P_OUT)}.dat"
    f = open(shared_w_datas, "w")
    for line in to_file_shared_w:
        f.write(line + "\n")
    f.close()

    shared_b_datas = f"shared_b_{CNAME}_FF{FF}_RD{RD}_R{RES_NUM}_I{INPUT_NUM}_PI{int(100*P_IN)}_PO{int(100*P_OUT)}.dat"
    f = open(shared_b_datas, "w")
    for line in to_file_shared_b:
        f.write(line + "\n")
    f.close()

    print(f"Id: *_{CNAME}_FF{FF}_RD{RD}_R{RES_NUM}_I{INPUT_NUM}_PI{int(100*P_IN)}_PO{int(100*P_OUT)}.dat")

    graph_image_file = None
    if DRAW:
        print("Drawing the graph...")
        fig = plt.figure(figsize=(10, 8))
        # Use the passed pos parameter instead of generating a new layout
        nx.draw(G, pos, with_labels=False, node_size=30, alpha=0.7, edge_color='gray')
        plt.title(f"Graph: {CNAME}")

        # Save figure to file instead of showing
        graph_image_file = f"graph_viz_{CNAME}_FF{FF}_RD{RD}_R{RES_NUM}_I{INPUT_NUM}_PI{int(100*P_IN)}_PO{int(100*P_OUT)}.png"
        plt.savefig(graph_image_file, dpi=150, bbox_inches='tight')
        print(f"Graph visualization saved to: {graph_image_file}")
        plt.close(fig)



    # Return file paths and trainable parameters
    return {
        'graph': graph_datas,
        'logic': logic_datas,
        'fixwb': fixwb_datas,
        'shared_w': shared_w_datas,
        'shared_b': shared_b_datas,
        'graph_viz': graph_image_file,
        'trainable_parameters': trainable_parameters
    }















