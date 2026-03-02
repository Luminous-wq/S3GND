import igraph as ig
import copy
import os
import random
import numpy as np
from Utils.utils import create_folder
from scipy.stats import zipfian
import networkx as nx
# --------------
# generate synthetic graph as igraph(GML)
# --------------
def generate_dataset(seed: int, keywords_per_vertex_num: int, all_keyword_num: int,
                      node_num: int, neighbor_num: int, add_edge_probability: float,
                      distribution:str):
    random.seed(seed)
    np.random.seed(seed)
    # 1. Generate a graph
    target_graph = nx.newman_watts_strogatz_graph(n=node_num, k=neighbor_num, p=add_edge_probability)
    # 1.1. Make sure the graph is connected
    while not nx.is_connected(target_graph):
        target_graph = nx.newman_watts_strogatz_graph(node_num, neighbor_num, add_edge_probability)
    # 2. Compute infos of graph
    # edges_num = target_graph.number_of_edges()
    average_degree = sum(
        d for v, d in nx.degree(target_graph)) / target_graph.number_of_nodes()
    print(target_graph)
    print('Average Degree: ', average_degree)
    print('Max Degree: ', max(d for v, d in nx.degree(target_graph)))
    print('Min Degree: ', min(d for v, d in nx.degree(target_graph)))

    # 2.1. Delete some edges until edge num equals node_num * neighbor_num / 2
    while target_graph.number_of_edges() > node_num * neighbor_num:
        edges = list(target_graph.edges)
        u, v = random.choice(edges)
        
        target_graph.remove_edge(u, v)
        if not nx.is_connected(target_graph):
            target_graph.add_edge(u, v)

    # Print infos
    # edges_num = target_graph.number_of_edges()
    average_degree = sum(
        d for v, d in nx.degree(target_graph)) / target_graph.number_of_nodes()
    print(target_graph)
    print('Average Degree: ', average_degree)
    print('Max Degree: ', max(d for v, d in nx.degree(target_graph)))
    print('Min Degree: ', min(d for v, d in nx.degree(target_graph)))
    
    if distribution == "zipf":
        a = 0.75 # 0.75 0.8
        ranks = np.arange(1, all_keyword_num + 1)
        weights = 1.0 / np.power(ranks, a)
        weights /= weights.sum()  # 归一化

    elif distribution == "gau":

        mean = all_keyword_num / 2
        stddev = all_keyword_num / 4  
        x = np.arange(0, all_keyword_num)
        weights = np.exp(-0.5 * ((x - mean) / stddev)**2)
        weights /= weights.sum()

    elif distribution == "uni":
        weights = np.ones(all_keyword_num) / all_keyword_num

    label_counter = [0 for _ in range(all_keyword_num)]

    for i, node in enumerate(target_graph):
        keyword_num = np.random.randint(max(keywords_per_vertex_num - 1, 1),
                                        keywords_per_vertex_num + 2)
        keywords = np.random.choice(
            np.arange(all_keyword_num), 
            size=min(keyword_num, all_keyword_num), 
            replace=False, 
            p=weights
        )
        for keyword in keywords:
            label_counter[keyword] += 1
        
        target_graph.nodes[node]['keywords'] = [int(x) for x in keywords]
        
        bv = 0
        for k in keywords:
            bv |= (1 << int(k))
        target_graph.nodes[node]['EK'] = str(bv)

    print([{i: label_counter[i]} for i in range(all_keyword_num)])

    for u, v in target_graph.edges():
        target_graph[u][v]['weight'] = float(np.random.uniform(0, 1))

    for node in target_graph.nodes():
        neighbor_data = []
        for neighbor in target_graph.neighbors(node):
            weight = target_graph[node][neighbor]['weight']
            neighbor_data.append((neighbor, weight))
        
        neighbor_data.sort(key=lambda x: x[1], reverse=True)
        
        nw_string = ",".join([f"{idx}:{round(w, 4)}" for idx, w in neighbor_data])
        target_graph.nodes[node]['NW'] = nw_string

    folder_name = os.path.join(
        "./Datasets",
        "precompute",
        "synthetic",
        "{}-{}-{}-{}".format(
            target_graph.number_of_nodes(),
            target_graph.number_of_edges(),
            all_keyword_num,
            keywords_per_vertex_num
        )
    )

    create_folder(folder_name)
    initial_directory = os.getcwd()
    os.chdir(folder_name)

    G = ig.Graph.from_networkx(target_graph)
    for node in G.vs:
        if "keywords" in node.attribute_names():
    
            if isinstance(node["keywords"], int):
                node["keywords"] = [node["keywords"]]

            elif not isinstance(node["keywords"], list):
                raise TypeError(f"Unexpected type for keywords: {type(node['keywords'])}")
    for node in G.vs:
        if "keywords" in node.attribute_names():
            # node["EK"] = ",".join(map(str, node["keywords"]))
            node["keywords"] = ",".join(map(str, node["keywords"]))
        # node["NW"] = node["NW_sorted"]
        # node["EK_bv"] = node["EK_bv"]
            # print(node["keywords"])
    # print(list(G.vs[0].neighbors()))
    # print(G.neighbors(G.vs[0]))
    G.write_gml('G-'+str(distribution)+'.gml')  
    print(folder_name, 'G-'+str(distribution)+'.gml', 'saved successfully!')
    os.chdir(initial_directory)

if __name__ == "__main__":
    # Set the parameters to generate dataset.
    seed = 2025

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=10000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=10000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=10000,  # 10K, 50K, 250K, 1M, 10M, 30M
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )
    
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=50000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=50000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=50000,  # 10K, 50K, 250K, 1M, 10M, 30M
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )

    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=250000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=250000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=250000,  # 10K, 50K, 250K, 1M, 10M, 30M
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )
    
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=1000000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=1000000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=1000000,  # 10K, 50K, 250K, 1M, 10M, 30M
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=10000000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=10000000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    generate_dataset(
        seed=seed,
        keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
        all_keyword_num=50,  # 10, 20, 50, 80
        node_num=10000000,  # 10K, 50K, 250K, 1M, 10M, 30M
        neighbor_num=5,
        # add_edge_probability=0.780132
        add_edge_probability=0.250132,
        distribution="zipf"
    )
    
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=30000000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="uni"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=30000000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="gau"
    # )
    # generate_dataset(
    #     seed=seed,
    #     keywords_per_vertex_num=3,  # 1, 2, 3, 4, 5
    #     all_keyword_num=50,  # 10, 20, 50, 80
    #     node_num=30000000,  # 10K, 50K, 250K, 1M, 10M, 30M
    #     neighbor_num=5,
    #     # add_edge_probability=0.780132
    #     add_edge_probability=0.250132,
    #     distribution="zipf"
    # )
