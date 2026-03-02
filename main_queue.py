
import argparse
import itertools
import os
import pickle
import numpy as np
from argparser import args_parser
from information import Info
import time
import random
import igraph as ig
import logging
# from Utils.utils import info_file_save
from Utils.utils import generate_query, pruning_power, calculate_expected_pruned_count, info_file_save
import torch
from collections import deque

def make_seed(seed):
    seed = 2026
    random.seed(seed)
    np.random.seed(seed)

def generate_query(G:ig.Graph, n:int, p:float, keyword_domain):
    # subgraph
    subgraph = get_random_subgraph(G, n)
    query_graph = remove_edges_optimized(subgraph=subgraph, p=p)
    query_graph = sample_keywords_subset(query_graph, keywords_domain=keyword_domain)
    return query_graph

def get_random_subgraph(G: ig.Graph, n: int):
    while True:
        valid_nodes = [v.index for v in G.vs() if is_valid_keywords(v["keywords"])]
        
        if not valid_nodes:
            raise ValueError("No valid starting node found (all nodes have empty or '0' keywords).")
        
        start_node = random.choice(valid_nodes)
        selected_nodes = {start_node}

        while len(selected_nodes) < n:
            neighbors = set()
            for node in selected_nodes:
                for neighbor_idx in G.neighbors(node):
                    neighbor_node = G.vs[neighbor_idx]
                    if is_valid_keywords(neighbor_node["keywords"]):
                        neighbors.add(neighbor_idx)

            neighbors -= selected_nodes

            if not neighbors:
                print("Cannot expand the subgraph while maintaining connectivity.")
                break
            
            new_node = random.choice(list(neighbors))
            selected_nodes.add(new_node)

        if len(selected_nodes) == n:
            return G.subgraph(selected_nodes)
        else:
            continue

def is_valid_keywords(keywords_str):
    if not keywords_str or str(keywords_str).strip() == "0":
        return False
    parts = [k for k in str(keywords_str).split(",") if k.strip()]
    return len(parts) > 0

def remove_edges_optimized(subgraph: ig.Graph, p=0.3):

    if subgraph.ecount() == 0:
        return subgraph
    st = subgraph.spanning_tree(weights=None)
    st_edges = set()
    for e in st.es:
        st_edges.add(subgraph.get_eid(e.source, e.target))

    all_edge_indices = set(range(subgraph.ecount()))
    removable_edges = list(all_edge_indices - st_edges)

    to_delete = [e_idx for e_idx in removable_edges if random.random() < p]
    subgraph.delete_edges(to_delete)
    
    return subgraph

def sample_keywords_subset(subgraph: ig.Graph, keywords_domain):
    for node in subgraph.vs():
        raw_keywords = str(node["keywords"])
        keywords = [int(k) for k in raw_keywords.split(",") if k.strip()]
        
        if len(keywords) > 3:
            sampled_keywords = random.sample(keywords, 3)
            node['keywords'] = ",".join(map(str, sampled_keywords))
        elif len(keywords) > 0:
            node['keywords'] = ",".join(map(str, keywords))
        else:
            print(f"Warning: Node {node.index} has no keywords during sampling.")
            
    return subgraph

class IndexEntry:
    def __init__(self, key, index_N, idx):
        self.key = key  # GND lower bound
        self.index_N = index_N
        self.idx = idx

    def __gt__(self, other):
        if self.index_N["L"] == other.index_N["L"]:
            return self.key > other.key
        else:
            return self.index_N["L"] < other.index_N["L"]


def keyword_pruning(entry, query_vec_Key, E):
    node_vec_key = entry["Aux"]["EK"]
    # print(node_vec_key, query_vec_Key)
    if (node_vec_key & query_vec_Key) != query_vec_Key:
        return True

    node_kw_embeddings = []
    temp_node_vec = node_vec_key
    idx = 0
    # print(E.items())
    while temp_node_vec > 0:
        if temp_node_vec & 1:
            node_kw_embeddings.append(E[str(idx)])
        temp_node_vec >>= 1
        idx += 1
        
    if not node_kw_embeddings:
        return True

    node_kw_matrix = np.array(node_kw_embeddings)
    
    node_min = np.min(node_kw_matrix, axis=0)
    node_max = np.max(node_kw_matrix, axis=0)
    temp_query = query_vec_Key
    q_idx = 0
    while temp_query > 0:
        if temp_query & 1:
            query_embedding = E[str(q_idx)]
            if not np.all((query_embedding >= node_min) & (query_embedding <= node_max)):
                return True
        
        temp_query >>= 1
        q_idx += 1

    return False

def ND_lower_bound_pruning(lb_ND, sigma):
    if lb_ND > sigma:
        return True
    return False

def lb_ND_score(query_weight, node_weight):
    n = len(query_weight)
    m = len(node_weight)
    s = 0.0
    for i in range(n):
        q_w = query_weight[i][1]
        n_w = node_weight[i][1] if i < m else 0.0
        s += max(q_w-n_w, 0)
    return s

def GND_lower_pruning(query_weight, node_weight, sigma, function):
    s = getKey(query_weight, node_weight)
    if s > sigma:
        return True
    return False
    
def getKey(query_weight, node_weight):
    # print(query_weight)
    # print(node_weight)
    n = len(query_weight)
    m = len(node_weight)
    total_key = 0.0
    for i in range(n):
        q_w = query_weight[i]
        n_w = node_weight[i][1] if i < m else 0.0
        total_key += (q_w - n_w)
    return total_key

def refinment(G, q, V_cand, q_nodes, q_EK, q_W, theta, f):
    S = [] 
    q_size = len(q_nodes)

    q_adj = {q_nodes[i]: set() for i in range(q_size)}
    for edge in q.es:
        source = int(q.vs[edge.source]["id"])
        target = int(q.vs[edge.target]["id"])
        q_adj[source].add(target)
        q_adj[target].add(source)

    sorted_q_edges = []
    for edge in q.es:
        source = int(q.vs[edge.source]["id"])
        target = int(q.vs[edge.target]["id"])
        sorted_q_edges.append((source, target, edge["weight"]))
    sorted_q_edges.sort(key=lambda x: x[2], reverse=True)

    print("\n=== 开始优化的 Refinement 阶段 ===")
    print(f"查询节点: {q_nodes}")
    print(f"查询边数量: {len(sorted_q_edges)}")

    G_adj = {}
    for edge in G.es:
        u, v = edge.source, edge.target
        w = edge["weight"]
        if u not in G_adj:
            G_adj[u] = {}
        if v not in G_adj:
            G_adj[v] = {}
        G_adj[u][v] = w
        G_adj[v][u] = w

    def calculate_incremental_gnd(current_mapping, remaining_query_nodes, q_W_sorted):
        if not current_mapping:
            return 0.0
        matched_weights = []

        for q_u, q_v, q_w in q_W_sorted:
            if q_u in current_mapping and q_v in current_mapping:
                g_u = current_mapping[q_u]
                g_v = current_mapping[q_v]
                if g_u in G_adj and g_v in G_adj[g_u]:
                    matched_weights.append(G_adj[g_u][g_v])
                else:
                    matched_weights.append(0.0)

        remaining_edges = len(q_W_sorted) - len(matched_weights)
        optimal_remaining_weight = max(q_W_sorted, key=lambda x: x[2])[2] if q_W_sorted else 0

        current_gnd = 0.0
        for i, (q_u, q_v, q_w) in enumerate(q_W_sorted):
            if i < len(matched_weights):
                current_gnd += max(q_w - matched_weights[i], 0)
            else:
                current_gnd += max(q_w - optimal_remaining_weight, 0)

        return current_gnd

    def neighborhood_expansion(start_candidates, remaining_q_nodes, current_mapping, used_nodes):
        if not remaining_q_nodes:
            if len(current_mapping) == q_size:
                final_gnd = 0.0
                subgraph_edges = []
                for q_u, q_v, q_w in sorted_q_edges:
                    if q_u in current_mapping and q_v in current_mapping:
                        g_u = current_mapping[q_u]
                        g_v = current_mapping[q_v]
                        if g_u in G_adj and g_v in G_adj[g_u]:
                            subgraph_edges.append(G_adj[g_u][g_v])
                        else:
                            return 

                subgraph_edges.sort(reverse=True)
                q_weights = [w for _, _, w in sorted_q_edges]

                for i in range(len(q_weights)):
                    final_gnd += max(q_weights[i] - subgraph_edges[i], 0)

                if final_gnd <= theta:
                    S.append({
                        "mapping": current_mapping.copy(),
                        "gnd_score": final_gnd
                    })
            return

        lower_bound = calculate_incremental_gnd(current_mapping, remaining_q_nodes, sorted_q_edges)
        if lower_bound > theta:
            return  


        next_q_node = max(remaining_q_nodes, key=lambda x: len(q_adj[x]))

        candidates = V_cand[next_q_node]

        def get_priority_score(candidate):

            score = 0
            for mapped_q in current_mapping.keys():
                mapped_g = current_mapping[mapped_q]
                if mapped_q in q_adj[next_q_node]:
                    if mapped_g in G_adj and candidate in G_adj[mapped_g]:
                        score += G_adj[mapped_g][candidate]
                    else:
                        score -= 1000
            return score


        sorted_candidates = sorted(candidates, key=get_priority_score, reverse=True)


        top_k = min(len(sorted_candidates), 50)  
        for candidate in sorted_candidates[:top_k]:
            if candidate not in used_nodes:

                current_mapping[next_q_node] = candidate
                used_nodes.add(candidate)

                new_remaining = set(remaining_q_nodes)
                new_remaining.remove(next_q_node)

                neighborhood_expansion([], new_remaining, current_mapping, used_nodes)


                del current_mapping[next_q_node]
                used_nodes.remove(candidate)

    start_q_node = min(q_nodes, key=lambda x: len(V_cand[x]))
    start_candidates = V_cand[start_q_node]

    top_start_candidates = start_candidates[:min(len(start_candidates), 20)]

    for start_candidate in top_start_candidates:
        initial_mapping = {start_q_node: start_candidate}
        used_nodes = {start_candidate}
        remaining = set(q_nodes)
        remaining.remove(start_q_node)

        neighborhood_expansion([], remaining, initial_mapping, used_nodes)

    print(f"找到 {len(S)} 个满足条件的子图")

    for i, result in enumerate(S[:5]):
        print(f"\n子图 {i+1}:")
        print(f"  映射: {result['mapping']}")
        print(f"  GND得分: {result['gnd_score']:.4f}")

    return S

def S3GND(G, q, root, f, theta, E):
    q_nodes = []
    q_EK = {}
    q_NW = {}
    q_WT = q.es["weight"]
    # print(q_WT)
    q_W = sorted(q_WT, reverse=True)
    # print(q_W)
    for node in q.vs():
        id = int(node["id"])
        q_nodes.append(id)
        q_key = int(node['EK'])
        nw_raw = [pair.split(':') for pair in node['NW'].split(",") if pair]
        nw_list = [(int(p[0]), float(p[1])) for p in nw_raw]
        q_EK[id] = q_key
        q_NW[id] = nw_list
        # q_N = list(q.neighbors(node))
    q_size = len(q_nodes)
    # print(f"{q_nodes}; \n{q_EK}; \n{q_NW}")
    
    V_cand = {q_node:[] for q_node in q_nodes}
    S = []
    
    H = deque()
    entry_node_visit_counter = 0
    entry_pruning_counter = 0
    for idx, entry in enumerate(root):
        entry_node_visit_counter += 1
        entry["Q"] = []
        for q_j in q_nodes:
            if not keyword_pruning(entry, q_EK[q_j], E):
                lb_ND = lb_ND_score(q_NW[q_j], entry["Aux"]["NW"])
                entry["Q"].append((q_j, lb_ND))
        if entry["Q"]:
            # print(entry["Q"])
            H.append(entry)
        else:
            entry_pruning_counter += 1
    # print("entry_node_visit_counter: {}".format(entry_node_visit_counter))
    while H:
        now_entry = H.popleft()
        
        if now_entry["T"]:
            I_NW = now_entry["Aux"]["NW"]
            I_KE = now_entry["Aux"]["EK"]
            IND = []
            for q_j, lb_ND in now_entry["Q"]:
                q_EK_i = q_EK[q_j]
                has_match = True
                if (now_entry["Aux"]["EK"] & q_EK_i) != q_EK_i:
                    has_match = False
                if has_match:
                    IND.append((q_j,lb_ND))
            for node_index, lb_ND in IND:
                if not keyword_pruning(now_entry, q_EK[node_index], E):
                    if not ND_lower_bound_pruning(lb_ND, theta):
                        V_cand[node_index].append(now_entry["P"])
        else:
            for child_entry in now_entry["P"]:
                child_entry["Q"] = []
                for q_j, _ in now_entry["Q"]:
                    if not keyword_pruning(child_entry, q_EK[q_j], E):
                        lb_ND = lb_ND_score(q_NW[q_j], child_entry["Aux"]["NW"])
                        if not ND_lower_bound_pruning(lb_ND, theta):
                            child_entry["Q"].append((q_j, lb_ND))
                if child_entry["Q"]:
                    H.append(child_entry)
    # print("V_cand : {}".format(V_cand))
    cand_lengths = {key: len(value) for key, value in V_cand.items()}
    print("vertex candidate length: {}".format(cand_lengths))
    
    total_candidates = sum(len(candidates) for candidates in V_cand.values())
    num_nodes_in_G = G.vcount()  # 原图总节点数
    
    max_search_space = q_size * num_nodes_in_G
    pruning_rate = 1.0 - (total_candidates / max_search_space)
    # S = []
    # print(f"剪枝率: {pruning_rate:.4%}")
    # return S, pruning_rate

    
    refinement_start = time.time()
    S = refinment(G=G, q=q, V_cand=V_cand, q_nodes=q_nodes, q_EK=q_EK, q_W=q_W, theta=theta, f=f)
    refinement_time = time.time() - refinement_start

    print(f"\nRefinement 时间: {refinement_time:.4f} 秒")
    print(f"找到 {len(S)} 个满足条件的子图")
    return S, pruning_rate

if __name__ == "__main__":
    # python main_queue.py -i ./Datasets/precompute/synthetic/50000-124812-50-3/G-uni.gml -d 5WUni -r ./Results/graph_index_tree_5WUni.pkl -E ./Results/hgnn_keyword_embeddings_5WUni.pt
    # python main_queue.py -i ./Datasets/precompute/synthetic/10000-24979-50-3/G-uni.gml -d 1WUni -r ./Results/graph_index_tree_1WUni.pkl -E ./Results/hgnn_keyword_embeddings_1WUni.pt
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="path of graph input file",
                        default="./Datasets/precompute/synthetic/10000-24979-50-3/G-uni.gml")
    parser.add_argument("-o", "--output", type=str, help="path of the output file",
                        default="./Results")
    parser.add_argument("-qs", "--querySize", type=int, 
                        help="the query vertex set size", 
                        default=5)
    parser.add_argument("-s", "--keywordDomain", type=int, 
                        help="the keyword domain size", 
                        default=50)
    parser.add_argument("-d", "--dataset", type=str, 
                        default="syn1w-uni")
    parser.add_argument("-r", "--index", type=str,
                        default="./Results/graph_index_tree_100WUni.pkl")
    parser.add_argument("-E", "--embedding", type=str,
                        default="./Results/hgnn_keyword_embeddings_100WUni.pt")
    parser.add_argument("-f", "--function", type=str,
                        default="MAX")
    args = parser.parse_args()
    info = Info(
        input=args.input,
        output=args.output,
        query_graph_size=args.querySize,
        keyword_domain=args.keywordDomain
    )
    
    info.start_time = time.time()

    logging.info("This is an info message")
    logging.info(args)

    Info.input = args.input
    iter = 50
    G = ig.Graph.Read_GML(info.input)
    print("Data Graph: {}".format(ig.summary(G)))
    E = torch.load(args.embedding, weights_only=False)
    
    with open(args.index, 'rb') as f:
        index =  pickle.load(f)
    all_pp = 0
    all_fp = 0
    all_fn = 0
    avg_time = 0
    pp = 0
    max_ans = 0
    info_s = None
    f = args.function
    sigma = 1
    for i in range(iter):
        
        node_keywords = {
            0: ["Neural_Networks", "SVM"],
            1: ["Neural_Networks"],
            2: ["SVM", "Decision_Trees"]
        }
        edge_weights = {
            (0, 1): 1.5,
            (1, 2): 2.0,
            (0, 2): 3.0
        }
        num_nodes = len(node_keywords)
        all_edges = list(itertools.combinations(range(num_nodes), 2))
        edges = list(edge_weights.keys())
        q = ig.Graph(n=num_nodes, edges=edges, directed=False)
        for node_id, keywords in node_keywords.items():
            q.vs[node_id]["keywords"] = keywords
        for i, edge in enumerate(q.es):
            u, v = edge.tuple
            # igraph 的 tuple 总是小的编号在前，但为了稳妥起见
            u_idx, v_idx = min(u, v), max(u, v)                
            q.es[i]["weight"] = edge_weights[(u_idx, v_idx)]
        
        
        q = generate_query(G=G, n=info.query_graph_size, p=0.0, keyword_domain=info.keywordDomain)
        # print("Query Graph: {}".format(ig.summary(q)))
        # for node in q.vs():
        #     print("Query Node: {}, Keywords: {}".format(int(node["id"]), node['keywords']))
        
        weights = q.es.attributes()
        # if 'weight' in weights:
        #     none_count = q.es["weight"].count(None)
        #     if none_count > 0:
        #         print(f"警告：原图中共有 {none_count} 条边缺少权重！")
        #     else:
        #         print("所有边均有权重。")
        
        online_start = time.time()
        # for q in G_q.vs():
        #     print(q["id"], q["keywords"], q.index)
        # pp, fp, fn = pruning_power(G_q, G_forq, E)
        # all_pp += pp
        # all_fn += fn
        # all_fp += fp
        # expected_pruned, _, _ = calculate_expected_pruned_count(G_q, G_forq)
        
        S, p_ = S3GND(G=G, q=q, root=index, f=f, theta=sigma, E=E)
        online_time = time.time() - online_start
        avg_time += online_time
        pp += p_
        print(f"\n{'='*60}")
        print(f"Iteration {i+1} 完成:")
        print(f"  Online 总时间: {online_time:.4f} 秒")
        print(f"  找到 {len(S)} 个结果子图")
        print(f"{'='*60}\n")

        if max_ans < len(S):
            max_ans = len(S)
            info_s = S
    avg_time /= iter
    avg_pp = pp / iter
    print(f"\n平均 Online 时间: {avg_time:.4f} 秒")
    info.finish_time = time.time()
    info.iterations = iter
    info.avg_time = avg_time
    info.S = max_ans
    info.ANS = info_s
    info.pruning_power = avg_pp
    info.sigma = sigma
    info.F = f
    info_file_save(info, args.dataset)
    print("Done")
    
    
    # print(all_pp/iter, all_fp/iter, all_fn/iter)
        # for q in G_q.vs():
        #     print(q["id"], q["keywords"], q.index)
        
            # 原始图对应顶点查看
            # oq = G_forq.vs.select(id=q["id"])
            # print(oq["id"], oq["keywords"])
        # print("Query Graph: {}".format(G_q))
        
        