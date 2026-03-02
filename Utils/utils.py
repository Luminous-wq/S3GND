import math
import os
import time
import numpy as np
from scipy import sparse
import torch
import random
import igraph as ig

import random
from collections import defaultdict
from itertools import combinations

from torch.utils.data import Dataset

def create_folder(folder_name: str) -> bool:
    # base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    # folder_path = os.path.join(base_path, folder_name)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_name)
    # return True
    folder_path = os.path.abspath(folder_name)
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            print(f"folder create sucess: {folder_path}")
            return True
        except Exception as e:
            print(f"folder create fail: {e}")
            return False
    else:
        print(f"folder alrealy exists: {folder_path}")
        return True

def build_hyperedge_training_pairs_fast(
    hyperedges_weight,
    keyword_to_idx,
    total_pairs=100000,
    pos_ratio=0.5,
    seed=42
):
    """
    使用倒排索引极速构建训练 pair 数据集
    """
    random.seed(seed)

    # ======================
    # Step 1: 超边列表和 idx-set
    # ======================
    hyperedges = list(hyperedges_weight.keys())
    n = len(hyperedges)

    edge_sets = [
        set(keyword_to_idx[k] for k in edge)
        for edge in hyperedges
    ]

    # ======================
    # Step 2: 构建倒排索引 keyword -> hyperedges
    # ======================
    inverted = defaultdict(list)

    for idx, edge in enumerate(edge_sets):
        for kw in edge:
            inverted[kw].append(idx)

    # ======================
    # Step 3: 利用倒排索引构建正样本 (不重复)
    # ======================
    positive_pairs_set = set()

    for kw, edges in inverted.items():
        if len(edges) > 1:
            # 所有包含 kw 的超边组合都相交
            for i, j in combinations(edges, 2):
                if i < j:
                    positive_pairs_set.add((i, j))

    positive_pairs = list(positive_pairs_set)
    random.shuffle(positive_pairs)

    # 目标正样本数
    num_pos_target = int(total_pairs * pos_ratio)

    if len(positive_pairs) > num_pos_target:
        positive_pairs = positive_pairs[:num_pos_target]
    else:
        num_pos_target = len(positive_pairs)   # 实际正样本数量

    # ======================
    # Step 4: 构建负样本（随机采样 + intersection=empty）
    # ======================
    negative_pairs = []
    num_neg_target = total_pairs - num_pos_target

    attempts = 0
    max_attempts = num_pos_target

    while len(negative_pairs) < num_neg_target and attempts < max_attempts:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue
        # 必须无交集
        if edge_sets[i].isdisjoint(edge_sets[j]):
            negative_pairs.append((i, j))
        attempts += 1

    # ======================
    # Step 5: 拼装最终数据 (hyperedge_tuples)
    # ======================
    dataset = []

    for i, j in positive_pairs:
        dataset.append((hyperedges[i], hyperedges[j], 1))

    for i, j in negative_pairs:
        dataset.append((hyperedges[i], hyperedges[j], 0))

    print(f"positive sample number: {len(positive_pairs)}, negative sample number: {len(negative_pairs)}")
    
    random.shuffle(dataset)
    return dataset


def generate_G_from_H(H, W, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, W, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, W, variable_weight))
        return G

# def _generate_G_from_H(H, W, variable_weight=False):
#     """
#     calculate G from hypgraph incidence matrix H
#     :param H: hypergraph incidence matrix H
#     :param variable_weight: whether the weight of hyperedge is variable
#     :return: G
#     """
#     H = np.array(H)
#     # n_edge = H.shape[1]
#     # the weight of the hyperedge
#     # W = np.ones(n_edge)
#     # print(W.shape)
#     # the degree of the node
#     DV = np.sum(H * W, axis=1)
#     # the degree of the hyperedge
#     DE = np.sum(H, axis=0)

#     # invDE_vector = np.power(DE, -1)
#     # invDE = sparse.diags(invDE_vector)
#     invDE = np.asmatrix(np.diag(np.power(DE, -1)))
#     DV2 = np.asmatrix(np.diag(np.power(DV, -0.5)))
#     W = np.asmatrix(np.diag(W))
#     H = np.asmatrix(H)
#     HT = H.T

#     if variable_weight:
#         DV2_H = DV2 * H
#         invDE_HT_DV2 = invDE * HT * DV2
#         return DV2_H, W, invDE_HT_DV2
#     else:
#         G = DV2 * H * W * invDE * HT * DV2
#         return G

def _generate_G_from_H(H, W, variable_weight=False):
    """
    使用稀疏矩阵优化计算，防止 144GiB 内存溢出
    """
    # 确保 H 是稀疏矩阵格式 (csc 或 csr)
    if not sparse.issparse(H):
        H = sparse.csc_matrix(H)
    
    # 1. 计算度矩阵 (D_v 和 D_e)
    # H * W: H 是 (N, M), W 是 (M,)
    # 注意：稀疏矩阵乘法需要 W 为对角阵或使用 multiply
    DV = np.array(H.dot(W)).flatten()        # 节点度
    DE = np.array(H.sum(axis=0)).flatten()   # 超边度

    # 处理除以 0 的情况（防止 degree 为 0 导致溢出）
    invDE_vec = np.power(DE, -1.0, where=DE!=0)
    DV2_vec = np.power(DV, -0.5, where=DV!=0)

    # 2. 构建稀疏对角矩阵
    invDE = sparse.diags(invDE_vec)
    DV2 = sparse.diags(DV2_vec)
    W_sparse = sparse.diags(W)
    
    HT = H.getH() # 获取共轭转置，对于实数等同于 .T

    if variable_weight:
        # 稀疏矩阵相乘
        DV2_H = DV2 @ H
        invDE_HT_DV2 = invDE @ HT @ DV2
        return DV2_H, W_sparse, invDE_HT_DV2
    else:
        # 计算 G = D_v^-1/2 * H * W * D_e^-1 * H^T * D_v^-1/2
        # 使用 @ 符号进行稀疏矩阵乘法
        G = DV2 @ H @ W_sparse @ invDE @ HT @ DV2
        return G

def get_mbr(keys, E):
    # print(keys)
    emb = np.array([E[str(ind)] for ind in keys])
    emb_tensor = torch.tensor(emb)
    # print(keys, emb_tensor.shape)
    # E[list(keys)]   # shape = [K, D]
    return emb_tensor.min(0).values, emb_tensor.max(0).values  # (min[D], max[D])

def mbr_disjoint(minA, maxA, minB, maxB):
    """
    若所有维度都不重叠 → 返回 True
    任一维度相交 → 返回 False
    """
    # A 高于 B 或 B 高于 A
    cond1 = maxA < minB   # shape = [D]
    cond2 = maxB < minA   # shape = [D]

    disjoint_dims = cond1 | cond2  # 每维是否不重叠
    
    # print(cond1, cond2, disjoint_dims)
    return torch.any(disjoint_dims).item()  # 所有维都不重叠？
    # return torch.all(disjoint_dims).item()  # 所有维都不重叠？

def predict_no_intersection(Q1, Q2, E):
    minA, maxA = get_mbr(Q1, E)
    minB, maxB = get_mbr(Q2, E)
    return mbr_disjoint(minA, maxA, minB, maxB)

def true_no_intersection(Q1, Q2):
    # print(Q1, Q2, Q1 & Q2)
    return len(Q1 & Q2) == 0

def generate_random_node(min_k, max_k, num_keywords, keywords):
    k = random.randint(min_k, max_k)
    return set(random.sample(keywords, k))

def test_accuracy(num_tests, E):
    correct = 0
    for _ in range(num_tests):
        Q1 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E))
        Q2 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E))
        
        gt = true_no_intersection(Q1, Q2)
        pred = predict_no_intersection(Q1, Q2, E)

        if gt == pred:
            correct += 1
            print(gt)

    print(correct, num_tests)
    return correct / num_tests

def test_accuracy_with_fp(num_tests, E):
    correct = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(num_tests):
        Q1 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E))
        Q2 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E))
        
        gt = true_no_intersection(Q1, Q2)  # 真实情况
        pred = predict_no_intersection(Q1, Q2, E)  # 预测情况
        
        if gt == pred:
            correct += 1
            # print(f"正确案例 {i}:")
            # print(f"  Q1: {Q1}")
            # print(f"  Q2: {Q2}") 
            # print(f"  交集: {Q1 & Q2}")
        else:
            # 假阴性：预测不相交但实际有交集
            if pred == True and gt == False:
                false_negatives += 1
                # print(f"假阴性案例 {i}:")
                # print(f"  Q1: {Q1}")
                # print(f"  Q2: {Q2}") 
                # print(f"  交集: {Q1 & Q2}")
            # 假阳性：预测相交但实际不相交
            elif pred == False and gt == True:
                false_positives += 1
                # print(f"假阳性案例 {i}:")
                # print(f"  Q1: {Q1}")
                # print(f"  Q2: {Q2}") 
                # print(f"  交集: {Q1 & Q2}")
    
    total = num_tests
    print(f"准确率: {correct/total:.3f}")
    print(f"假阳性率: {false_positives/total:.3f} ({false_positives}/{total})")
    print(f"假阴性率: {false_negatives/total:.3f} ({false_negatives}/{total})")
    
    return correct / total, false_positives / total, false_negatives / total

# 1. MBR 包含性预测 (您应该已经有了这个函数，但我们在此定义以确保)：
def mbr_contains(minA, maxA, minB, maxB):
    """
    判断MBR A是否包含MBR B (A ⊇ B)
    """
    # A 必须包含 B 的所有维度
    contains_dims = (minA <= minB) & (maxA >= maxB)
    return torch.all(contains_dims).item() 

# 2. 真实值：集合包含性
def true_containment(Q1, Q2):
    """
    判断集合 Q1 是否包含 Q2 (Q1 ⊇ Q2)
    """
    # Q2 是否是 Q1 的子集
    return Q2.issubset(Q1) 

# 3. 预测：MBR 包含性
def predict_containment(Q1, Q2, E):
    minA, maxA = get_mbr(Q1, E)
    minB, maxB = get_mbr(Q2, E)
    # 预测 MBR(Q1) 包含 MBR(Q2)
    return mbr_contains(minA, maxA, minB, maxB)

def generate_containment_pair(num_keywords, keywords):
    # 强制 Q1 包含 Q2
    k2 = random.randint(1, 4)  # Q2 较小
    Q2 = set(random.sample(keywords, k2))
    
    k1 = random.randint(k2 + 1, 5) # Q1 较大
    
    # 确保 Q1 包含 Q2
    keywords_not_in_Q2 = [i for i in keywords if i not in Q2]
    # keywords_not_in_Q2 = random.sample(keywords, num_keywords)
    keywords_to_add = random.sample(keywords_not_in_Q2, k1 - k2)
    
    Q1 = Q2.union(set(keywords_to_add))
    return Q1, Q2

def test_containment_accuracy_fixed(num_tests, E):
    correct = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    
    # 用于计算正确分母
    actual_positives = 0  # 实际包含 (GT=True)
    actual_negatives = 0  # 实际不包含 (GT=False)
    
    for i in range(num_tests):
        
        # 样本生成逻辑 (保持不变)
        if i % 2 == 0: 
            Q1, Q2 = generate_containment_pair(len(E))
            gt = True
        else: 
            Q1 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E))
            Q2 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E))
            gt = true_containment(Q1, Q2)
        
        pred = predict_containment(Q1, Q2, E)
        
        # 统计实际标签
        if gt:
            actual_positives += 1
        else:
            actual_negatives += 1
        
        # 统计四种分类情况 (TP, TN, FP, FN)
        if gt == True and pred == True:
            true_positives += 1 # TP
        elif gt == False and pred == False:
            true_negatives += 1 # TN
        elif gt == False and pred == True:
            false_positives += 1 # FP (假阳性)
        elif gt == True and pred == False:
            false_negatives += 1 # FN (假阴性)
    
    # 汇总
    correct = true_positives + true_negatives
    total = num_tests
    acc = correct / total
    
    # 标准 FNR (分母: 实际正样本)
    fn_rate = false_negatives / actual_positives if actual_positives > 0 else 0
    # 标准 FPR (分母: 实际负样本)
    fp_rate = false_positives / actual_negatives if actual_negatives > 0 else 0
    
    print(f"准确率 (ACC): {acc:.3f} ({correct}/{total})")
    print(f"假阳性率 (FPR): {fp_rate:.3f} ({false_positives}/{actual_negatives})")
    print(f"假阴性率 (FNR): {fn_rate:.3f} ({false_negatives}/{actual_positives})")
    print(f"实际包含的测试样本总数 (TP+FN): {actual_positives}")
    print(f"TP: {true_positives}, TN: {true_negatives}, FP: {false_positives}, FN: {false_negatives}")
    
    return acc, fp_rate, fn_rate

def test_containment_accuracy(num_tests, E, keywords):
    correct = 0
    false_positives = 0
    false_negatives = 0
    total_true_containment = 0
    
    for i in range(num_tests):
        
        if i % 2 == 0: # 50% 强制包含
            Q1, Q2 = generate_containment_pair(len(E), keywords)
            gt = True
        else: # 50% 随机（大部分不包含）
            Q1 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E), keywords=keywords)
            Q2 = generate_random_node(min_k=1, max_k=5, num_keywords=len(E), keywords=keywords)
            gt = true_containment(Q1, Q2)
            
        # # 真实值：Q1 包含 Q2
        # gt = true_containment(Q1, Q2) 
        # # 预测值：MBR(Q1) 包含 MBR(Q2)
        pred = predict_containment(Q1, Q2, E)
        
        if gt == pred:
            correct += 1
            total_true_containment += (1 if pred else 0)
        else:
            # FPR (假阳性): 预测包含(True) 但实际不包含(False)
            if pred == True and gt == False:
                false_positives += 1
            # FNR (假阴性): 预测不包含(False) 但实际包含(True)
            elif pred == False and gt == True:
                false_negatives += 1
    
    total = num_tests
    acc = correct / total
    fp_rate = false_positives / total
    fn_rate = false_negatives / total
    
    print(f"准确率 (包含性): {acc:.3f}")
    print(f"假阳性率 (预测包含/实际不包含): {fp_rate:.3f} ({false_positives}/{total})")
    print(f"假阴性率 (预测不包含/实际包含): {fn_rate:.3f} ({false_negatives}/{total})")
    print(f"Total True Containment Samples in Test: {total_true_containment}")
    return acc, fp_rate, fn_rate

def evaluate(file, Embedding, epochs, keywords):
    # E = np.load(file)
    # E = torch.tensor(E, dtype=torch.float32)
    if file:
        E = torch.load(file, weights_only=False)
    else:
        E = Embedding
    print(len(E), E["0"])
    epochs = 10
    all_acc = 0
    for _ in range(epochs):
        acc, acc_fp, acc_fn = test_containment_accuracy(1000, E, keywords)
        all_acc += acc
        print(f"Accuracy = {acc * 100:.2f}%")
    print(f"Avg Accuracy = {all_acc/epochs * 100:.2f}%")
    return all_acc/epochs

def sample_real_node_keywords(keywords_list):
    """
    从原始超边/节点关键词列表中随机采样一个真实的集合。
    keywords_list: 训练时使用的 hyperedges (List[List[str]])
    """
    real_set = random.choice(keywords_list)
    return set(real_set)

def generate_containment_pair_from_real(keywords_list):
    """
    基于真实节点生成具有严格包含关系的对 (Q1, Q2)，确保 Q1 ⊇ Q2
    """
    # 1. 采样一个真实存在的关键词集合作为母集
    Q1 = sample_real_node_keywords(keywords_list)
    
    # 确保 Q1 不为空且足够大以进行拆分
    if len(Q1) <= 1:
        # 如果太小，递归重试或手动添加一个随机词
        return generate_containment_pair_from_real(keywords_list)

    # 2. 随机删除 1 到 len-1 个元素，生成子集 Q2
    drop_count = random.randint(1, len(Q1) - 1)
    Q2 = set(random.sample(list(Q1), len(Q1) - drop_count))
    
    return Q1, Q2

def test_containment_accuracy_real_based(num_tests, E, keywords_list):
    """
    使用真实节点分布的测试函数
    """
    correct = 0
    false_positives = 0
    false_negatives = 0
    
    for i in range(num_tests):
        if i % 2 == 0:
            # --- 50% 构造包含关系 (从真实节点演变) ---
            Q1, Q2 = generate_containment_pair_from_real(keywords_list)
            gt = True
        else:
            # --- 50% 随机采样两个真实节点 (大概率不包含) ---
            Q1 = sample_real_node_keywords(keywords_list)
            Q2 = sample_real_node_keywords(keywords_list)
            gt = Q2.issubset(Q1) # 真实判断它们是否具有包含关系
        
        # 预测逻辑
        pred = predict_containment(Q1, Q2, E)
        
        if gt == pred:
            correct += 1
        else:
            if pred == True and gt == False:
                false_positives += 1
            elif pred == False and gt == True:
                false_negatives += 1
                
    return correct / num_tests, false_positives / num_tests, false_negatives / num_tests

# 重新包装 evaluate 函数
def evaluate_real_distribution(file, Embedding, keywords_list):
    if file:
        E = torch.load(file, weights_only=False)
    else:
        E = Embedding
        
    num_tests = 1000
    acc, fp, fn = test_containment_accuracy_real_based(num_tests, E, keywords_list)
    
    print(f"--- 真实分布评估结果 ---")
    print(f"准确率: {acc:.4f}")
    print(f"假阳性率 (FPR): {fp:.4f}")
    print(f"假阴性率 (FNR): {fn:.4f}")
    return acc

def generate_query(G:ig.Graph, n:int, p:float, keyword_domain):
    # subgraph
    subgraph = get_random_subgraph(G, n)
    query_graph = remove_edges_optimized(subgraph=subgraph, p=p)
    # 0.95
    query_graph = sample_keywords_subset(query_graph, keywords_domain=keyword_domain, sample_rate=0.90)
    return query_graph

def get_random_subgraph(G: ig.Graph, n: int):
    while True:
        # 修改点 1：使用更严谨的有效性判断
        valid_nodes = [v.index for v in G.vs() if is_valid_keywords(v["keywords"])]
        
        if not valid_nodes:
            raise ValueError("No valid starting node found (all nodes have empty or '0' keywords).")
        
        start_node = random.choice(valid_nodes)
        selected_nodes = {start_node}

        while len(selected_nodes) < n:
            neighbors = set()
            for node in selected_nodes:
                # 修改点 2：在扩展邻居时也调用判断函数
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

# def get_random_subgraph(G: ig.Graph, n: int):
#     while True:
#         # 随机选择一个起始节点，且该节点的 keywords 不为 "0"
#         valid_nodes = [int(node["id"]) for node in G.vs() if node["keywords"] != "0"]
#         if not valid_nodes:
#             raise ValueError("No valid starting node found (all nodes have keywords='0').")
        
#         start_node = random.choice(valid_nodes)
#         selected_nodes = {start_node}

#         # print(f"selected {selected_nodes}")
#         while len(selected_nodes) < n:
#             neighbors = set()
#             for node in selected_nodes:
#                 # 获取邻居节点，并排除 keywords 为 "0" 的节点
#                 neighbors.update(neighbor for neighbor in G.neighbors(node) if G.vs[neighbor]["keywords"] != "0")

#             neighbors -= selected_nodes

#             if not neighbors:
#                 print("Cannot expand the subgraph while maintaining connectivity.")
#                 break
            
#             # print(neighbors)
#             new_node = random.choice(list(neighbors))
#             # print(new_node)
#             selected_nodes.add(new_node)

#         if len(selected_nodes) == n:
#             # print(f"selected {selected_nodes}")
#             return G.subgraph(selected_nodes)
#         else:
#             continue

def is_valid_keywords(keywords_str):
    """判断关键词字符串是否包含有效的数字"""
    if not keywords_str or str(keywords_str).strip() == "0":
        return False
    # 检查 split 后是否有任何非空元素
    parts = [k for k in str(keywords_str).split(",") if k.strip()]
    return len(parts) > 0

def remove_edges(subgraph:ig.Graph, p=0.3):
    """
    随机裁剪子图的边，以概率 p 对每条边进行裁剪，同时保证子图的连通性。
    """
    edges = subgraph.get_edgelist()
    for edge in edges:
        if random.random() < p:
            subgraph.delete_edges([edge])
            if not subgraph.is_connected():
                subgraph.add_edges([edge])
    return subgraph

    
def remove_edges_optimized(subgraph: ig.Graph, p=0.3):
    """
    高效裁剪边：利用生成树保证连通性，剩余边按概率删除。
    """
    if subgraph.ecount() == 0:
        return subgraph

    # 1. 找到一棵生成树的边索引 (这些边是保证连通的“骨架”)
    # 使用 spanning_tree 得到的是一个新图，我们需要的是它在原图中的边索引
    st = subgraph.spanning_tree(weights=None)
    
    # 技巧：通过给边打标签来识别哪些是生成树里的边
    # 也可以简单通过遍历边来判断，但最快的是利用 set 
    st_edges = set()
    for e in st.es:
        # 获取生成树边在原图中的对应点对，并查出原图索引
        st_edges.add(subgraph.get_eid(e.source, e.target))

    # 2. 找出所有不在生成树中的边
    all_edge_indices = set(range(subgraph.ecount()))
    removable_edges = list(all_edge_indices - st_edges)

    # 3. 从可删除的边中，根据概率 p 挑选要删除的边
    to_delete = [e_idx for e_idx in removable_edges if random.random() < p]

    # 4. 一次性批量删除（性能最高，且不会导致属性丢失的问题，因为没被删的边属性不动）
    subgraph.delete_edges(to_delete)
    
    return subgraph

# def sample_keywords_subset(subgraph:ig.Graph, keywords_domain):
#     """
#     对子图中每个节点的 keywords 属性进行子集采样。
#     对于每个节点，随机生成一个 m (1 <= m <= len(keywords)),
#     然后从 keywords 中随机选择 m 个作为子集。
#     """
#     # k_domain = keywords_domain
#     for node in subgraph.vs():
#         # if 'keywords' in node:
#         keywords = [int(k) for k in node["keywords"].split(",") if k.strip()]
#         # keywords = [int(k) for k in node["keywords"].split(",")]
#         if len(keywords) > 3:
#             sampled_keywords = random.sample(keywords, 3)
#             node['keywords'] = ",".join(map(str, sampled_keywords))

#     return subgraph
# def sample_keywords_subset(subgraph: ig.Graph, keywords_domain):
#     for node in subgraph.vs():
#         # 获取有效列表
#         raw_keywords = str(node["keywords"])
#         keywords = [int(k) for k in raw_keywords.split(",") if k.strip()]
        
#         # 只有在关键词足够多时才采样
#         if len(keywords) > 3:
#             sampled_keywords = random.sample(keywords, 3)
#             node['keywords'] = ",".join(map(str, sampled_keywords))
#         elif len(keywords) > 0:
#             # 如果关键词很少 (1-3个)，保留原样不采样，防止变为空集
#             node['keywords'] = ",".join(map(str, keywords))
#         else:
#             # 理论上经过 get_random_subgraph 过滤不应走到这里
#             # 但为了安全，可以赋予一个 domain 里的随机值或报错
#             print(f"Warning: Node {node.index} has no keywords during sampling.")
            
#     return subgraph

def sample_keywords_subset(subgraph, keywords_domain, sample_rate=0.8):
    """
    按比例采样关键词。
    :param subgraph: ig.Graph 对象
    :param keywords_domain: 关键词定义域
    :param sample_rate: 保留比例，如 0.8 表示保留 80%
    """
    for node in subgraph.vs():
        # 1. 获取原始关键词列表
        raw_keywords = str(node["keywords"])
        keywords = [int(k) for k in raw_keywords.split(",") if k.strip()]
        
        num_existing = len(keywords)
        
        if num_existing > 0:
            # 2. 计算采样数量：向上取整，确保至少保留 1 个（如果原先就有的话）
            # 你也可以用 int(num_existing * sample_rate) 然后取 max(1, ...)
            k_to_sample = math.ceil(num_existing * sample_rate)
            
            # 3. 执行采样
            sampled_keywords = random.sample(keywords, k_to_sample)
            
            # 4. 回填数据
            node['keywords'] = ",".join(map(str, sorted(sampled_keywords)))
        else:
            # 异常处理：原节点本身没有关键词
            print(f"Warning: Node {node.index} has no keywords during sampling.")
            
    return subgraph

def mbr_contains(minA, maxA, minB, maxB):
    """
    判断MBR A是否包含MBR B
    A包含B当且仅当: minA <= minB 且 maxA >= maxB (所有维度)
    """
    contains_dims = (minA <= minB) & (maxA >= maxB)
    return torch.all(contains_dims).item()

def pruning_power(G_q, G, embeddings):
    """
    计算查询图G_q在大图G上的剪枝能力
    
    Args:
        G_q: 查询图 (igraph对象)
        G: 目标大图 (igraph对象)  
        embeddings: 关键词embedding字典
    
    Returns:
        pp: 剪枝准确率
        fp: 假阳性率 (被错误剪枝的节点比例)
        fn: 假阴性率 (该被剪枝但未被剪枝的节点比例)
    """
    # 统计变量
    true_pruned = 0      # 正确剪枝的节点数
    false_positive = 0   # 假阳性：预测剪枝但不应剪枝
    false_negative = 0   # 假阴性：预测不剪枝但应剪枝
    total_pruned = 0     # 总剪枝节点数
    total_nodes = G.vcount()  # 总节点数
    
    # 为查询图G_q的每个节点计算MBR
    query_node_info = []
    for v_q in G_q.vs:
        str_Kvq = v_q['keywords']
        # keywords_q = set(str_Kvq.strip().split(","))
        keywords_q = {k.strip() for k in str_Kvq.split(",") if k.strip()}
        # keywords_q = set(v_q['keywords'])
        # print(v_q['keywords'], keywords_q)
        if keywords_q:  # 确保有关键词
            min_vec, max_vec = get_mbr(list(keywords_q), embeddings)
            query_node_info.append({
                'min_vec': min_vec,
                'max_vec': max_vec,
                'keywords': keywords_q,
                'original_node': v_q["id"]
            })
    
    # print(f"查询图有 {len(query_node_info)} 个包含关键词的节点")
    
    # 遍历大图G中的所有节点
    for i, v_g in enumerate(G.vs):
        str_Kvg = v_g['keywords']
        # keywords_g = set(str_Kvg.strip().split(","))
        keywords_g = [k.strip() for k in str(str_Kvg).split(",") if k.strip()]
        
        # 真实情况：如果G节点不包含任意一个查询节点的所有关键词，应该被剪枝
        should_be_pruned_true = True
        matched_query_nodes = []
        for q_info in query_node_info:
            if q_info['keywords'].issubset(keywords_g):
                should_be_pruned_true = False
                matched_query_nodes.append(q_info['original_node'])
                break  # 只要匹配一个查询节点就不剪枝
        
        # 基于MBR的预测：如果G节点的MBR不包含任何查询节点的MBR，预测剪枝
        predicted_pruned = True
        matched_by_mbr = []
        if keywords_g:  # G节点有关键词
            min_g, max_g = get_mbr(list(keywords_g), embeddings)
            
            for q_info in query_node_info:
                # 检查G节点的MBR是否包含查询节点的MBR
                if mbr_contains(min_g, max_g, q_info['min_vec'], q_info['max_vec']):
                    predicted_pruned = False
                    matched_by_mbr.append(q_info['original_node'])
                    break  # 只要有一个MBR包含关系就不剪枝
        
        # 统计结果
        if predicted_pruned:
            total_pruned += 1
            
            if should_be_pruned_true:
                true_pruned += 1  # 正确剪枝
            else:
                false_positive += 1  # 假阳性：错误剪枝
                # if false_positive <= 5:  # 只打印前5个假阳性案例
                #     print(f"假阳性案例: G节点{i} 关键词{keywords_g}")
                #     print(f"  实际匹配查询节点: {matched_query_nodes}")
        
        else:  # 预测不剪枝
            if should_be_pruned_true:
                false_negative += 1  # 假阴性：该剪枝但未剪枝
                # if false_negative <= 5:  # 只打印前5个假阴性案例
                #     print(f"假阴性案例: G节点{i} 关键词{keywords_g}")
                #     print(f"  MBR匹配查询节点: {matched_by_mbr}")
    
    # 计算指标
    if total_pruned > 0:
        pp = true_pruned / total_nodes  # 剪枝准确率
        fp = false_positive / total_nodes  # 假阳性率（在剪枝节点中）
    else:
        pp = 1.0
        fp = 0.0
    
    fn = false_negative / total_nodes  # 假阴性率（在总节点中）
    
    # 输出详细统计信息
    # print(f"\n=== 剪枝统计结果 ===")
    # print(f"总节点数: {total_nodes}")
    # print(f"剪枝节点数: {total_pruned} ({total_pruned/total_nodes*100:.2f}%)")
    # print(f"正确剪枝: {true_pruned}")
    # print(f"假阳性: {false_positive} (被错误剪枝但实际应保留的节点)")
    # print(f"假阴性: {false_negative} (该剪枝但未被剪枝的节点)")
    # print(f"剪枝准确率: {pp:.4f}")
    # print(f"假阳性率: {fp:.4f}")
    # print(f"假阴性率: {fn:.4f}")
    
    return pp, fp, fn

def calculate_expected_pruned_count(G_q, G):
    """
    计算理论上应该被剪枝的节点数量
    """
    expected_pruned_count = 0
    pruned_nodes_info = []
    
    # 收集所有查询节点的关键词集合
    query_keywords_sets = []
    for v_q in G_q.vs:
        str_Kvq = v_q['keywords']
        keywords_q = set(str_Kvq.strip().split(","))
        if keywords_q:  # 只考虑有关键词的查询节点
            query_keywords_sets.append(keywords_q)
    
    # 检查每个G节点
    for v_g in G.vs:
        str_Kvg = v_g['keywords']
        keywords_g = set(str_Kvg.strip().split(","))
        should_be_pruned = True
        
        # 检查是否包含任何查询节点的所有关键词
        for keywords_q in query_keywords_sets:
            if keywords_q.issubset(keywords_g):
                should_be_pruned = False
                break
        
        if should_be_pruned:
            expected_pruned_count += 1
            pruned_nodes_info.append({
                'node_id': v_g.index,
                'keywords': keywords_g,
                'reason': "不包含任何查询节点的所有关键词"
            })
    
    total_nodes = G.vcount()
    pruned_ratio = expected_pruned_count / total_nodes if total_nodes > 0 else 0
    
    # print(f"\n=== 理论剪枝统计 ===")
    # print(f"总节点数: {total_nodes}")
    # print(f"理应被剪枝的节点数: {expected_pruned_count}")
    # print(f"理论剪枝比例: {pruned_ratio:.4f} ({pruned_ratio*100:.2f}%)")
    
    # # 显示一些样例
    # if pruned_nodes_info:
    #     print(f"\n前5个理应被剪枝的节点样例:")
    #     for i, info in enumerate(pruned_nodes_info[:5]):
    #         print(f"  节点{info['node_id']}: 关键词{info['keywords']}")
    
    return expected_pruned_count, pruned_ratio, pruned_nodes_info

def info_file_save(info, dataset_name: str) -> bool:
    infor_path = "./Results/info"
    create_folder(folder_name=infor_path)
    base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    stat_file = 'information-' + dataset_name + "-" + time.strftime('%m%d-%H%M%S', time.localtime())+ '.txt'
    info.output_info_file_name = stat_file
    result_stat_file = open(os.path.join(base_path, infor_path, stat_file), 'w')
    result_stat_file.write(info.get_S3GND_answer())
    result_stat_file.close()
    print(info.output_info_file_name, "saved successfully!")
    return True

if __name__ == "__main__":
    evaluate(file="../KeywordEmbedding/hgnn_keyword_embeddings.pt", epochs=10)
    # evaluate(file="./hgnn_keyword_embeddings_matrix.npy", epochs=10)