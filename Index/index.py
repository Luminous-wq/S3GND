import argparse
import json
import pickle
import os

import random
import time
import numpy as np
import torch
import igraph as ig
from collections import Counter

# 假设的辅助工具类，实际使用时请确保这些逻辑存在
class IndexUtils:
    @staticmethod
    def merge_nw(existing_nw, new_nw, top_k=20):
        """
        合并两个有序的权重列表并保持 Top-K。
        NW 格式假设为: [(neighbor_id, weight), ...]
        """
        # 如果是简单的权重列表聚合，可以使用归并排序逻辑
        combined = existing_nw + new_nw
        # 按权重降序排列
        combined.sort(key=lambda x: x[1], reverse=True)
        # 去重并保留 Top-K (针对同一个邻居可能由多个子节点导出的情况)
        seen = {}
        result = []
        for n_id, w in combined:
            if n_id not in seen:
                seen[n_id] = w
                result.append((n_id, w))
            if len(result) >= top_k:
                break
        return result

    @staticmethod
    def get_bit_vector(ek_list):
        """将关键词 ID 列表转换为位掩码 (Integer representation)"""
        bv = 0
        for k in ek_list:
            bv |= (1 << int(k))
        return bv

def root_tree(nodes_list, num_partition, level, Graph, keyword_embedding):
    # --- 递归基：叶子节点层 ---
    if len(nodes_list) <= num_partition:
        leaf_entries = []
        for node in nodes_list:
            # 解析原始属性
            # NW 存储格式为 "id1:w1,id2:w2"
            nw_raw = [pair.split(':') for pair in Graph.vs[node]['NW'].split(",") if pair]
            nw_list = [(int(p[0]), float(p[1])) for p in nw_raw]
            ek_list = [int(k) for k in str(Graph.vs[node]['keywords']).split(",")]
            
            leaf_entries.append({
                "P": node,
                "Aux": {
                    "NW": nw_list,
                    "EK": IndexUtils.get_bit_vector(ek_list) # 存储为 bit vector
                },
                "T": True,
                "L": level
            })
        return leaf_entries

    # --- 分区逻辑 ---
    # 调用你定义的初始化和代价模型函数
    partition_ans, centers_bv, Graph1 = initialize_partition(
        Graph=Graph, nodes_list=nodes_list, 
        num_partition=num_partition, keyword_embedding=keyword_embedding
    )
    
    final_partition, _, Graph = cost_model(
        Graph=Graph1, partition=partition_ans, t=20,
        centers_bv=centers_bv, nodes_list=nodes_list, 
        num_partition=num_partition
    )

    # --- 递归构建与属性聚合 ---
    current_level_entries = []

    # 遍历每个分区构建子树
    for i in range(1, num_partition + 1):
        partition_nodes = final_partition.get(i, [])
        if not partition_nodes:
            continue
            
        # 递归调用
        child_entries = root_tree(partition_nodes, num_partition, level + 1, Graph, keyword_embedding)

        # 聚合该分区（即当前父节点）的属性
        agg_nw = []
        agg_ek = 0
        
        for child in child_entries:
            # 权重聚合：合并排序
            agg_nw = IndexUtils.merge_nw(agg_nw, child['Aux']['NW'])
            # 关键词聚合：按位或
            agg_ek |= child['Aux']['EK']

        # 创建父节点条目
        current_level_entries.append({
            "P": child_entries, # 指向子节点列表
            "Aux": {
                "NW": agg_nw,
                "EK": agg_ek
            },
            "T": False,
            "L": level
        })

    return current_level_entries

class IndexPersistence:
    @staticmethod
    def save_index_json(tree_data, filename):
        """
        保存为 JSON 格式。
        注意：Python 的大整数位向量在 JSON 中会自动转为数字字符串或数字。
        """
        def convert_to_serializable(obj):
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            return obj

        serializable_data = convert_to_serializable(tree_data)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        print(f"索引已保存至 JSON: {filename}")

    @staticmethod
    def save_index_pickle(tree_data, filename):
        """
        保存为 Pickle 格式。支持所有 Python 原生对象，加载速度最快。
        """
        with open(filename, 'wb') as f:
            pickle.dump(tree_data, f)
        print(f"索引已保存至 Pickle: {filename}")

    @staticmethod
    def load_index(filename):
        """根据后缀名自动加载索引"""
        ext = os.path.splitext(filename)[1]
        if ext == '.json':
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.pkl' or ext == '.pickle':
            with open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("不支持的文件格式")

    @staticmethod
    def print_tree_structure(nodes, indent=0):
        """
        递归打印树结构预览
        """
        for entry in nodes:
            prefix = "  " * indent
            node_type = "Leaf" if entry["T"] else "Internal"
            ek_count = bin(entry["Aux"]["EK"]).count('1')
            nw_count = len(entry["Aux"]["NW"])
            
            print(f"{prefix}[Level {entry['L']} {node_type}] EK_Bits: {ek_count}, NW_Size: {nw_count}")
            
            if not entry["T"]: # 如果不是叶子节点，递归打印子节点
                IndexPersistence.print_tree_structure(entry["P"], indent + 1)

def print_tree_summary(tree_data):
    """
    层序遍历统计：只打印每一层有多少个 block (分区)
    """
    print("\n" + "="*40)
    print(f"{'Level':<10} | {'Total Blocks':<15} | {'Type Distribution'}")
    print("-" * 40)
    
    current_layer = tree_data # root_tree 返回的是一个 list
    level = 0
    
    while current_layer:
        total_count = len(current_layer)
        leaf_count = sum(1 for node in current_layer if node.get('T') == True)
        internal_count = total_count - leaf_count
        
        print(f"Level {level:<5} | {total_count:<15} | {internal_count} Internal, {leaf_count} Leaf")
        
        # 收集下一层的所有子节点
        next_layer = []
        for node in current_layer:
            # 只有内部节点的 P 才是列表，叶子节点的 P 是整数 ID
            if not node.get('T') and isinstance(node.get('P'), list):
                next_layer.extend(node['P'])
        
        current_layer = next_layer
        level += 1
    print("="*40)
    
def cost_model(Graph, partition, t, centers_bv, nodes_list, num_partition):
    # 1. 预计算：节点到分区的映射
    node_to_part = {}
    for p_idx, p_nodes in partition.items():
        for node in p_nodes:
            node_to_part[node] = p_idx

    # 2. 预计算：节点位向量 (使用局部变量加速访问)
    node_bvs = {}
    for node in nodes_list:
        ek_list = [int(k) for k in str(Graph.vs[node]['keywords']).split(",") if k]
        bv = 0
        for k in ek_list:
            bv |= (1 << k)
        node_bvs[node] = bv

    # 超参数：alpha(结构权重), beta(平衡权重)
    alpha = 0.1
    beta = 0.5  
    # 理想的分区大小，用于平衡约束
    ideal_size = len(nodes_list) / num_partition

    for iter_idx in range(t):
        moved = 0
        random.shuffle(nodes_list)
        
        for node in nodes_list:
            current_p = node_to_part[node]
            node_bv = node_bvs[node]
            node_k_count = node_bv.bit_count() # Python 3.10+ 高效位统计

            # --- 优化：预计算该节点对每个分区的结构贡献 ---
            # 不在 P 循环内查询邻居，减少 16 倍的图查询压力
            p_struct_contrib = {p: 0.0 for p in range(1, num_partition + 1)}
            neighbors = Graph.neighbors(node)
            for nb in neighbors:
                if nb in node_to_part:
                    nb_part = node_to_part[nb]
                    # 获取边权
                    eid = Graph.get_eid(node, nb)
                    p_struct_contrib[nb_part] += Graph.es[eid]['weight']

            best_score = -float('inf')
            target_p = current_p

            # 评估移动
            for p_idx in range(1, num_partition + 1):
                # 1. 结构得分
                struct_score = p_struct_contrib[p_idx]

                # 2. 语义得分
                center_bv = centers_bv[p_idx-1]
                if node_k_count == 0:
                    semantic_score = 1.0
                else:
                    # 使用 bit_count 替代 bin().count('1')
                    intersection = (node_bv & center_bv).bit_count()
                    semantic_score = intersection / node_k_count

                # 3. 平衡得分 (防止 Level 29 的关键)
                # 如果分区节点数超过理想大小，分数会降低
                size_penalty = beta * (len(partition[p_idx]) / ideal_size)

                total_score = (alpha * struct_score) + ((1 - alpha) * semantic_score) - size_penalty

                if total_score > best_score:
                    best_score = total_score
                    target_p = p_idx

            # 执行移动
            if target_p != current_p:
                partition[current_p].remove(node)
                partition[target_p].append(node)
                node_to_part[node] = target_p
                # 更新位向量中心
                centers_bv[target_p-1] |= node_bv
                moved += 1
        
        # 每一轮迭代后，重新精准计算所有分区的 centers_bv，防止位向量过度膨胀
        for p_idx in range(1, num_partition + 1):
            new_bv = 0
            for n_idx in partition[p_idx]:
                new_bv |= node_bvs[n_idx]
            centers_bv[p_idx-1] = new_bv

        if moved == 0:
            break
            
    return partition, centers_bv, Graph


def initialize_partition(Graph, nodes_list, num_partition, keyword_embedding):
    """
    基于关键词位向量(Bit Vector)相似度初始化分区
    目标：让关键词集具有包含关系或高度重合的节点进入同一个分区
    """
    # 1. 提取所有节点的位向量
    # 假设 Graph.vs[node]['keywords'] 存储的是 "id1,id2" 字符串
    node_bvs = {}
    for node in nodes_list:
        ek_list = [int(k) for k in str(Graph.vs[node]['keywords']).split(",") if k]
        bv = 0
        for k in ek_list:
            bv |= (1 << k)
        node_bvs[node] = bv

    # 2. 选取初始中心点 (使用 K-Means++ 思想或随机选取)
    # 这里随机选取 num_partition 个节点作为种子
    seeds = random.sample(nodes_list, min(num_partition, len(nodes_list)))
    centers_bv = [node_bvs[seed] for seed in seeds]

    # 3. 分配节点到最近的中心 (基于 Jaccard 相似度)
    partition_ans = {i: [] for i in range(1, num_partition + 1)}
    
    def calculate_jaccard_sim(bv1, bv2):
        intersection = bin(bv1 & bv2).count('1')
        union = bin(bv1 | bv2).count('1')
        return intersection / union if union > 0 else 0

    for node in nodes_list:
        node_bv = node_bvs[node]
        # 寻找相似度最高的中心
        best_sim = -1
        best_idx = 1
        for idx, c_bv in enumerate(centers_bv):
            sim = calculate_jaccard_sim(node_bv, c_bv)
            if sim > best_sim:
                best_sim = sim
                best_idx = idx + 1
        
        partition_ans[best_idx].append(node)

    # 4. 这里的 Graph1 可以是原图的子图诱导，或者保持原样返回用于后续 cost_model
    # 视你的 cost_model 是否需要重新计算子图边权而定
    return partition_ans, centers_bv, Graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", type=str, 
                        default="../Results/hgnn_keyword_embeddings_100WUni.pt")
    parser.add_argument("--gml", type=str, default="../Datasets/precompute/synthetic/1000000-2500431-50-3/G-uni.gml")
    parser.add_argument("--dataset", type=str, default="1WGau")
    args = parser.parse_args()
    # 1. 环境准备与数据加载
    # pt_file = "../Results/hgnn_keyword_embeddings_100WUni.pt"
    # gmlfile = "../Datasets/precompute/synthetic/1000000-2500431-50-3/G-uni.gml"
    Graph = ig.Graph.Read_GML(args.gml)
    # E = np.load(file)
    keyword_embedding = torch.load(args.pt, weights_only=False)
    # print(keyword_embedding)
    
    # 假设 Graph, keyword_embedding 已经定义好
    nodes_list = list(range(Graph.vcount())) # 构建所有顶点的索引
    
    print("开始构建多层索引树...")
    # 2. 调用递归构建函数 (之前补全的 root_tree)
    # 假设根节点的 level 从 0 开始
    t1 = time.time()
    index_tree = root_tree(
        nodes_list=nodes_list, 
        num_partition=16, 
        level=0, 
        Graph=Graph, 
        keyword_embedding=keyword_embedding
    )
    print(f"索引树构建完成，耗时 {time.time() - t1:.2f} 秒。")
    # 3. 持久化
    persistence = IndexPersistence()
    persistence.save_index_pickle(index_tree, "../Results/graph_index_tree_" + args.dataset + ".pkl")
    # persistence.save_index_json(index_tree, "../Results/graph_index_tree_" + args.dataset + ".json")
    
    # 4. 预览结构
    print("\n索引树结构预览:")
    # persistence.print_tree_structure(index_tree)
    print_tree_summary(index_tree)

if __name__ == "__main__":
    main()