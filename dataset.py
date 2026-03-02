import time
from torch_geometric.datasets import AttributedGraphDataset, EllipticBitcoinDataset, CityNetwork
import torch
import igraph as ig

def check_feature_density(data):
    density = torch.count_nonzero(data.x) / data.x.numel()
    has_negative = torch.any(data.x < 0)
    
    print(f"特征密度 (Density): {density:.4f}")
    print(f"含有负数: {has_negative}")
    
    if density < 0.1 and not has_negative:
        print("结论：该数据集非常适合使用 (x > 0) 处理并计算 Jaccard。")
    elif has_negative:
        print("结论：该数据集含有负数，可能是嵌入向量，建议改用余弦相似度。")
    else:
        print("结论：特征较稠密，请根据业务逻辑决定是否二值化。")

def add_jaccard_weights_to_pyg(type, dataset_name):
    if type == 1:
        dataset = AttributedGraphDataset(root='./Datasets/real', name=dataset_name)
    else:
        dataset = CityNetwork(root='./Datasets/real', name=dataset_name)
    data = dataset[0]
    is_binary = torch.all((data.x == 0) | (data.x == 1))
    print(f"该数据集是严格的 0/1 向量吗? {is_binary}")
    
    
    check_feature_density(data)
    
    x = data.x.bool() 
    # x = data.x = (data.x > 0).float()
    edge_index = data.edge_index
    
    # 3. 提取每条边的起点 (u) 和终点 (v) 的特征矩阵
    # row, col 形状均为 [E], E 是边的数量
    row, col = edge_index
    
    # x[row] 会得到所有起点特征, x[col] 得到所有终点特征
    # 形状均为 [E, 特征维度]
    feat_u = x[row]
    feat_v = x[col]

    # 4. 向量化计算 Jaccard 相似度
    # 交集：两个向量对应位置都是 True 的数量
    intersection = torch.logical_and(feat_u, feat_v).sum(dim=1).float()
    
    # 并集：两个向量中至少有一个位置是 True 的数量
    union = torch.logical_or(feat_u, feat_v).sum(dim=1).float()

    # 计算权重，处理除以 0 的情况（如果并集为 0，则权重为 0）
    edge_weight = intersection / union
    edge_weight[torch.isnan(edge_weight)] = 0.0

    # 5. 将权重存回 data 对象
    data.edge_attr = edge_weight

    print(f"数据集: {dataset_name}")
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"前5条边的Jaccard权重: {data.edge_attr[:5]}")
    
    # 创建掩码：权重 > 0 的位置为 True
    mask = data.edge_attr > 0

    # 提取非 0 权重
    non_zero_weights = data.edge_attr[mask]

    print(f"非零权重的数量: {non_zero_weights.size(0)}")
    print(f"前 10 个非零权重: {non_zero_weights[:10]}")
    
    num_total = data.num_edges
    num_non_zero = torch.sum(mask).item()
    percentage = (num_non_zero / num_total) * 100

    print(f"总边数: {num_total}")
    print(f"非零权重边数: {num_non_zero}")
    print(f"非零占比: {percentage:.2f}%")

    # 查看权重的统计信息（均值、最大、最小）
    if num_non_zero > 0:
        print(f"非零权重的平均值: {non_zero_weights.mean():.4f}")
        print(f"非零权重的最大值: {non_zero_weights.max():.4f}")
    
    return data

def add_jaccard_weights_to_pygE():
    dataset = EllipticBitcoinDataset(root='./Datasets/real')
    data = dataset[0]
    is_binary = torch.all((data.x == 0) | (data.x == 1))
    print(f"该数据集是严格的 0/1 向量吗? {is_binary}")
    
    
    check_feature_density(data)
    
    # 2. 准备特征
    # 确保特征是布尔类型或 0/1 类型，方便计算交集和并集
    x = data.x.bool() 
    # x = data.x = (data.x > 0).float()
    edge_index = data.edge_index
    
    # 3. 提取每条边的起点 (u) 和终点 (v) 的特征矩阵
    # row, col 形状均为 [E], E 是边的数量
    row, col = edge_index
    
    # x[row] 会得到所有起点特征, x[col] 得到所有终点特征
    # 形状均为 [E, 特征维度]
    feat_u = x[row]
    feat_v = x[col]

    # 4. 向量化计算 Jaccard 相似度
    # 交集：两个向量对应位置都是 True 的数量
    intersection = torch.logical_and(feat_u, feat_v).sum(dim=1).float()
    
    # 并集：两个向量中至少有一个位置是 True 的数量
    union = torch.logical_or(feat_u, feat_v).sum(dim=1).float()

    # 计算权重，处理除以 0 的情况（如果并集为 0，则权重为 0）
    edge_weight = intersection / union
    edge_weight[torch.isnan(edge_weight)] = 0.0

    # 5. 将权重存回 data 对象
    data.edge_attr = edge_weight

    print(f"数据集: Elliptic")
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"前5条边的Jaccard权重: {data.edge_attr[:5]}")
    
    # 创建掩码：权重 > 0 的位置为 True
    mask = data.edge_attr > 0

    # 提取非 0 权重
    non_zero_weights = data.edge_attr[mask]

    print(f"非零权重的数量: {non_zero_weights.size(0)}")
    print(f"前 10 个非零权重: {non_zero_weights[:10]}")
    
    num_total = data.num_edges
    num_non_zero = torch.sum(mask).item()
    percentage = (num_non_zero / num_total) * 100

    print(f"总边数: {num_total}")
    print(f"非零权重边数: {num_non_zero}")
    print(f"非零占比: {percentage:.2f}%")

    # 查看权重的统计信息（均值、最大、最小）
    if num_non_zero > 0:
        print(f"非零权重的平均值: {non_zero_weights.mean():.4f}")
        print(f"非零权重的最大值: {non_zero_weights.max():.4f}")
    
    return data

def add_jaccard_weights_to_pyg2(dataset_name):
    dataset = AttributedGraphDataset(root='./Datasets/real', name=dataset_name)
    data = dataset[0]
    
    # 将特征转为布尔型以节省空间
    x = data.x.bool()
    edge_index = data.edge_index
    num_edges = edge_index.size(1)
    
    # 预分配一个全 0 的权重张量，放在 CPU 上以节省显存
    edge_weight = torch.zeros(num_edges, dtype=torch.float32)

    # 设定每一批处理的边数（根据你的内存调整，10万到50万比较稳妥）
    chunk_size = 50000
    
    print(f"开始分块计算 {dataset_name} 的 Jaccard 权重...")

    for i in range(0, num_edges, chunk_size):
        end = min(i + chunk_size, num_edges)
        
        # 仅提取当前批次的边索引
        row_chunk = edge_index[0, i:end]
        col_chunk = edge_index[1, i:end]
        
        # 提取对应特征
        feat_u = x[row_chunk]
        feat_v = x[col_chunk]
        
        # 计算交集和并集
        intersection = torch.logical_and(feat_u, feat_v).sum(dim=1).float()
        union = torch.logical_or(feat_u, feat_v).sum(dim=1).float()
        
        # 计算权重并填入预分配的张量中
        w = intersection / union
        w[torch.isnan(w)] = 0.0
        edge_weight[i:end] = w
        
        if i % (chunk_size * 2) == 0:
            print(f"进度: {end}/{num_edges} 边已处理...")

    data.edge_attr = edge_weight
    
    # --- 后续的统计代码保持不变 ---
    print(f"计算完成！")
    print(f"数据集: {dataset_name}")
    print(f"节点数: {data.num_nodes}")
    print(f"边数: {data.num_edges}")
    print(f"前5条边的Jaccard权重: {data.edge_attr[:5]}")
    
    # 创建掩码：权重 > 0 的位置为 True
    mask = data.edge_attr > 0

    # 提取非 0 权重
    non_zero_weights = data.edge_attr[mask]

    print(f"非零权重的数量: {non_zero_weights.size(0)}")
    print(f"前 10 个非零权重: {non_zero_weights[:10]}")
    
    num_total = data.num_edges
    num_non_zero = torch.sum(mask).item()
    percentage = (num_non_zero / num_total) * 100

    print(f"总边数: {num_total}")
    print(f"非零权重边数: {num_non_zero}")
    print(f"非零占比: {percentage:.2f}%")

    # 查看权重的统计信息（均值、最大、最小）
    if num_non_zero > 0:
        print(f"非零权重的平均值: {non_zero_weights.mean():.4f}")
        print(f"非零权重的最大值: {non_zero_weights.max():.4f}")
        
        
    return data

def generate_igraph(data, save_name):
    g = ig.Graph(n=data.num_nodes, directed=False)
    
    # 1. 找到现有的最大关键字索引
    row_indices, col_indices = data.x.nonzero(as_tuple=True)
    # 如果 data.x 为空，设默认最大值为 -1
    max_kw_idx = col_indices.max().item() if col_indices.numel() > 0 else -1
    special_idx = max_kw_idx + 1  # 定义特殊关键字
    
    # 2. 将关键词按节点分组
    node_kws = [[] for _ in range(data.num_nodes)]
    for r, c in zip(row_indices.tolist(), col_indices.tolist()):
        node_kws[r].append(str(c))
        
    # 修改点：如果节点没有关键字，添加特殊索引 special_idx
    formatted_kws = []
    for kw in node_kws:
        if not kw:
            formatted_kws.append(str(special_idx)) # 存入特殊索引
        else:
            formatted_kws.append(",".join(kw))
            
    g.vs["keywords"] = formatted_kws
    g.vs["igraphnxname"] = list(range(data.num_nodes))

    print("正在添加边和权重...")
    edges = data.edge_index.t().tolist()
    weights = data.edge_attr.tolist()
    
    # 不能过滤掉权重为 0 的边（提高 GML 读取效率）
    valid_edges = []
    valid_weights = []
    for i in range(len(edges)):
        # if weights[i] > 0:
        valid_edges.append(edges[i])
        valid_weights.append(weights[i])
    
    # 批量添加边
    g.add_edges(valid_edges)
    # 给边赋予权重属性
    g.es["weight"] = valid_weights

    # 3. 处理位掩码 EK 和 邻居数据 NW
    for node_idx in range(g.vcount()):
        # --- NW 处理逻辑 (保持你之前的代码) ---
        neighbor_data = []
        for neighbor_idx in g.neighbors(node_idx):
            edge_id = g.get_eid(node_idx, neighbor_idx)
            weight = g.es[edge_id]['weight']
            neighbor_data.append((neighbor_idx, weight))
        
        neighbor_data.sort(key=lambda x: x[1], reverse=True)
        nw_string = ",".join([f"{idx}:{round(w, 4)}" for idx, w in neighbor_data])
        g.vs[node_idx]['NW'] = nw_string

        # --- EK 位运算处理 ---
        kw_str = g.vs[node_idx]['keywords']
        bv = 0
        if kw_str:
            keywords = kw_str.split(',')
            for k in keywords:
                if k.strip():
                    # 此时这里的 k 可能是正常的索引，也可能是 special_idx
                    bv |= (1 << int(k))
        
        g.vs[node_idx]['EK'] = str(bv)

    print(f"构建完成。特殊关键字索引为: {special_idx}")
    print(f"节点数: {g.vcount()}, 边数: {g.ecount()}")
    
    print(f"正在导出到 {save_name}...")
    # igraph 默认导出的 GML 格式非常标准
    g.write_gml(save_name)
    print("导出成功！")

# def generate_igraph(data, save_name):
    # g = ig.Graph(n=data.num_nodes, directed=False)
    
    # # 找到所有非零特征的坐标 (row: 节点索引, col: 关键词索引)
    # row_indices, col_indices = data.x.nonzero(as_tuple=True)
    

    # # 将关键词按节点分组
    # node_kws = [[] for _ in range(data.num_nodes)]
    # # 使用 list 存储，稍后合并为字符串
    # for r, c in zip(row_indices.tolist(), col_indices.tolist()):
    #     node_kws[r].append(str(c))
        
    # g.vs["keywords"] = [",".join(kw) for kw in node_kws]
    # g.vs["igraphnxname"] = list(range(data.num_nodes))
    
    # print("正在添加边和权重...")
    # edges = data.edge_index.t().tolist()
    # weights = data.edge_attr.tolist()
    
    # # 不能过滤掉权重为 0 的边（提高 GML 读取效率）
    # valid_edges = []
    # valid_weights = []
    # for i in range(len(edges)):
    #     # if weights[i] > 0:
    #     valid_edges.append(edges[i])
    #     valid_weights.append(weights[i])
    
    # # 批量添加边
    # g.add_edges(valid_edges)
    # # 给边赋予权重属性
    # g.es["weight"] = valid_weights
    
    # for node_idx in range(g.vcount()):
    #     neighbor_data = []
        
    #     # igraph 获取邻居的方式
    #     for neighbor_idx in g.neighbors(node_idx):
    #         # 获取连接这两个节点的边 ID
    #         edge_id = g.get_eid(node_idx, neighbor_idx)
    #         # 从边属性中获取权重 (假设你的边权重属性名叫 'weight')
    #         weight = g.es[edge_id]['weight']
    #         neighbor_data.append((neighbor_idx, weight))
        
    #     # 按权重降序排序
    #     neighbor_data.sort(key=lambda x: x[1], reverse=True)
        
    #     # 序列化为 "id:weight,id:weight" 格式
    #     nw_string = ",".join([f"{idx}:{round(w, 4)}" for idx, w in neighbor_data])
    #     g.vs[node_idx]['NW'] = nw_string

    #     # 处理 keywords 属性
    #     # 注意：igraph 中 node 直接迭代得到的是 Vertex 对象，访问属性用字典方式
    #     node_obj = g.vs[node_idx]
    #     if node_obj['keywords']:
    #         keywords = str(node_obj['keywords']).split(',')
    #         bv = 0
    #         for k in keywords:
    #             if k.strip(): # 避免空字符串报错
    #                 bv |= (1 << int(k))
    #         g.vs[node_idx]['EK'] = str(bv)
    # print(f"构建完成。节点数: {g.vcount()}, 边数: {g.ecount()}")

    # # 4. 导出为 GML
    # print(f"正在导出到 {save_name}...")
    # # igraph 默认导出的 GML 格式非常标准
    # g.write_gml(save_name)
    # print("导出成功！")
    
if __name__ == "__main__":
    
    t1 = time.time()
    data_with_weights = add_jaccard_weights_to_pyg(1, 'Cora')
    generate_igraph(data_with_weights, "./Datasets/precompute/real/cora.gml")
    print(f"Cora 处理总时间: {time.time() - t1:.2f} 秒")
    
    t2 = time.time()
    data_with_weights = add_jaccard_weights_to_pyg(1, 'Wiki')
    generate_igraph(data_with_weights, "./Datasets/precompute/real/wiki.gml")
    print(f"Wiki 处理总时间: {time.time() - t2:.2f} 秒")
    
    # data_with_weights = add_jaccard_weights_to_pyg(1, 'PPI')
    # generate_igraph(data_with_weights, "./Datasets/precompute/real/ppi.gml")
    
    t3 = time.time()
    data_with_weights = add_jaccard_weights_to_pyg(1, 'PubMed')
    generate_igraph(data_with_weights, "./Datasets/precompute/real/pubmed.gml")
    print(f"PubMed 处理总时间: {time.time() - t3:.2f} 秒")
    
    t4 = time.time()
    data_with_weights = add_jaccard_weights_to_pyg(2, 'shanghai')
    generate_igraph(data_with_weights, "./Datasets/precompute/real/shanghai.gml")
    print(f"shanghai 处理总时间: {time.time() - t4:.2f} 秒")
    
    # t5 = time.time()
    # data_with_weights = add_jaccard_weights_to_pyg2('TWeibo')
    # generate_igraph(data_with_weights, "./Datasets/precompute/real/tweibo.gml")
    # print(f"TWeibo 处理总时间: {time.time() - t5:.2f} 秒")
    # data_with_weights = add_jaccard_weights_to_pygE()
    # Cora
    # 该数据集是严格的 0/1 向量吗? True
    # 特征密度 (Density): 0.0127
    # 含有负数: False
    # 结论：该数据集非常适合使用 (x > 0) 处理并计算 Jaccard。
    # 数据集: Cora
    # 节点数: 2708
    # 边数: 5429
    # 前5条边的Jaccard权重: tensor([0.0476, 0.0256, 0.0513, 0.2381, 0.0000])
    # 非零权重的数量: 4851
    # 前 10 个非零权重: tensor([0.0476, 0.0256, 0.0513, 0.2381, 0.0513, 0.0513, 0.0278, 0.0833, 0.0513,
    #         0.0750])
    # 总边数: 5429
    # 非零权重边数: 4851
    # 非零占比: 89.35%
    # 非零权重的平均值: 0.1074
    # 非零权重的最大值: 1.0000

    # PubMed
    # 该数据集是严格的 0/1 向量吗? False
    # 特征密度 (Density): 0.1002
    # 含有负数: False
    # 结论：特征较稠密，请根据业务逻辑决定是否二值化。
    # 数据集: PubMed
    # 节点数: 19717
    # 边数: 44338
    # 前5条边的Jaccard权重: tensor([0.1635, 0.1481, 0.1406, 0.1750, 0.2426])
    # 非零权重的数量: 44299
    # 前 10 个非零权重: tensor([0.1635, 0.1481, 0.1406, 0.1750, 0.2426, 0.0800, 0.3043, 0.2264, 0.2062,
    #         0.0968])
    # 总边数: 44338
    # 非零权重边数: 44299
    # 非零占比: 99.91%
    # 非零权重的平均值: 0.1704
    # 非零权重的最大值: 1.0000

    # PPI
    # 该数据集是严格的 0/1 向量吗? True
    # 特征密度 (Density): 0.0185
    # 含有负数: False
    # 结论：该数据集非常适合使用 (x > 0) 处理并计算 Jaccard。
    # 数据集: PPI
    # 节点数: 56944
    # 边数: 1612348
    # 前5条边的Jaccard权重: tensor([0., 0., 0., 0., 0.])
    # 非零权重的数量: 100754
    # 前 10 个非零权重: tensor([1.0000, 0.1667, 0.1667, 0.5000, 0.5000, 0.5000, 0.5000, 0.3333, 0.3333,
    #         0.3333])
    # 总边数: 1612348
    # 非零权重边数: 100754
    # 非零占比: 6.25%
    # 非零权重的平均值: 0.5196
    # 非零权重的最大值: 1.0000

    # elliptic
    # 该数据集是严格的 0/1 向量吗? False
    # 特征密度 (Density): 1.0000
    # 含有负数: True
    # 结论：该数据集含有负数，可能是嵌入向量，建议改用余弦相似度。
    # 数据集: Elliptic
    # 节点数: 203769
    # 边数: 234355
    # 前5条边的Jaccard权重: tensor([1., 1., 1., 1., 1.])
    # 非零权重的数量: 234355
    # 前 10 个非零权重: tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
    # 总边数: 234355
    # 非零权重边数: 234355
    # 非零占比: 100.00%
    # 非零权重的平均值: 1.0000
    # 非零权重的最大值: 1.0000

    # shanghai
    # 该数据集是严格的 0/1 向量吗? False
    # 特征密度 (Density): 0.3147
    # 含有负数: False
    # 结论：特征较稠密，请根据业务逻辑决定是否二值化。
    # 数据集: shanghai
    # 节点数: 183917
    # 边数: 524184
    # 前5条边的Jaccard权重: tensor([0.7692, 0.7500, 0.6000, 0.8182, 0.7692])
    # 非零权重的数量: 524184
    # 前 10 个非零权重: tensor([0.7692, 0.7500, 0.6000, 0.8182, 0.7692, 0.7143, 0.6667, 0.7500, 1.0000,
    #         0.8333])
    # 总边数: 524184
    # 非零权重边数: 524184
    # 非零占比: 100.00%
    # 非零权重的平均值: 0.9051
    # 非零权重的最大值: 1.0000

    # wiki
    # 该数据集是严格的 0/1 向量吗? False
    # 特征密度 (Density): 0.1301
    # 含有负数: False
    # 结论：特征较稠密，请根据业务逻辑决定是否二值化。
    # 数据集: Wiki
    # 节点数: 2405
    # 边数: 17981
    # 前5条边的Jaccard权重: tensor([0., 0., 0., 0., 0.])
    # 非零权重的数量: 15466
    # 前 10 个非零权重: tensor([0.1747, 0.1792, 1.0000, 1.0000, 0.2331, 0.2869, 0.3304, 0.2617, 0.3210,
    #         0.3144])
    # 总边数: 17981
    # 非零权重边数: 15466
    # 非零占比: 86.01%
    # 非零权重的平均值: 0.3121
    # 非零权重的最大值: 1.0000