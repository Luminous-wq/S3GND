conda activate WQ
cd ../

#gau
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml --dataset 5WGau --n 5WGau
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_5WGau.pt --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml --dataset 5WGau
# cd ../
# python main_queue.py -i ./Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml -qs 3 -d 5WGau -r ./Results/graph_index_tree_5WGau.pkl -E ./Results/hgnn_keyword_embeddings_5WGau.pt
# python main_queue.py -i ./Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml -qs 8 -d 5WGau -r ./Results/graph_index_tree_5WGau.pkl -E ./Results/hgnn_keyword_embeddings_5WGau.pt
# python main_queue.py -i ./Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml -qs 10 -d 5WGau -r ./Results/graph_index_tree_5WGau.pkl -E ./Results/hgnn_keyword_embeddings_5WGau.pt

python main_queue.py -i ./Datasets/precompute/synthetic/250000-624992-50-3/G-gau.gml -d 25WGau -r ./Results/graph_index_tree_25WGau.pkl -E ./Results/hgnn_keyword_embeddings_25WGau.pt