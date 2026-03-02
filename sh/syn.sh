conda activate WQ
cd ../

# uni
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/10000-24979-50-3/G-uni.gml --dataset 1WUni --n 1WUni
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_1WUni.pt --gml ../Datasets/precompute/synthetic/10000-24979-50-3/G-uni.gml --dataset 1WUni
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/10000-24979-50-3/G-uni.gml -d 1WUni -r ./Results/graph_index_tree_1WUni.pkl -E ./Results/hgnn_keyword_embeddings_1WUni.pt
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-uni.gml --dataset 5WUni --n 5WUni
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_5WUni.pt --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-uni.gml --dataset 5WUni
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/50000-124812-50-3/G-uni.gml -d 5WUni -r ./Results/graph_index_tree_5WUni.pkl -E ./Results/hgnn_keyword_embeddings_5WUni.pt
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/250000-624992-50-3/G-uni.gml --dataset 25WUni --n 25WUni
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_25WUni.pt --gml ../Datasets/precompute/synthetic/250000-624992-50-3/G-uni.gml --dataset 25WUni
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/250000-624992-50-3/G-uni.gml -d 25WUni -r ./Results/graph_index_tree_25WUni.pkl -E ./Results/hgnn_keyword_embeddings_25WUni.pt
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/1000000-2500431-50-3/G-uni.gml --dataset 100WUni --n 100WUni
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_100WUni.pt --gml ../Datasets/precompute/synthetic/1000000-2500431-50-3/G-uni.gml --dataset 100WUni
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/1000000-2500431-50-3/G-uni.gml -d 100WUni -r ./Results/graph_index_tree_100WUni.pkl -E ./Results/hgnn_keyword_embeddings_100WUni.pt

# gau
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/10000-24979-50-3/G-gau.gml --dataset 1WGau --n 1WGau
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_1WGau.pt --gml ../Datasets/precompute/synthetic/10000-24979-50-3/G-gau.gml --dataset 1WGau
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/10000-24979-50-3/G-gau.gml -d 1WGau -r ./Results/graph_index_tree_1WGau.pkl -E ./Results/hgnn_keyword_embeddings_1WGau.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml --dataset 5WGau --n 5WGau
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_5WGau.pt --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml --dataset 5WGau
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/50000-124812-50-3/G-gau.gml -d 5WGau -r ./Results/graph_index_tree_5WGau.pkl -E ./Results/hgnn_keyword_embeddings_5WGau.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/250000-624992-50-3/G-gau.gml --dataset 25WGau --n 25WGau
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_25WGau.pt --gml ../Datasets/precompute/synthetic/250000-624992-50-3/G-gau.gml --dataset 25WGau
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/250000-624992-50-3/G-gau.gml -d 25WGau -r ./Results/graph_index_tree_25WGau.pkl -E ./Results/hgnn_keyword_embeddings_25WGau.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/1000000-2500431-50-3/G-gau.gml --dataset 100WGau --n 100WGau
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_100WGau.pt --gml ../Datasets/precompute/synthetic/1000000-2500431-50-3/G-gau.gml --dataset 100WGau
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/1000000-2500431-50-3/G-gau.gml -d 100WGau -r ./Results/graph_index_tree_100WGau.pkl -E ./Results/hgnn_keyword_embeddings_100WGau.pt

# zipf
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/10000-24979-50-3/G-zipf.gml --dataset 1WZipf --n 1WZipf
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_1WZipf.pt --gml ../Datasets/precompute/synthetic/10000-24979-50-3/G-zipf.gml --dataset 1WZipf
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/10000-24979-50-3/G-zipf.gml -d 1WZipf -r ./Results/graph_index_tree_1WZipf.pkl -E ./Results/hgnn_keyword_embeddings_1WZipf.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-zipf.gml --dataset 5WZipf --n 5WZipf
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_5WZipf.pt --gml ../Datasets/precompute/synthetic/50000-124812-50-3/G-zipf.gml --dataset 5WZipf
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/50000-124812-50-3/G-zipf.gml -d 5WZipf -r ./Results/graph_index_tree_5WZipf.pkl -E ./Results/hgnn_keyword_embeddings_5WZipf.pt
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/250000-624992-50-3/G-zipf.gml --dataset 25WZipf --n 25WZipf
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_25WZipf.pt --gml ../Datasets/precompute/synthetic/250000-624992-50-3/G-zipf.gml --dataset 25WZipf
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/250000-624992-50-3/G-zipf.gml -d 25WZipf -r ./Results/graph_index_tree_25WZipf.pkl -E ./Results/hgnn_keyword_embeddings_25WZipf.pt
# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/synthetic/1000000-2500431-50-3/G-zipf.gml --dataset 100WZipf --n 100WZipf
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_100WZipf.pt --gml ../Datasets/precompute/synthetic/1000000-2500431-50-3/G-zipf.gml --dataset 100WZipf
# cd ../
# python main.py -i ./Datasets/precompute/synthetic/1000000-2500431-50-3/G-zipf.gml -d 100WZipf -r ./Results/graph_index_tree_100WZipf.pkl -E ./Results/hgnn_keyword_embeddings_100WZipf.pt