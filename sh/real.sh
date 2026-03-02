conda activate WQ
cd ../

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/real/cora.gml --dataset cora --n cora
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_cora.pt --gml ../Datasets/precompute/real/cora.gml --dataset cora
# cd ../
# python main.py -i ./Datasets/precompute/real/cora.gml -d cora -r ./Results/graph_index_tree_cora.pkl -E ./Results/hgnn_keyword_embeddings_cora.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/real/wiki.gml --dataset wiki --n wiki
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_wiki.pt --gml ../Datasets/precompute/real/wiki.gml --dataset wiki
# cd ../
# python main.py -i ./Datasets/precompute/real/wiki.gml -d wiki -r ./Results/graph_index_tree_wiki.pkl -E ./Results/hgnn_keyword_embeddings_wiki.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/real/pubmed.gml --dataset pubmed --n pubmed
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_pubmed.pt --gml ../Datasets/precompute/real/pubmed.gml --dataset pubmed
# cd ../
# python main.py -i ./Datasets/precompute/real/pubmed.gml -d pubmed -r ./Results/graph_index_tree_pubmed.pkl -E ./Results/hgnn_keyword_embeddings_pubmed.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/real/ppi.gml --dataset ppi --n ppi
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_ppi.pt --gml ../Datasets/precompute/real/ppi.gml --dataset ppi
# cd ../
# python main.py -i ./Datasets/precompute/real/ppi.gml -d ppi -r ./Results/graph_index_tree_ppi.pkl -E ./Results/hgnn_keyword_embeddings_ppi.pt

# cd ./KeywordEmbedding
# python train.py --gml ../Datasets/precompute/real/shanghai.gml --dataset shanghai --n shanghai
# cd ../Index
# python index.py --pt ../Results/hgnn_keyword_embeddings_shanghai.pt --gml ../Datasets/precompute/real/shanghai.gml --dataset shanghai
# cd ../
# python main.py -i ./Datasets/precompute/real/shanghai.gml -d shanghai -r ./Results/graph_index_tree_shanghai.pkl -E ./Results/hgnn_keyword_embeddings_shanghai.pt

python BF.py -i ./Datasets/precompute/real/cora.gml -d cora_bf 
python BF.py -i ./Datasets/precompute/real/wiki.gml -d wiki_bf
python BF.py -i ./Datasets/precompute/real/pubmed.gml -d pubmed_bf
python BF.py -i ./Datasets/precompute/real/shanghai.gml -d shanghai_bf
# python BF.py -i ./Datasets/precompute/real/tweibo.gml -d tweibo_bf 
