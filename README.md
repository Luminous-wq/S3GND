## S3GND: An Effective Learning-Based Approach for Subgraph Similarity Search Under Generalized Neighbor Difference Semantics

This is the code-repo for **"S3GND: An Effective Learning-Based Approach for Subgraph Similarity Search Under Generalized Neighbor Difference Semantics"**.

## Check List

Code    &#x2705;

Dataset Source  &#x2705;

README  &#x2705;



## Required Environment

1. networkx 3.1 or above
2. igraph
3. torch_geometric (for loading datasets)
4. torch



## Data Sets

| Name     | \|V(G)\|  | \|E(G)\|   | \|∑\|     |
| -------- | --------- | ---------- | --------- |
| Cora | 2,708     | 5,429     | 1,433     |
| Wiki | 2,405     | 17,981     | 4,973     |
| PubMed   | 19,717    | 44,338     | 500       |
| shanghai | 183,917   | 524,184     | 37       |
| TWeibo   | 2,320,895 | 9,840,066  | 1,658     |



## Usage

```
usage: main_queue.py [-h] [-i INPUT] [-o OUTPUT] [-qs QUERYSIZE] [-s KEYWORDDOMAIN] [-d DATASET] [-r INDEX] [-E EMBEDDING] [-f FUNCTION]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        path of graph input file
  -o OUTPUT, --output OUTPUT
                        path of the output file
  -qs QUERYSIZE, --querySize QUERYSIZE
                        the query vertex set size
  -s KEYWORDDOMAIN, --keywordDomain KEYWORDDOMAIN
                        the keyword domain size
  -d DATASET, --dataset DATASET
  -r INDEX, --index INDEX
  -E EMBEDDING, --embedding EMBEDDING
  -f FUNCTION, --function FUNCTION
```



## Running Way

```
(A) For real data sets
    Step-1: (dataset.py) load initial files and obtain the initial graph G-xxxx.gml with keywords
    Step-2: (argparser.py) set the query in argparser.py or in the command line
    Step-3: (KeywordEmbedding/train.py) training hgnn model
    Step-4: (Index/index.py) generate index
    Step-5: (main_queue.py) python main_queue.py ................ (or not, if already set the query in argparser.py)
    
(B) For synthetic
    Step-1: (generate.py) generate the G-distribution.gml data graph with keywords
    Step-2: (argparser.py) set the query in argparser.py or in the command line
    Step-3: (KeywordEmbedding/train.py) training hgnn model
    Step-4: (Index/index.py) generate index
    Step-5: (main_queue.py) python main_queue.py ................ (or not, if already set the query in argparser.py)
```



## Conference

```
wait.
```