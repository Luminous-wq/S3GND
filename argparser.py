import argparse

def args_parser():
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
    
    args = parser.parse_args()
    return args