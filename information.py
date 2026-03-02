import os
import time



class Info:
    def __init__(
        self,
        input,
        output,
        query_graph_size,
        keyword_domain
    ) -> None:
        self.input = os.path.join(input)
        self.output = os.path.join(output)

        self.query_graph_size = query_graph_size
        self.keywordDomain = keyword_domain

        self.start_time = 0
        self.finish_time = 0
        self.iterations = 0
        self.avg_time = 0
        self.S = 0
        self.ANS = None
        self.pruning_power = 0
        self.F = "MAX"
        self.sigma = 1
        
    def get_S3GND_answer(self) -> str:
        ans = ""
        ans += "INFORMATION RESULTS\n"
        ans += "------------------ FILE INFO ------------------\n"
        ans += "Input File Path: {}\n".format(self.input)
        ans += "Output File Path: {}\n".format(self.output)
        ans += "Keyword Domain Size: {}\n".format(self.keywordDomain)
        ans += "------------------ ANS INFO ------------------\n"
        ans += "Query Graph Size: {}\n".format(self.query_graph_size)
        ans += "Function F: {}\n".format(self.F)
        ans += "Sigma: {}\n".format(self.sigma)
        ans += "Start Time: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)))
        ans += "Finish Time: {}\n".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.finish_time)))
        ans += "Total Iterations: {}\n".format(self.iterations)
        ans += "Average Time per Iteration: {:.4f} seconds\n".format(self.avg_time)
        ans += "The Last Answers Found: {}\n".format(self.S)
        ans += "The Last Answer Subgraphs: {}\n".format(self.ANS)
        ans += "\n"
        ans += "------------------ PRUNING INFO ------------------\n"
        ans += "Pruning Power: {:.4f}%\n".format(self.pruning_power * 100)
        ans += "\n"
        return ans
