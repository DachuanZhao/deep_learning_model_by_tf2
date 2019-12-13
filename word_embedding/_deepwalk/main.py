import logging
import random
import datetime
import csv
import time
import collections
import numpy as np
logging.basicConfig(level=logging.INFO)

class Graph(collections.defaultdict):
    """
    比nx的graph性能更好，无环图;
    这里最好不要用邻接矩阵存，而应该用dict，因为前者太稀疏太占内存
    """
    def __init__(self):
        super(Graph, self).__init__(dict)
        self.node_dict = {}
        self.prob_dict = {}

    def nodes(self):
        """返回所有点"""
        return self.keys()

    def make_consistent(self):
        """
        保持一致性，剔除重复的点和剔除环
        """
        #t0 = time.time()
        #for k in self:
        #    self[k] = list(sorted(set(self[k])))
        #t1 = time.time()
        #logging.info('make_consistent: made consistent in {}s'.format(t1-t0))

        self.remove_self_loops()

        return self

    def remove_self_loops(self):
        """
        剔除环
        """
        removed = 0
        t0 = time.time()
        for x in self:
            if x in self[x]:
                logging.info("remove_loop_node: {}".format(x))
                del self[x][x]
                removed += 1
        t1 = time.time()
        logging.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
        return self

    def prepare_for_random_walk(self):
        self.node_dict = {}
        self.prob_dict = {}
        for x in self:
            self.node_dict[x] = []
            self.prob_dict[x] = []
            for y,prob in self[x].items():
                self.node_dict[x].append(y)
                self.prob_dict[x].append(prob)
            self.node_dict[x] = np.array(self.node_dict[x],dtype=np.int32)
            self.prob_dict[x] = np.array(self.prob_dict[x],dtype=np.float32)
            self.prob_dict[x] = self.prob_dict[x] / np.sum(self.prob_dict[x])

    def random_walk(self, path_length,start_node):
        """
        返回随机路径
        """
        path = [start_node]

        while len(path) < path_length:
            current_node = path[-1]
            #不在孤立的点生成路径
            if len(self[current_node]) > 0:
                path.append(np.random.choice(
                        self.node_dict[current_node],
                        size=1,
                        replace=True,
                        p=self.prob_dict[current_node],
                    )[0]
                )
            else:
                break
        return path

delimiter = " "
quotechar = '"'

def load_edgelist(_file, undirected=True):
    G = Graph()
    with open(_file) as f:
        for i,l in enumerate(f):
            if i % 100000 == 0:
                logging.info("正在读取第{}行".format(i))
            x, y ,confidence = l.strip().split(" ")
            confidence = float(confidence)
            x = int(x)
            y = int(y)
            G[x][y] = confidence
            if undirected:
                G[y][x] = confidence
    G.make_consistent()
    return G



def main(edge_list_file_path,
        output_file_path,
        is_undirected,
        walk_length,
        epochs):

    with open(output_file_path,'w',newline='') as write_file:
        csv_writer = csv.writer(write_file,delimiter=delimiter,quotechar=quotechar)

        G = load_edgelist(edge_list_file_path, undirected=is_undirected)
        logging.info("Number of nodes: {}".format(len(G.nodes())))
        G.prepare_for_random_walk()

        nodes = list(G.nodes())
        for cnt in range(epochs):
            logging.info("每轮的长度为{},目前正在生成第{}轮".format(len(G.nodes()),cnt))
            for node in nodes:
                csv_writer.writerow(G.random_walk(walk_length,start_node=node))

if __name__ ==  "__main__":
    #输入文件
    #edge_list_file_path = "./data/output_file_id_test.csv"
    time_now = datetime.datetime.now().strftime("_%Y%m%d_%H%M%S")
    edge_list_file_path = "./data/all_relationship.txt"
    output_file_path = "./data/output_file_random_walk_with_confidence{}.txt".format(time_now)
    is_undirected = True
    walk_length = 20
    epochs = 100
    main(edge_list_file_path=edge_list_file_path,
            output_file_path=output_file_path,
            is_undirected=is_undirected,
            walk_length=walk_length,
            epochs=epochs,
            )

