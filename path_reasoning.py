#coding=utf8

import pickle

from graph_tool.all import *
import pandas as pd

if __name__ == '__main__':
    
    print('Loading graph and edge property map...')
    g = Graph()
    g.load('directed-graph.gt')

    # key: target#source, value: label
    edge_property_map = pickle.load(open('edge-property-map.pickle', 'rb'))

    print('Reading test set...')
    test = pd.read_csv('datasets/fakenews/test.csv')

    result = dict()
    for i, row in test.iterrows():
        sample_id = str(row['id'])
        tid1 = str(row['tid1'])
        tid2 = str(row['tid2'])
        v1 = g.vertex(int(tid1))
        v2 = g.vertex(int(tid2))
        #vlist, elist = graph_tool.topology.shortest_path(g, v2, v1)
        path_list = graph_tool.topology.all_shortest_paths(g, v2, v1)
        # 找到所有的最短路径，因为可能一条路径无法确定，而其他路径却可以
        # 无法遍历所有路径，因为所需要的时间太长
        for path in path_list:
            if not sample_id in result:
                result[sample_id] = [] # 用于存储可能的多个结果

            label_list = []
            for j in range(len(path) - 1):
                source = path[j]
                target = path[j+1]
                key = str(source) + '#' + str(target)
                if key in edge_property_map:
                    label = edge_property_map[key]
                    label_list.append(label)
                else:
                    print('Error: tid1: %d, tid2: %d not found in edge list' % (target, source))
                    label_list = []
                    break

            if len(label_list) > 0:
                if label_list.count('unrelated') >= 2: # 一个链条有两个或两个以上unrelated，表示无法确定
                    pass
                elif label_list.count('unrelated') == 1: # 只有一个unrelated，则一定unrelated
                    result[sample_id].append('unrelated')
                elif label_list.count('disagreed') == 0: # 全部都是agreed，则一定agreed，表示都是谣言
                    result[sample_id].append('agreed')
                elif label_list.count('disagreed') == 1: # 只有一个disagreed，则一定位于链条的第一个，表示是辟谣信息，剩下都为谣言
                    if label_list[0] != 'disagreed':
                        print('Error: disagreed should be the first one')
                    else:
                        result[sample_id].append('disagreed')
                else:
                    print('Error: there are never more than one disagreed labels in one path')

        # 后处理
        if not sample_id in result:
            pass
        elif len(result[sample_id]) == 0: # 虽然有路径，但是任何的最短路径都无法确认
            result.pop(sample_id)
        elif result[sample_id].count(result[sample_id][0]) != len(result[sample_id]):
            print('Error: multiple shortest paths, but with different results: %s --> %s: %r' % (tid2, tid1, result[sample_id]))
            result.pop(sample_id)
        else:
            result[sample_id] = result[sample_id][0]

        if i % 1000 == 0:
            print('Index count: %d, num of reasoned result: %d' % (i, len(result)))

        #if i > 1000:
        #    break

    print('Find %d paths' % len(result))
    pickle.dump(result, open('graph-reason-result.pickle', 'wb'))
    #import ipdb; ipdb.set_trace()
    pass
