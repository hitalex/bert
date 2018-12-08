#coding=utf8

import pickle

from graph_tool.all import *
import pandas as pd
import numpy as np

# 默认设置最大的节点数
MAX_NUM_TID = 200000

def build_graph(train_path, graph_path, edge_property_map_path):
    '''
    Input:
        train_path: 读入的Dataframe
    Output:
        graph_path: 输出的graph的路径
        edge_property_map_path: 输出的edge property map路径
    '''
    print('Loading train dataset...')
    train_df = pd.read_csv(train_path)

    print('Building graph...')
    g = Graph()
    g.add_vertex(MAX_NUM_TID)
    edge_property_map = dict()
    edge_list = []
    for i, row in train_df.iterrows():
        tid1 = str(row['tid1']).strip()
        tid2 = str(row['tid2']).strip()
        v1 = g.vertex(int(tid1))
        v2 = g.vertex(int(tid2))
        label = row['label']
        if label == 'unrelated' or label == 'agreed':
            edge_list.append((v1, v2))
            edge_list.append((v2, v1))
            edge_property_map[tid1 + '#' + tid2] = label
            edge_property_map[tid2 + '#' + tid1] = label
        else:
            edge_list.append((v2, v1))
            edge_property_map[tid2 + '#' + tid1] = label

    g.add_edge_list(edge_list)
    print('Adding %d edges' % len(edge_list))

    print('Pickling the graph and edge property...')
    g.save(graph_path)
    pickle.dump(edge_property_map, open(edge_property_map_path, 'wb'))
    
    print('Done')

def graph_reason(graph_path, edge_property_map_path, test_path, result_path):
    '''
    Input:
        graph_path: 导入的graph路径
        edge_property_map_path: edge property map路径
        test_path: 测试集路径
    Output:
        test_path: 推理结果的路径
    '''
    print('Loading graph and edge property map...')
    g = Graph()
    g.load(graph_path)

    # key: target#source, value: label
    edge_property_map = pickle.load(open(edge_property_map_path, 'rb'))

    print('Reading test set...')
    test = pd.read_csv(test_path)
    print('Num. of test samples: %d' % len(test))

    # key: test sample的id，value为各类的概率
    #key: id, value: [prob. of unrelated, prob. of agreed, prob. of disagreed]
    label_index_map = {'unrelated': 0, 'agreed': 1, 'disagreed':2}
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
                result[sample_id] = np.array([0] * len(label_index_map), float) # 存储每个类的个数

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
                    #result[sample_id].append('unrelated')
                    result[sample_id][label_index_map['unrelated']] += 1
                elif label_list.count('disagreed') == 0: # 全部都是agreed，则一定agreed，表示都是谣言
                    #result[sample_id].append('agreed')
                    result[sample_id][label_index_map['agreed']] += 1
                elif label_list.count('disagreed') == 1: # 只有一个disagreed，则一定位于链条的第一个，表示是辟谣信息，剩下都为谣言
                    if label_list[0] != 'disagreed':
                        print('Error: disagreed should be the first one')
                    else:
                        #result[sample_id].append('disagreed')
                        result[sample_id][label_index_map['disagreed']] += 1
                else:
                    print('Error: there are never more than one disagreed labels in one path')

        # 后处理
        if not sample_id in result:
            pass
        elif sum(result[sample_id]) == 0: # 虽然有路径，但是任何的最短路径都无法确认
            result.pop(sample_id)
        else:
            result[sample_id] = 1.0 * result[sample_id] / sum(result[sample_id])

        if i % 1000 == 0:
            print('Index count: %d, num of reasoned result: %d' % (i, len(result)))

        #if i > 1000:
        #    break

    print('Find %d paths' % len(result))
    pickle.dump(result, open(result_path, 'wb'))
    #import ipdb; ipdb.set_trace()

def eval_graph_reason_result(result_path, test_path):
    ''' 对基于graph reason的结果进行评估
    Input:
        result_path: graph reason的结果
        test_path: test集合的结果
    '''
    result = pickle.load(open(result_path, 'rb'))
    id_list = []
    for key in result:
        id_list.append(int(key))

    test = pd.read_csv(test_path)
    y_pred = []
    y_true = []
    for key in id_list:
        tmp = test[test['id'] == key]
        if len(tmp) == 1:
            y_pred.append(result[str(key)])
            y_true.append(tmp.iloc[0]['label'])

    from sklearn.metrics import classification_report
    print(classification_report(y_true, y_pred))
 
if __name__ == '__main__':

     pass
