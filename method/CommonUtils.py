import pandas as pd
import numpy as np
from typing import List
from method.detector.BleuDbscan import dbscan
from method.detector.CosKmeans import kmeans_cos
from method.detector.EditSpectral import spectral_clustering
from method.detector.RougeLof import LOF


def f1Score(label, pred, freq, freqFlag=False):
    TP = 0
    FN = 0
    FP = 0
    for i in range(len(label)):
        val = 1 if freqFlag else freq[i]
        if label[i] == 1 and pred[i] == 1:
            TP += val
        elif label[i] == 1 and pred[i] == 0:
            FN += val
        elif label[i] == 0 and pred[i] == 1:
            FP += val
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    return round((2 * TP) / (2 * TP + FP + FN), 4), round(p, 4), round(r, 4)


# 把聚类结果处理到dataframe中，返回值是一个与排序后的traces数组下标一一对应的label一维数组
def resultProcessor(data, clusterResult):
    label = np.zeros(len(data))
    for cluster in clusterResult:
        for t in cluster:
            label[data.loc[data['trace'] == t].index] = 1
    return label


# 数据集读取器
def realLogLoader(inputFile, k):
    # 如果你需要自己处理数据集，需要把csv格式文件表示案例的列名改成'case'，表示事件的列名改成'event'
    case_id = "case"
    activity = "event"
    event_log = pd.read_csv(inputFile).loc[:, [case_id, activity]]
    # 创建数据变量，统计数据集各项参数
    # 去重轨迹表
    all_traces: List[str] = []
    # 轨迹发生频率字典
    traces_dict: dict = {}
    # 活动数字典
    activity_dict: dict = {}
    traces_num = 0
    case_dict: dict = {}
    for i in range(len(event_log)):
        # 统计活动数
        if activity_dict.get(event_log[activity][i]) is None:
            activity_dict[event_log[activity][i]] = 0
        else:
            activity_dict[event_log[activity][i]] += 1
        # 将活动串联成轨迹
        if case_dict.get(event_log[case_id][i]) is None:
            case_dict[event_log[case_id][i]] = []
        case_dict[event_log[case_id][i]].append(event_log[activity][i])
    for val in case_dict.values():
        if len(val) >= 3:
            # 原始日志的活动名可能包含空格，而轨迹的各个活动之间也是由空格分隔，因此要把活动名中的空格替换成下划线
            val = [s.replace(" ", "_") for s in val]
            activities = ' '.join(val)
            if traces_dict.get(activities) is None:
                all_traces.append(activities)
                traces_dict[activities] = 1
            else:
                traces_dict[activities] += 1
            traces_num += 1
    # 这里是按照轨迹的发生频率对轨迹列表降序排序
    all_traces.sort(key=lambda trace: traces_dict[trace], reverse=True)
    data = [[trace, traces_dict[trace]] for trace in all_traces]
    df = pd.DataFrame(data, columns=['trace', 'freq'])
    # 进行kmeans聚类，作为评估F1值的标准，之前用的dbscan作为标准，现在把dbscan作为对比方法了
    kmeans_result, k_time = kmeans_cos(df, 0.6, k)
    # label标签为1表示正常轨迹，为0表示离群点
    df['label'] = resultProcessor(df, kmeans_result)
    print('共有轨迹', sum(traces_dict.values()), '条，去重轨迹', len(all_traces), '条，活动', len(activity_dict.keys()),
          '个，事件',
          sum(activity_dict.values()),
          '次')
    return df


# kn值就是k近邻，threshold值是lof值高于阈值时被视作离群点
# 而k值是聚类的簇的数量，别搞混
def stater(filename, kn=5, threshold=2, k=5, bleu_eps=0.6, edit_eps=0.6):
    print(f"开始处理数据集:{filename}")
    # 读取CSV文件
    data = realLogLoader('data/realLog/' + filename, k)

    # 计算LOF
    lof_scores, lof_time = LOF(data, kn)
    # 将结果添加到原始数据中
    data['lof_scores'] = lof_scores
    data['lof_pred'] = data['lof_scores'].apply(lambda x: 0 if x > threshold else 1)
    # 分别计算F1分数
    f1, p, r = f1Score(data['label'], data['lof_pred'], data['freq'])
    print(f"Lof       F1 Score: {f1} precision: {p} recall: {r} cost time: {lof_time}")

    # dbscan同理
    dbscan_result, dbscan_time = dbscan(data, bleu_eps)
    data['dbscan_pred'] = resultProcessor(data, dbscan_result)
    f1, p, r = f1Score(data['label'], data['dbscan_pred'], data['freq'])
    print(f"DBSCAN    F1 Score: {f1} precision: {p} recall: {r} cost time: {dbscan_time}")
    # 谱聚类同理
    spectral_result, spec_time = spectral_clustering(data, edit_eps, k)
    data['spectral_pred'] = resultProcessor(data, spectral_result)
    f1, p, r = f1Score(data['label'], data['spectral_pred'], data['freq'])
    print(f"Spectral  F1 Score: {f1} precision: {p} recall: {r} cost time: {spec_time}")

    # 结果导出到result/下
    pd.DataFrame(data).to_csv('result/' + filename, index=None)
