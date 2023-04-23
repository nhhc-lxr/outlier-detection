from datetime import datetime
import numpy as np
from rouge import Rouge

rouge = Rouge()

# 定义指标的权重，这个你可以拿来写一写，调整一下比例，找一个比较好的比例讲一讲
weights = {"rouge-1": 0.4, "rouge-2": 0.4, "rouge-l": 0.2}


# 计算加权rouge12L
def calRougeScore(hyp, ref):
    scores = rouge.get_scores(hyp, ref)
    # 获取Rouge-1，Rouge-2和Rouge-L的得分
    rouge_1_score = scores[0]['rouge-1']['f']
    rouge_2_score = scores[0]['rouge-2']['f']
    rouge_l_score = scores[0]['rouge-l']['f']
    # 计算加权平均得分
    weighted_avg_score = weights["rouge-1"] * rouge_1_score + weights["rouge-2"] * rouge_2_score + weights[
        "rouge-l"] * rouge_l_score
    return weighted_avg_score


# 计算rouge距离矩阵
def getRougeMetric(traces, n):
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            if i == j:
                distances[i, j] = np.inf
            else:
                distances[i, j] = distances[j, i] = calRougeScore(traces[i], traces[j])
    return distances


def LOF(data, kn):
    ssss = datetime.now()
    traces = data['trace']
    freq = data['freq']
    n = len(traces)
    # 这里用的是距离不是相似度，所以用1减去相似度
    distances = 1 - getRougeMetric(traces, n)
    # 每个点的k临近距离
    k_distances = np.sort(distances, axis=0)[kn]
    k_distances[k_distances == 0] = np.inf
    # 计算第i个点到其第j个近邻的可达距离矩阵
    reachability_matrix = np.zeros((n, n))
    for i in range(n):
        # j小于等于k
        k_nearest_neighbors = np.argsort(distances[i])[:kn]
        for j in k_nearest_neighbors:
            if i != j:
                reachability_matrix[i, j] = max(k_distances[j], distances[i, j])
    # 局部可达密度，这里加入了log10来使轨迹发生频率的分布显得更平均一些（实际上对结果影响不大）注意用了np.add(x,1)，因为有些轨迹频率为1，log10(1)=0，所以计算时轨迹频率都要+1
    local_reachability_density = np.multiply(kn / np.sum(reachability_matrix, axis=1), np.log10(np.add(freq, 1)))
    lof_ratios = np.zeros(n)
    for i in range(n):
        k_nearest_neighbors = np.argsort(distances[i])[:kn]
        lof_ratios[i] = (np.sum(local_reachability_density[k_nearest_neighbors]) / kn) / local_reachability_density[i]
    eeee = datetime.now()
    return lof_ratios, round((eeee - ssss).total_seconds(), 6)

# # 用于作为F1值计算标准的rouge+dbscan
# def dbscanRouge(data, initCenter, eps):
#     traces = data['trace']
#     freq = np.log10(np.add(data['freq'], 1))
#     weight = (freq - np.min(freq)) / (np.max(freq) - np.min(freq))
#     cluster = [i for i in range(initCenter)]  # [0,1,2,3,4]
#     rm = rougeMetric.getRougeMetric(traces, len(traces))
#     noiseTraces = []
#     for i in range(initCenter, len(traces)):
#         maxSimi = max(rm[i][cluster[:]]) * 0.9 + weight[i] * 0.1
#         # print(maxSimi, "------", traces[i])
#         if maxSimi > eps:
#             cluster.append(i)
#         else:
#             noiseTraces.append(i)
#     return noiseTraces
