import collections
import math
import pandas as pd
import numpy as np
from typing import List
from rouge import Rouge

rouge = Rouge()

# 定义指标的权重
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


# 一个简单的单例模式存储rouge距离矩阵避免重复计算
class RougeMetric:
    def __init__(self):
        self.distances = None

    def getRougeMetric(self, traces, n):
        if self.distances is None:
            distances = np.zeros((n, n))
            for i in range(n):
                print(i)
                for j in range(i, n):
                    if i == j:
                        distances[i, j] = np.inf
                    else:
                        distances[i, j] = distances[j, i] = calRougeScore(traces[i], traces[j])
            self.distances = distances
        return self.distances


rougeMetric = RougeMetric()


# 计算两条轨迹之间的Bleu值
def calBleu(cand, refer, n=3):
    if cand == refer:
        return 1.0
    tmp_cand = cand.split(' ')
    temp_ref = refer.split(' ')
    cand_len = len(tmp_cand)
    ref_len = len(temp_ref)
    score = 0.0
    for i in range(n):
        times = 0
        ref_dict = collections.defaultdict(int)
        for k in range(ref_len - i):
            w = " ".join(temp_ref[k: k + i + 1])
            if ref_dict[w] is None:
                ref_dict[w] = 1
            else:
                ref_dict[w] += 1
        for k in range(cand_len - i):
            w = " ".join(tmp_cand[k: k + i + 1])
            if ref_dict.get(w) is not None and ref_dict[w] > 0:
                times += 1
                ref_dict[w] -= 1
        gram = times / (cand_len - i)
        if gram == 0:
            return 0.0
        score += math.log(gram)
    score /= n
    bp = math.exp(min(0, 1 - ref_len / cand_len))
    return math.exp(score) * bp


# Bleu的DBSCAN算法
def dbscan(traces_dict, all_traces,
           init_center, bleu_eps, len_eps,
           min_clust_size):
    cluster_groups = [[all_traces[i]] for i in range(init_center)]
    noise_traces = []
    for i in range(init_center, len(all_traces)):
        trace = all_traces[i]
        best_bleu, best_index = max((calBleu(trace, cluster_groups[i][0]), i) for i in range(len(cluster_groups)))
        is_bleu_sati = best_bleu >= bleu_eps
        is_len_sati = traces_dict[cluster_groups[best_index][0]] * len_eps >= traces_dict[trace]
        if is_bleu_sati and is_len_sati:
            cluster_groups[best_index].append(trace)
            new_center_index = findCenterBleu(traces_dict, cluster_groups[best_index], 3)
            cluster_groups[best_index][0], cluster_groups[best_index][new_center_index] = \
                cluster_groups[best_index][new_center_index], cluster_groups[best_index][0]
            continue
        if ((is_bleu_sati and not is_len_sati) or (not is_bleu_sati)) and traces_dict[trace] >= min_clust_size:
            cluster_groups.append([trace])
            continue
        noise_traces.append(trace)
    return noise_traces


# 多对多，返回中心点对应的字符串
def findCenterBleu(traces_dict, cluster, n=3):
    temp_k, temp_trace_index = max(
        (([sum([calBleu(cluster[i], trace, n) * traces_dict[trace] for trace in cluster])], i) for i in
         range(len(cluster))), key=lambda x: x[0])
    return temp_trace_index


# 用于作为F1值计算标准的rouge+dbscan
def dbscanRouge(data, initCenter, eps):
    traces = data['trace']
    freq = np.log10(np.add(data['freq'], 1))
    weight = (freq - np.min(freq)) / (np.max(freq) - np.min(freq))
    cluster = [i for i in range(initCenter)]  # [0,1,2,3,4]
    rm = rougeMetric.getRougeMetric(traces, len(traces))
    noiseTraces = []
    for i in range(initCenter, len(traces)):
        maxSimi = max(rm[i][cluster[:]]) * 0.9 + weight[i] * 0.1
        print(maxSimi, "------", traces[i])
        if maxSimi > eps:
            cluster.append(i)
        else:
            noiseTraces.append(i)
    return noiseTraces


def LOF(data, k):
    traces = data['trace']
    freq = data['freq']
    n = len(traces)
    # 这里用的是距离不是相似度，所以用1减去相似度
    distances = 1 - rougeMetric.getRougeMetric(traces, n)
    # 每个点的k临近距离
    k_distances = np.sort(distances, axis=0)[k]
    k_distances[k_distances == 0] = np.inf
    # 计算第i个点到其第j个近邻的可达距离矩阵
    reachability_matrix = np.zeros((n, n))
    for i in range(n):
        # j小于等于k
        k_nearest_neighbors = np.argsort(distances[i])[:k]
        for j in k_nearest_neighbors:
            if i != j:
                reachability_matrix[i, j] = max(k_distances[j], distances[i, j])
    # 局部可达密度，这里加入了log10来使轨迹发生频率的分布显得更平均一些（实际上对结果影响不大）注意用了np.add(x,1)，因为有些轨迹频率为1，log10(1)=0，所以计算时轨迹频率都要+1
    local_reachability_density = np.multiply(k / np.sum(reachability_matrix, axis=1), np.log10(np.add(freq, 1)))
    lof_ratios = np.zeros(n)
    for i in range(n):
        k_nearest_neighbors = np.argsort(distances[i])[:k]
        lof_ratios[i] = (np.sum(local_reachability_density[k_nearest_neighbors]) / k) / local_reachability_density[i]
    return lof_ratios


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
    return (2 * TP) / (2 * TP + FP + FN)


# def initSimData():
#     file = open("data/generatedTraces.json")
#     ori2dev_dict = json.loads(file.read())
#     file.close()
#     # 创建数据变量
#     all_traces: List[str] = []
#     traces_dict: dict = {}
#     traces_num = 0
#     for key in ori2dev_dict:
#         oriTra = key.replace(',', ' ')
#         all_traces.append(oriTra)
#         ori_trace_num = random.randint(1500, 2500)
#         traces_dict[oriTra] = ori_trace_num
#         traces_num += ori_trace_num
#         for t in ori2dev_dict[key]:
#             t = t.replace(',', ' ')
#             dev_trace_num = random.randint(1, 20)
#             traces_num += dev_trace_num
#             if traces_dict.get(t) is None:
#                 all_traces.append(t)
#                 traces_dict[t] = dev_trace_num
#             else:
#                 traces_dict[t] += dev_trace_num
#     all_traces.sort(key=lambda x: traces_dict[x], reverse=True)
#     cluster_groups, noise_traces = dbscan(traces_dict, all_traces, 5, 0.5, 0.5, 100)
#     data = [[trace, (0 if (trace in noise_traces) else 1), traces_dict[trace]] for trace in all_traces]
#     dataframe = pd.DataFrame(data, columns=['trace', 'label', 'freq'])
#     dataframe.to_csv("data/genTraces.csv", index=None)
#     return


# 真实数据集读取
def realLogLoader(inputFile):
    case_id = "case"
    activity = "event"
    event_log = pd.read_csv(inputFile).loc[:, [case_id, activity]]
    # 创建数据变量
    all_traces: List[str] = []
    traces_dict: dict = {}
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
            val = [s.replace(" ", "_") for s in val]
            activities = ' '.join(val)
            if traces_dict.get(activities) is None:
                all_traces.append(activities)
                traces_dict[activities] = 1
            else:
                traces_dict[activities] += 1
            traces_num += 1
    all_traces.sort(key=lambda trace: traces_dict[trace], reverse=True)
    data = [[trace, traces_dict[trace]] for trace in all_traces]
    df = pd.DataFrame(data, columns=['trace', 'freq'])
    noiseTraces = dbscanRouge(df, 5, 0.5)
    df['label'] = [0 if (x in noiseTraces) else 1 for x in range(len(data))]
    print('共有轨迹', sum(traces_dict.values()), '条，去重轨迹', len(all_traces), '条，活动', len(activity_dict.keys()),
          '个，事件',
          sum(activity_dict.values()),
          '次')
    return df


# k值就是k近邻，threshold值是lof值高于改阈值时被视作离群点
def stater(filename, k, threshold):
    # 读取CSV文件
    data = realLogLoader('data/realLog/' + filename)
    # 算法的效率从这里开始计算，读取数据的耗时就不要算了

    # 计算局部离群因子
    lof_scores = LOF(data, k)
    # 将结果添加到原始数据中
    data['outlier'] = lof_scores
    data['pred_label'] = data['outlier'].apply(lambda x: 0 if x > threshold else 1)
    # 计算F1分数
    f1 = f1Score(data['label'], data['pred_label'], data['freq'])
    print(f"F1 Score: {f1}")
    # 耗时到这里结束

    # 结果导出
    pd.DataFrame(data).to_csv('result/' + filename, index=None)
