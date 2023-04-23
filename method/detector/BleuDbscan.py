# 计算两条轨迹之间的Bleu值
import collections
import math
from datetime import datetime


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
def dbscan(data, bleu_eps=0.6, size_eps=0.3, init_center=3, min_clust_size=10):
    ssss = datetime.now()
    traces = data['trace'].tolist()
    freq = data['freq'].tolist()
    cluster_groups = [[traces[i]] for i in range(init_center)]
    for i in range(init_center, len(traces)):
        trace = traces[i]
        best_bleu, best_index = max((calBleu(trace, cluster_groups[i][0]), i) for i in range(len(cluster_groups)))
        is_bleu_sati = best_bleu >= bleu_eps
        is_size_sati = freq[traces.index(cluster_groups[best_index][0])] * size_eps >= freq[traces.index(trace)]
        if is_bleu_sati and is_size_sati:
            cluster_groups[best_index].append(trace)
            new_center_index = findCenterBleu(traces, freq, cluster_groups[best_index], 3)
            cluster_groups[best_index][0], cluster_groups[best_index][new_center_index] = \
                cluster_groups[best_index][new_center_index], cluster_groups[best_index][0]
            continue
        if ((is_bleu_sati and not is_size_sati) or (not is_bleu_sati)) and freq[traces.index(trace)] >= min_clust_size:
            cluster_groups.append([trace])
            continue
    eeee = datetime.now()
    return cluster_groups, round((eeee - ssss).total_seconds(), 2)


# 多对多，返回中心点对应的字符串
def findCenterBleu(traces, freq, cluster, n=3):
    temp_k, temp_trace_index = max(
        (([sum([calBleu(cluster[i], trace, n) * freq[traces.index(trace)] for trace in cluster])], i) for i in
         range(len(cluster))), key=lambda x: x[0])
    return temp_trace_index
