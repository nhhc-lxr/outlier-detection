from datetime import datetime
import random
import numpy as np


def get_cos_similarity(trace1, trace2):
    temp_trace1 = trace1.split(" ")
    temp_trace2 = trace2.split(" ")
    trace_union = list(set(temp_trace1).union(set(temp_trace2)))
    vector1 = np.zeros(len(trace_union))
    vector2 = np.zeros(len(trace_union))
    for i in range(len(trace_union)):
        vector1[i] = temp_trace1.count(trace_union[i])
        vector2[i] = temp_trace2.count(trace_union[i])
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def find_center_cos(traces, freq, cluster):
    temp_k, temp_trace_index = max(
        (([sum([get_cos_similarity(cluster[i], trace) * freq[traces.index(trace)] for trace in cluster])], i) for i in
         range(len(cluster))), key=lambda x: x[0])
    return temp_trace_index


def kmeans_cos(data, cos_eps, k):
    ssss = datetime.now()
    traces = data['trace'].tolist()
    freq = data['freq'].tolist()
    is_center_changed = True
    iter_time = 1
    centers = [traces[i] for i in range(k)]
    # centers = random.sample(traces, k)
    clusters = []
    inertia, inertia_after_iter = 0.0, 0.0
    while is_center_changed or inertia_after_iter < inertia or iter_time <= 10:
        inertia, inertia_after_iter = inertia_after_iter, 0.0
        clusters = [[val] for val in centers]
        is_center_changed = False
        for trace in traces:
            if trace in centers:
                continue
            temp, index = max([(get_cos_similarity(trace, centers[i]), i) for i in range(k)], key=lambda x: x[0])
            if temp >= cos_eps:
                inertia_after_iter += temp
                clusters[index].append(trace)
        for i in range(k):
            new_center_index = find_center_cos(traces, freq, clusters[i])
            if new_center_index == 0:
                continue
            is_center_changed = True
            clusters[i][0], clusters[i][new_center_index] = clusters[i][new_center_index], clusters[i][0]
            centers[i] = clusters[i][0]
        iter_time += 1
    eeee = datetime.now()
    return clusters, round((eeee - ssss).total_seconds(), 6)
