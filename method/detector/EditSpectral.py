from datetime import datetime
import numpy as np
from multiprocessing import Process, cpu_count
import multiprocessing


def get_edit_distance(trace1, trace2):
    trace1 = trace1.split(" ")
    trace2 = trace2.split(" ")
    matrix_ed = np.zeros((len(trace1) + 1, len(trace2) + 1), dtype=int)
    matrix_ed[0] = np.arange(len(trace2) + 1)
    matrix_ed[:, 0] = np.arange(len(trace1) + 1)
    for i in range(1, len(trace1) + 1):
        for j in range(1, len(trace2) + 1):
            # 表示删除a_i
            dist_1 = matrix_ed[i - 1, j] + 1
            # 表示插入b_i
            dist_2 = matrix_ed[i, j - 1] + 1
            # 表示替换b_i
            dist_3 = matrix_ed[i - 1, j - 1] + (1 if trace1[i - 1] != trace2[j - 1] else 0)
            # 取最小距离
            matrix_ed[i, j] = np.min([dist_1, dist_2, dist_3])
    return matrix_ed[-1, -1]


def normalization(matrix):  # 归一化
    return matrix / np.sqrt(np.sum(matrix ** 2, axis=1, keepdims=True))  # 求数组的正平方根


def euclid_dist(vector_a, vector_b, sqrt_flag=False):
    dist = sum([(vector_a[i] - vector_b[i]) ** 2 for i in range(len(vector_a))])
    if sqrt_flag:
        dist = np.sqrt(dist)
    return dist


def spectral_clustering(data, edit_eps, k):
    ssss = datetime.now()
    traces = data['trace'].tolist()
    freq = data['freq'].tolist()
    H = process_data_(traces, k).tolist()
    result = []
    kmeans_result = kmeans_edit_dist(freq, H, edit_eps, k)
    top = 0
    for cluster in kmeans_result:
        result.append([])
        for vector in cluster:
            result[top].append(traces[H.index(vector)])
        top += 1
    eeee = datetime.now()
    return result, round((eeee - ssss).total_seconds(), 6)


def run_faster(all_traces, start, end, total, k, result):
    neighbour_matrix = np.zeros((total, total))
    for i in range(start, end):
        for j in range(total):
            if i != j:  # 计算不重复点的距离
                neighbour_matrix[i][j] = neighbour_matrix[j][i] = 1 - get_edit_distance(all_traces[i],
                                                                                        all_traces[j]) / max(
                    len(all_traces[i]), len(all_traces[j]))
        t = np.argsort(neighbour_matrix[i, :])
        for x in range(total - k):
            neighbour_matrix[i][t[x]] = 0
        result.append(neighbour_matrix[i])


def process_data_(all_traces, k=500):
    # ssss = datetime.now()
    traces_type_num = len(all_traces)
    core_num = cpu_count()
    size = int(traces_type_num / core_num)
    remain = traces_type_num % core_num
    result_list = []
    process_list = []
    for i in range(core_num):
        result_list.append(multiprocessing.Manager().list())
    for i in range(core_num):
        if i == core_num - 1:
            process_list.append(
                Process(target=run_faster,
                        args=(all_traces, i * size, (i + 1) * size + remain, traces_type_num, k, result_list[i])))
            continue
        process_list.append(
            Process(target=run_faster, args=(all_traces, i * size, (i + 1) * size, traces_type_num, k, result_list[i])))
    for i in range(core_num):
        process_list[i].start()
    for i in range(core_num):
        process_list[i].join()
    for i in range(core_num):
        result_list[i] = np.array(result_list[i])
    result0 = np.vstack(result_list)
    neighbour_matrix = (result0 + result0.T) / 2
    # 创建拉普拉斯矩阵
    degree_matrix = np.sum(neighbour_matrix, axis=1)
    laplacian_matrix = np.diag(degree_matrix) - neighbour_matrix
    sqrt_degree_matrix = np.diag(1.0 / (degree_matrix ** 0.5))
    laplacian_matrix = np.dot(np.dot(sqrt_degree_matrix, laplacian_matrix), sqrt_degree_matrix)
    # 特征值、特征向量分解
    eigen_values, eigen_vectors = np.linalg.eig(laplacian_matrix)
    e_i = np.argsort(eigen_values)
    H = normalization(np.vstack([eigen_vectors[:, i] for i in e_i[:100]]).T)
    # eeee = datetime.now()
    # print("耗时", (eeee - ssss).seconds, "秒")
    return H


def find_center(vectors_dict, cluster, matrix):
    min_dist_val, min_dist_index = float("inf"), -1
    for i in range(len(cluster)):
        vector = cluster[i]
        dist_val = sum(
            [euclid_dist(vector, cluster[j]) * vectors_dict[matrix.index(cluster[j])] for j in range(len(cluster))])
        if min_dist_val > dist_val:
            min_dist_val = dist_val
            min_dist_index = i
    return min_dist_index


def kmeans_edit_dist(freq, matrix, edit_eps, k):
    # ssss = datetime.now()
    is_center_changed = True
    iter_time = 1
    inertia, inertia_after_iter = 0.0, 0.0
    centers = [matrix[i] for i in range(k)]
    vectors_dict: dict = {}
    for i in range(len(matrix)):
        vectors_dict[i] = freq[i]
    while is_center_changed or inertia_after_iter != inertia or iter_time <= 3:
        inertia, inertia_after_iter = inertia_after_iter, 0.0
        clusters = [[val] for val in centers]
        is_center_changed = False
        for vector in matrix:
            if vector in centers:
                continue
            temp, index = min([(euclid_dist(vector, centers[i]), i) for i in range(k)], key=lambda x: x[0])
            if temp <= edit_eps:
                inertia_after_iter += temp
                clusters[index].append(vector)
        for i in range(k):
            new_center_index = find_center(vectors_dict, clusters[i], matrix)
            if new_center_index == 0:
                continue
            is_center_changed = True
            clusters[i][0], clusters[i][new_center_index] = clusters[i][new_center_index], clusters[i][0]
            centers[i] = clusters[i][0]
        iter_time += 1
    # eeee = datetime.now()
    return clusters
