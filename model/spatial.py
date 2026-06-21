import json
import copy
import scipy
import openai
import numpy as np
import pandas as pd
import networkx as nx
import datetime
import folium
import requests
from python_tsp.heuristics import solve_tsp_simulated_annealing
from model.utils.funcs import get_max_summation_idx, get_top_k_sets, get_topk_location_pairs, find_clusters_containing_all_elements
from itertools import permutations
from pulp import LpVariable, LpProblem, LpMinimize, value, lpSum, LpBinary, PULP_CBC_CMD


class SpatialHandler:
    def __init__(self, data, min_clusters, min_pois, citywalk=False, citywalk_thresh=5000):
        self.data = data
        self.min_pois = min_pois
        self.min_clusters = min_clusters
        self.citywalk = citywalk
        self.citywalk_thresh = citywalk_thresh
        # 真实道路距离缓存: {(id1, id2): distance_m}
        self._road_dist_cache = {}

    def remove_outliers(self, poi_candidates: list, selected_clusters: list):
        # Fetch the coordinates of the POI candidates
        coordinates = self.data.loc[poi_candidates, ["x", "y"]].astype('float').to_numpy()
        
        # Calculate the centroid of all coordinates
        centroid = np.mean(coordinates, axis=0)
        
        # Calculate the distances of each point to the centroid
        distances = np.linalg.norm(coordinates - centroid, axis=1)
        
        # Calculate mean and standard deviation of the distances
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        # Filter out outliers based on distance
        non_outliers = [poi for i, poi in enumerate(poi_candidates) if abs(distances[i] - mean_distance) <= 1.5 * std_distance]
        
        # Update clusters to remove outlier POIs
        filtered_clusters = []
        for cluster in selected_clusters:
            filtered_cluster = [poi for poi in cluster if poi in non_outliers]
            if filtered_cluster:
                filtered_clusters.append(filtered_cluster)
                
        return non_outliers, filtered_clusters

    def get_clusters(self, poi_idlist: list, thresh: int = 5000) -> list:
        """
        Identify clusters of points within a given distance threshold in a set of points.

        Args:
            poi_idlist (list): A list of unique point identifiers.
            thresh (int, optional): The distance threshold defining cluster membership. Defaults to 5000.

        Returns:
            list: A list of clusters, where each cluster is represented as a set of point identifiers.
        """
        data = self.data.loc[poi_idlist]
        coords = data[['x', 'y']].astype(float).to_numpy()

        dist_matrix = scipy.spatial.distance.cdist(coords, coords)
        np.fill_diagonal(dist_matrix, thresh + 100)
        N = len(coords)
        G = nx.Graph()
        for i in range(N):
            G.add_edge(i, i)
            for j in range(i+1, N):  # avoid duplicates and self-loops
                if dist_matrix[i, j] < thresh:
                    G.add_edge(i, j)

        all_clusters = []
        
        if G.number_of_edges() == 0:
            all_clusters = [[i for i in poi_idlist]]
            return all_clusters
        
        while G.number_of_nodes() > 0:    
            cliques = list(nx.find_cliques(G))
            index_of_longest = max(enumerate(cliques), key=lambda x: len(x[1]))[0]
            biggest_cluster_list = list(set(cliques[index_of_longest]))
            G.remove_nodes_from(biggest_cluster_list)
            all_clusters.append(set(np.array(poi_idlist)[np.array(biggest_cluster_list)].tolist()))
        
        return all_clusters
    
    def solve_tsp_with_start_end(self, dist_matrix, start_point, end_point):
        """
        Solves the Traveling Salesman Problem (TSP) with specified start and end points using the Linear Programming approach.

        The function aims to find the shortest possible route that visits every node once and returns to the original node, while starting and ending at specified points.

        Arguments:
        - self (object): The object instance on which this method is called.
        - dist_matrix (list of lists): A square matrix representing the distances between nodes. The element `dist_matrix[i][j]` represents the distance from node `i` to node `j`.
        - start_point (int): The index of the starting node for the route.
        - end_point (int): The index of the ending node for the route.

        Returns:
        - tuple: A tuple containing two elements:
            1. The total distance of the optimal route.
            2. A list of nodes representing the optimal path, starting from the `start_point` and ending at the `end_point`.
        """

        n = len(dist_matrix)
        dist = {(i, j): dist_matrix[i][j] for i in range(n) for j in range(n) if i != j}
        prob = LpProblem("TSP", LpMinimize)
        x = LpVariable.dicts('x', dist, 0, 1, LpBinary)
        prob += lpSum([x[(i, j)] * dist[(i, j)] for (i, j) in x])

        for k in range(n):
            if k == start_point:
                prob += lpSum([x[(k, i)] for i in range(n) if i != k]) == 1
                prob += lpSum([x[(i, k)] for i in range(n) if i != k]) == 0
            elif k == end_point:
                prob += lpSum([x[(k, i)] for i in range(n) if i != k]) == 0
                prob += lpSum([x[(i, k)] for i in range(n) if i != k]) == 1
            else:
                prob += lpSum([x[(k, i)] for i in range(n) if i != k]) == 1
                prob += lpSum([x[(i, k)] for i in range(n) if i != k]) == 1

        while True:
            prob.solve(PULP_CBC_CMD(msg=0))
            edges = [(i, j) for (i, j) in x if value(x[(i, j)]) > 0.5]
            G = nx.Graph()
            G.add_edges_from(edges)
            subtours = [c for c in nx.connected_components(G) if len(c) < n]
            if not subtours:
                break
            for s in subtours:
                prob += lpSum([x[(i, j)] for (i, j) in permutations(s, 2)]) <= len(s) - 1

        edges = [(i, j) for (i, j) in x if value(x[(i, j)]) > 0.5]
        path = [start_point]
        while len(edges) > 0:
            (i, j) = [edge for edge in edges if edge[0] == path[-1]][0]
            path.append(j)
            edges.remove((i, j))

        return value(prob.objective), path
    
    def _get_road_dist_matrix(self, poi_idlist: list) -> np.ndarray:
        """
        通过 OSRM 免费 API 获取 POI 之间的真实驾车距离矩阵。
        结果缓存于 self._road_dist_cache，同一 POI 对不重复请求。

        Args:
            poi_idlist: POI ID 列表（DataFrame 索引）

        Returns:
            np.ndarray: N×N 距离矩阵（米），失败时返回 None
        """
        n = len(poi_idlist)
        dist_matrix = np.zeros((n, n))

        # 收集需要查询的 POI 对（查缓存）
        pairs_to_query = []
        for i in range(n):
            for j in range(i + 1, n):
                key = (min(poi_idlist[i], poi_idlist[j]),
                       max(poi_idlist[i], poi_idlist[j]))
                if key in self._road_dist_cache:
                    d = self._road_dist_cache[key]
                    dist_matrix[i, j] = d
                    dist_matrix[j, i] = d
                else:
                    pairs_to_query.append((i, j, key))

        if not pairs_to_query:
            return dist_matrix

        # 批量查询 OSRM
        print(f"  🚗 OSRM 真实道路距离：查询 {len(pairs_to_query)} 对...", end=" ")
        queried = 0
        failed = 0
        for i, j, key in pairs_to_query:
            id_i = poi_idlist[i]
            id_j = poi_idlist[j]
            try:
                lon1, lat1 = (float(self.data.loc[id_i, "lon"]),
                              float(self.data.loc[id_i, "lat"]))
                lon2, lat2 = (float(self.data.loc[id_j, "lon"]),
                              float(self.data.loc[id_j, "lat"]))
            except Exception:
                failed += 1
                continue

            d = self._fetch_osrm_distance(lon1, lat1, lon2, lat2)
            if d is not None:
                self._road_dist_cache[key] = d
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
                queried += 1
            else:
                failed += 1
                # fallback: 用直线距离 * 1.4（城市道路系数）
                loc_i = self.data.loc[id_i][["x", "y"]].astype(float).to_numpy()
                loc_j = self.data.loc[id_j][["x", "y"]].astype(float).to_numpy()
                d_euclid = np.linalg.norm(loc_i - loc_j)
                dist_matrix[i, j] = d_euclid * 1.4
                dist_matrix[j, i] = d_euclid * 1.4

        print(f"✅ {queried} 成功" + (f", {failed} 失败(已fallback)" if failed else ""))
        return dist_matrix

    def _fetch_osrm_distance(self, lon1, lat1, lon2, lat2) -> int | None:
        """调 OSRM 获取两点驾车距离（米），失败返回 None"""
        try:
            url = (f"https://router.project-osrm.org/route/v1/driving/"
                   f"{lon1},{lat1};{lon2},{lat2}?overview=false")
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                return int(data["routes"][0]["distance"])
        except Exception:
            pass
        return None

    def get_tsp_order(self, poi_candidates_list: list = None, locs: list = None):
        """
        Find the optimal traveling salesman problem (TSP) order for a list of points.

        Args:
            poi_candidates_list (list): A list of point identifiers for candidate locations.
                                        If None, only locs must be provided (used for cluster centroids).

        Returns:
            tuple: A tuple containing the following elements:
                - np.ndarray: An array representing the optimal TSP order of point identifiers.
                - np.ndarray: An array containing the coordinates (x, y) of the candidate locations, with shape (n, 2).
                - np.ndarray: A distance matrix representing pairwise distances between candidate locations.
        """

        if locs is None:
            locs = self.data.loc[poi_candidates_list, ["x", "y"]].astype(float).to_numpy()

        # ---- 旧：直线距离（保留注释）----
        # dist_matrix = scipy.spatial.distance.cdist(locs, locs)

        # ---- 新：OSRM 真实道路距离（需要 poi_ids；聚类中心点退回直线）----
        if poi_candidates_list is not None and len(poi_candidates_list) >= 2:
            dist_matrix = self._get_road_dist_matrix(poi_candidates_list)
        else:
            dist_matrix = None

        if dist_matrix is None:
            # fallback: 直线距离（聚类中心点或 OSRM 失败时）
            dist_matrix = scipy.spatial.distance.cdist(locs, locs)

        if locs.shape[0] > 2:
            order, distance = solve_tsp_simulated_annealing(dist_matrix)
        elif locs.shape[0] == 2:
            order = [0, 1]
        else:
            order = [0]

        return np.array(order), locs, dist_matrix
    
    def get_cluster_centroids(self, clusters: list, lonlat: bool = False) -> list:
        """
        Computes the centroids for the given clusters.

        Arguments:
        - self (object): The object instance on which this method is called.
        - clusters (list): A list of clusters, where each cluster is a subset of the data's index.
        - lonlat (bool, optional): Flag indicating if the coordinates should be in "longitude, latitude" format. If set to False, "x, y" format will be used. Default is False.

        Returns:
        - list: A list of cluster centroids. Each centroid is represented as a list containing two elements, the x (or longitude) and y (or latitude) coordinates.
        """

        if lonlat:
            return [np.mean(self.data.loc[cluster][["lon", "lat"]].to_numpy(), axis=0).tolist() for cluster in clusters]
        else:
            return [np.mean(self.data.loc[cluster][["x", "y"]].astype(float).to_numpy(), axis=0).tolist() for cluster in clusters]
    
    def get_poi_pairs_across_clusters(self, clusters_order: np.ndarray, clusters: list):
        """
        Retrieves the top pairs of Points of Interest (POIs) across the given order of clusters.

        Arguments:
        - self (object): The object instance on which this method is called.
        - clusters_order (np.ndarray): An ordered array representing the sequence in which clusters should be considered.
        - clusters (list): A list of clusters, where each cluster is a subset of the data's index.

        Returns:
        - list: A list of top pairs of POIs between consecutive clusters in the given order. Each pair is represented as a list containing two elements, the indices of the POIs in their respective clusters.
        """

        all_pairs = []
        for i in range(len(clusters_order)-1):
            locsCluster1, locsCluster2 = self.data.loc[clusters[clusters_order[i]]][["x", "y"]].astype(float).to_numpy(), self.data.loc[clusters[clusters_order[i+1]]][["x", "y"]].astype(float).to_numpy()
            pair = get_topk_location_pairs(locsCluster1, locsCluster2, k=min(3, locsCluster1.shape[0], locsCluster2.shape[0]))
            pair = [clusters[clusters_order[i]][pair[0][0]], clusters[clusters_order[i+1]][pair[0][1]]]
            all_pairs.append(pair)

        return all_pairs

    def get_poi_candidates(self, allpoi_idlist: list, must_see_poi_idlist: list, req_topk_pois: np.ndarray, min_num_candidate: int, thresh : int, pseudo_must_see_pois: list = []):
        """
        Perform cluster-based point-of-interest (POI) selection to generate a list of candidate POIs.

        Args:
            ori_req_topk_pois (numpy.ndarray): Original POI ranking with shape (N, 2), where the first column contains
                                            POI IDs and the second column contains their respective scores.
            allpoi_idlist (list): List of all POI identifiers.
            must_see_poi_idlist (list): List of must-see POI identifiers.
            min_num_candidate (int): Minimum number of candidate POIs to select.

        Returns:
            tuple: A tuple containing two elements:
                - list: A list of selected candidate POIs.
                - list: A list of scores corresponding to the selected candidate POIs.        
        """

        poi_candidates, num_clusters, selected_clusters, mark_citywalk = [], 9999, [], True
        cur_ids = req_topk_pois[:, 0].astype(int)

        for poi in pseudo_must_see_pois:
            if poi not in cur_ids:
                req_topk_pois = np.insert(req_topk_pois, 0, np.array([poi, 10]), axis=0)
            else:
                row_idx = cur_ids.tolist().index(poi)
                req_topk_pois[row_idx, 1] = 10

        for poi in must_see_poi_idlist:
            if poi not in cur_ids:
                req_topk_pois = np.insert(req_topk_pois, 0, np.array([poi, 1000]), axis=0)
            else:
                row_idx = cur_ids.tolist().index(poi)
                req_topk_pois[row_idx, 1] = 1000
        
        if self.citywalk:
            clusters = self.get_clusters(allpoi_idlist, thresh=self.citywalk_thresh)
            index_candidates = get_top_k_sets(clusters, req_topk_pois, k=min(len(clusters), 2))
            index = np.random.choice(index_candidates, size=1)[0] 
            selected_cluster = []

            for poi in clusters[index]:
                if poi not in poi_candidates:
                    selected_cluster.append(poi)
                    poi_candidates.append(poi)

            for poi in must_see_poi_idlist:
                if poi not in selected_cluster or len(poi_candidates) < self.min_pois - 2:
                    mark_citywalk = False
                    break
            
            for poi in pseudo_must_see_pois:
                if poi not in selected_cluster or len(poi_candidates) < self.min_pois - 2:
                    mark_citywalk = False
                    break
            
            if len(selected_cluster) > 0 and mark_citywalk:
                selected_clusters.append(selected_cluster)

        if not mark_citywalk or not self.citywalk or len(poi_candidates) < 10:
            poi_candidates, selected_clusters = [], []
            clusters = self.get_clusters(allpoi_idlist, thresh=thresh)
            # The following code guarantees the inclusion of all user-requested POIs in the candidate set.
            merge_must_see_poi_idlist = []
            merge_must_see_poi_idlist.extend(must_see_poi_idlist)

            if pseudo_must_see_pois is not None:
                merge_must_see_poi_idlist.extend(pseudo_must_see_pois)

            merge_must_see_poi_idlist = list(set(merge_must_see_poi_idlist))
            idx = find_clusters_containing_all_elements(clusters, merge_must_see_poi_idlist)
            
            for index in idx:
                selected_cluster = []
                for poi in clusters[index]:
                    if poi not in poi_candidates:
                        selected_cluster.append(poi)
                        poi_candidates.append(poi)
                if len(selected_cluster) > 0:
                    selected_clusters.append(selected_cluster)

            if len(idx) <= self.min_clusters or len(poi_candidates) < min(min_num_candidate, self.data.shape[0]):
                while True: # this loop will end if <<< len(poi_candidates) > min_num_candidate >>> or <<< no remaining candidates in clusters >>>
                    index_candidates = get_top_k_sets(clusters, req_topk_pois, k=min(len(clusters), 2))

                    if len(index_candidates) == 0 or (len(selected_clusters) > self.min_clusters and len(poi_candidates) >= min(min_num_candidate, self.data.shape[0])):
                        break

                    index = np.random.choice(index_candidates, size=1)[0] # this introduces some randomness        
                    selected_cluster = []
                    for poi in clusters[index]:
                        if poi not in poi_candidates:
                            selected_cluster.append(poi)
                            poi_candidates.append(poi)

                    if len(selected_cluster) > 0:
                        selected_clusters.append(selected_cluster)

                    clusters.pop(index)

            poi_candidates, selected_clusters = self.remove_outliers(poi_candidates, selected_clusters)

        for must_see_poi in must_see_poi_idlist:
            if must_see_poi not in poi_candidates:
                poi_candidates.append(must_see_poi)

        poi_candidatescores = []
        for poi in poi_candidates:
            if poi in must_see_poi_idlist:
                poi_candidatescores.append(1000)
            else:
                poi_candidatescores.append(req_topk_pois[req_topk_pois[:, 0] == poi, 1].tolist()[0])
                
        return poi_candidates, poi_candidatescores, selected_clusters, mark_citywalk


# ==================== 多日规划扩展 ====================

    def get_multiday_clusters(self,
                               allpoi_idlist: list,
                               num_days: int,
                               thresh: int = 5000) -> list[list]:
        """
        将 POI 列表按空间邻近性划分为 N 个"日簇"（每天一个地理区域）。
        
        算法思路：
        1. 先用小阈值聚类得到细粒度空间簇
        2. 将细粒度簇合并为 N 个超簇，使各超簇 POI 数量尽量均衡
        3. 每个超簇对应一天的行程

        Args:
            allpoi_idlist: 全部候选 POI ID 列表
            num_days: 行程天数
            thresh: 聚类距离阈值（米）

        Returns:
            list[list]: 每天对应的 POI ID 列表，长度为 num_days
        """
        # 先做细粒度聚类
        fine_clusters = self.get_clusters(allpoi_idlist, thresh=thresh)
        # 将 set 转为 list，pandas .loc[] 不接受 set 索引
        fine_clusters = [list(c) for c in fine_clusters]

        # 如果聚类数 <= 天数，直接按聚类分配
        if len(fine_clusters) <= num_days:
            day_groups = [[poi for cluster in fine_clusters[i:i+1] for poi in cluster]
                          for i in range(len(fine_clusters))]
            # 不足天数则补空组
            while len(day_groups) < num_days:
                day_groups.append([])
            # 重新平衡：从 POI 多的天移一些到少的天
            day_groups = self._balance_day_groups(day_groups, allpoi_idlist)
            return day_groups

        # 聚类数 > 天数：合并邻近聚类
        # 计算聚类中心
        centroids = self.get_cluster_centroids(fine_clusters)
        centroids_arr = np.array(centroids)

        # 用贪心合并：每次合并距离最近的两个聚类
        from itertools import combinations
        cluster_sets = [set(c) for c in fine_clusters]

        while len(cluster_sets) > num_days:
            # 找距离最近的两个聚类
            min_dist = float('inf')
            merge_pair = (0, 1)
            for i, j in combinations(range(len(cluster_sets)), 2):
                # 用两个聚类中最近 POI 对的距离作为聚类间距离
                locs_i = self.data.loc[list(cluster_sets[i])][["x", "y"]].astype(float).to_numpy()
                locs_j = self.data.loc[list(cluster_sets[j])][["x", "y"]].astype(float).to_numpy()
                dist = np.min(scipy.spatial.distance.cdist(locs_i, locs_j))
                if dist < min_dist:
                    min_dist = dist
                    merge_pair = (i, j)

            # 合并
            i, j = merge_pair
            cluster_sets[i] = cluster_sets[i] | cluster_sets[j]
            cluster_sets.pop(j)

        # 转换为列表，按 POI 数量降序排列
        day_groups = [list(c) for c in cluster_sets]
        day_groups.sort(key=len, reverse=True)

        # 确保返回 num_days 个组
        while len(day_groups) < num_days:
            day_groups.append([])

        day_groups = self._balance_day_groups(day_groups, allpoi_idlist)

        return day_groups

    def _balance_day_groups(self,
                            day_groups: list[list],
                            allpoi_idlist: list) -> list[list]:
        """
        平衡各天 POI 数量：从 POI 最多的天向最少的天转移，使每日 POI 数尽量均匀。

        策略：
        1. 计算目标每夭 POI 数 = ceil(总POI / 天数)
        2. 从超过目标的组中，选择与目标组空间最近的 POI 转移过去
        3. 确保 must-see POI 留在原组（已在调用方处理）

        Args:
            day_groups: 每天 POI 分组
            allpoi_idlist: 全部 POI（用于补充空组）

        Returns:
            平衡后的分组（list of lists）
        """
        num_days = len(day_groups)
        total_pois = sum(len(g) for g in day_groups)
        if total_pois == 0:
            return day_groups

        target_per_day = max(1, int(np.ceil(total_pois / num_days)))

        # 找出剩余未分配的 POI
        used_pois = set(poi for group in day_groups for poi in group)
        remaining = list(set(allpoi_idlist) - used_pois)

        # 先填充空组
        for i, group in enumerate(day_groups):
            if len(group) == 0 and remaining:
                # 从 remaining 中取
                take = min(target_per_day, len(remaining))
                day_groups[i] = remaining[:take]
                remaining = remaining[take:]

        # 迭代平衡：从超标的组中移出 POI 到不足的组
        max_iterations = 50
        for _ in range(max_iterations):
            over = [(i, g) for i, g in enumerate(day_groups) if len(g) > target_per_day]
            under = [(i, g) for i, g in enumerate(day_groups) if len(g) < target_per_day]

            if not over or not under:
                break

            # 从超标最多的组移 POI 到最缺的组
            over_idx, over_group = max(over, key=lambda x: len(x[1]))
            under_idx, under_group = min(under, key=lambda x: len(x[1]))

            # 一次转移多个 POI，加速平衡
            excess = len(over_group) - target_per_day
            deficit = target_per_day - len(under_group)
            move_count = max(1, min(excess, deficit, 5))  # 每次最多移5个

            # 选择 over_group 中与 under_group 空间最近的 move_count 个 POI
            if len(under_group) > 0:
                under_centroid = np.mean(
                    self.data.loc[under_group][["x", "y"]].astype(float).to_numpy(),
                    axis=0
                )
                over_locs = self.data.loc[over_group][["x", "y"]].astype(float).to_numpy()
                dists = np.linalg.norm(over_locs - under_centroid, axis=1)
                move_indices = np.argsort(dists)[:move_count]
            else:
                move_indices = list(range(min(move_count, len(over_group))))

            # 从后往前删，避免索引错乱
            for idx in sorted(move_indices, reverse=True):
                if idx < len(over_group):
                    day_groups[under_idx].append(over_group.pop(idx))

        # 把 remaining 全部分配到最少的组
        while remaining:
            min_idx = min(range(num_days), key=lambda i: len(day_groups[i]))
            day_groups[min_idx].append(remaining.pop(0))

        return day_groups

    def get_multiday_poi_candidates(self,
                                     allpoi_idlist: list,
                                     must_see_poi_idlist: list,
                                     req_topk_pois: np.ndarray,
                                     num_days: int,
                                     min_pois_per_day: int,
                                     thresh: int = 5000) -> tuple:
        """
        多日行程的 POI 候选选择。
        
        与单日版 (get_poi_candidates) 的区别：
        - 为每天分别选择候选 POI
        - 确保 must-see POI 被分配到最合适的日期
        - 跨天不重复

        Args:
            allpoi_idlist: 全部 POI ID
            must_see_poi_idlist: 必去 POI ID
            req_topk_pois: 排序后的 POI 得分
            num_days: 天数
            min_pois_per_day: 每天最少 POI 数
            thresh: 距离阈值

        Returns:
            tuple: (day_poi_candidates, day_poi_scores, day_clusters, mark_citywalk)
                   每个元素都是长度为 num_days 的列表
        """
        # 先将 POI 按空间分成 N 组
        day_groups_raw = self.get_multiday_clusters(
            allpoi_idlist, num_days, thresh
        )

        # 确保 must-see POI 在合适的组中
        for must_poi in must_see_poi_idlist:
            found = False
            for group in day_groups_raw:
                if must_poi in group:
                    found = True
                    break
            if not found:
                # 找距离 must_poi 最近的非空组
                must_loc = self.data.loc[must_poi][["x", "y"]].astype(float).to_numpy()
                best_group_idx = 0
                best_dist = float('inf')
                for i, group in enumerate(day_groups_raw):
                    if len(group) == 0:
                        continue
                    locs = self.data.loc[group][["x", "y"]].astype(float).to_numpy()
                    dist = np.min(np.linalg.norm(locs - must_loc, axis=1))
                    if dist < best_dist:
                        best_dist = dist
                        best_group_idx = i
                day_groups_raw[best_group_idx].append(must_poi)

        # 为每天构建候选列表
        day_poi_candidates = []
        day_poi_scores = []
        day_clusters_list = []
        mark_citywalk = True  # 多日规划默认启用 citywalk

        for day_idx, group in enumerate(day_groups_raw):
            if len(group) == 0:
                day_poi_candidates.append([])
                day_poi_scores.append([])
                day_clusters_list.append([])
                continue

            # 在组内聚类
            try:
                clusters = self.get_clusters(group, thresh=thresh)
            except Exception:
                clusters = [group]

            # 选择候选（简化的单日逻辑）
            candidates = []
            candidate_scores = []
            selected_clusters = []

            for cluster in clusters:
                selected = []
                for poi in cluster:
                    if poi not in candidates:
                        candidates.append(poi)
                        selected.append(poi)
                        # 得分：must-see 优先
                        if poi in must_see_poi_idlist:
                            candidate_scores.append(1000)
                        else:
                            score_row = req_topk_pois[req_topk_pois[:, 0].astype(int) == poi]
                            score = float(score_row[0, 1]) if len(score_row) > 0 else 1.0
                            candidate_scores.append(score)
                if selected:
                    selected_clusters.append(selected)

            day_poi_candidates.append(candidates)
            day_poi_scores.append(candidate_scores)
            day_clusters_list.append(selected_clusters)

        return day_poi_candidates, day_poi_scores, day_clusters_list, mark_citywalk

    def get_multiday_full_order(self,
                                 day_poi_candidates: list[list],
                                 day_clusters: list[list],
                                 start_poi_per_day: list = None,
                                 end_poi_per_day: list = None) -> list[list]:
        """
        为每一天分别计算 TSP 最优访问顺序。

        Args:
            day_poi_candidates: 每天候选 POI 列表
            day_clusters: 每天聚类列表
            start_poi_per_day: 每天起点（None 则自动选）
            end_poi_per_day: 每天终点（None 则自由）

        Returns:
            list[list]: 每天排序后的 POI ID 列表
        """
        all_day_orders = []

        for day_idx in range(len(day_poi_candidates)):
            candidates = day_poi_candidates[day_idx]
            clusters = day_clusters[day_idx]

            if len(candidates) == 0:
                all_day_orders.append([])
                continue

            if len(candidates) == 1:
                all_day_orders.append(candidates)
                continue

            # 聚类间顺序
            if len(clusters) > 1:
                centroids = self.get_cluster_centroids(clusters)
                cluster_order, _, _ = self.get_tsp_order(locs=np.array(centroids))
            else:
                cluster_order = [0]

            # 聚类内 TSP
            day_order = []
            for ci, cluster_idx in enumerate(cluster_order):
                if cluster_idx >= len(clusters):
                    continue
                cluster = clusters[cluster_idx]
                if len(cluster) <= 2:
                    day_order.extend(cluster)
                else:
                    order_in, _, _ = self.get_tsp_order(cluster)
                    day_order.extend(np.array(cluster)[order_in].tolist())

            # 去重
            seen = set()
            day_order_unique = []
            for poi in day_order:
                if poi not in seen:
                    seen.add(poi)
                    day_order_unique.append(poi)
            all_day_orders.append(day_order_unique)

        return all_day_orders
