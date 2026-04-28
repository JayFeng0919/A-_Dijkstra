import heapq
import matplotlib.pyplot as plt
import numpy as np
import time


class AGVPathPlanner:
    def __init__(self, map_grid, start, goal, use_heuristic=True):
        """
        初始化路径规划器
        :param map_grid: 2D numpy array (0=空地, 1=障碍物)
        :param start: (row, col) 起点坐标
        :param goal: (row, col) 终点坐标
        :param use_heuristic: True=使用A*(有启发), False=使用Dijkstra(无启发)
        """
        self.map = map_grid
        self.start = start
        self.goal = goal
        self.use_heuristic = use_heuristic  # 算法开关
        self.rows, self.cols = map_grid.shape
        # 8个移动方向: (行变化, 列变化, 移动距离)
        self.movements = [
            (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),  # 直线
            (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)  # 对角线
        ]

    def heuristic(self, a, b):
        if not self.use_heuristic:
            return 0                                                   # 如果是 Dijkstra 算法，强制启发函数为 0
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)        # A* 使用欧几里得距离作为启发函数

    def plan_path(self):
        # 优先队列 (F_cost, G_cost, current_node)
        open_list = []
        heapq.heappush(open_list, (0, 0, self.start))

        came_from = {}          # 记录父节点以便回溯路径: came_from[node] = parent
        g_score = {self.start: 0}           # 记录从起点到当前点的代价 G
        f_score = {self.start: self.heuristic(self.start, self.goal)}          # 记录 F = G + H
        closed_set = set()          # 记录已经探索过的点

        nodes_visited = 0              # 统计探索的节点数量（衡量效率的关键指标）

        while open_list:
            current_f, current_g, current = heapq.heappop(open_list)    # 取出F值最小的节点

            if current in closed_set:
                continue                    # 只有从 open_list 取出并放入 closed_set 时才算真正“访问”

            nodes_visited += 1
            closed_set.add(current)

            if current == self.goal:                # 返回：路径列表, 探索节点数, 总距离
                return self.reconstruct_path(came_from, current), nodes_visited, g_score[current]

            # 遍历邻居
            for dr, dc, move_cost in self.movements:
                neighbor = (current[0] + dr, current[1] + dc)

                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):     # 检查边界
                    continue
                if self.map[neighbor[0]][neighbor[1]] == 1:     # 检查障碍物
                    continue
                if neighbor in closed_set:      # 检查是否已在关闭列表中
                    continue

                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:    # 如果这是一条更短的路径，或者邻居还没被发现
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, self.goal)
                    f_score[neighbor] = f
                    heapq.heappush(open_list, (f, tentative_g, neighbor))

        print("未找到路径！")
        return None, nodes_visited, 0

    def reconstruct_path(self, came_from, current):
        """
        回溯路径
        :param came_from:
        :param current:
        :return:
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def visualize(self, path=None, title_suffix=""):
        plt.figure(figsize=(6, 6))
        plt.imshow(self.map, cmap='gray_r', origin='upper')

        plt.grid(which='major', axis='both', linestyle='-', color='k', linewidth=0.5)
        plt.xticks(np.arange(-0.5, self.cols, 1), [])
        plt.yticks(np.arange(-0.5, self.rows, 1), [])

        plt.plot(self.start[1], self.start[0], 'go', markersize=15, label='Start')  # 绘制起点（绿色圆圈）
        plt.plot(self.goal[1], self.goal[0], 'ro', markersize=15, label='Goal')     # 绘制终点（红色圆圈）

        if path:
            rows = [p[0] for p in path]
            cols = [p[1] for p in path]
            plt.plot(cols, rows, 'b-', linewidth=3, label='Path')

        plt.title(f"Path Planning: {title_suffix}")
        # plt.legend()
        plt.show()


if __name__ == "__main__":
    # Map 1: 6x6
    map1_data = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0]
    ])

    # Map 2: 15x15
    map2_data = np.array([
        [0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]
    ])

    # Map 3: 20x20
    map3_data = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    # 2. 批量任务定义
    experiments = [
        (map1_data, (5, 0), (0, 5), "Map 1 (6x6)"),
        (map2_data, (14, 0), (0, 14), "Map 2 (15x15)"),
        (map3_data, (19, 0), (0, 19), "Map 3 (20x20)")
    ]

    # 3. 循环执行对比
    for m_data, start_pos, goal_pos, name in experiments:
        print(f"\n>>>>>> 正在测试: {name} <<<<<<")

        # --- 运行 A* ---
        planner_a = AGVPathPlanner(m_data, start_pos, goal_pos, use_heuristic=True)
        t_start = time.time()
        path_a, visited_a, cost_a = planner_a.plan_path()
        time_a = (time.time() - t_start) * 1000

        # --- 运行 Dijkstra ---
        planner_d = AGVPathPlanner(m_data, start_pos, goal_pos, use_heuristic=False)
        t_start = time.time()
        path_d, visited_d, cost_d = planner_d.plan_path()
        time_d = (time.time() - t_start) * 1000

        # --- 输出结果对比 ---
        print(f"  [A*算法]       耗时: {time_a:6.2f} ms | 探索节点: {visited_a:4d} | 路径长度: {cost_a:.2f}")
        print(f"  [Dijkstra算法] 耗时: {time_d:6.2f} ms | 探索节点: {visited_d:4d} | 路径长度: {cost_d:.2f}")

        efficiency_gain = (visited_d - visited_a) / visited_d * 100 if visited_d > 0 else 0
        print(f"  --> A* 效率提升: {efficiency_gain:.1f}% (少搜索了 {visited_d - visited_a} 个无效节点)")

        planner_a.visualize(path_a, title_suffix=f"{name} - A* (Vis: {visited_a})")
        # planner_d.visualize(path_d, title_suffix=f"{name} - Dijkstra (Vis: {visited_d})")
