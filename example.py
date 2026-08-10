"""
PyPDMS 求解关键节点问题的用法示例。

本示例刻意把*每一个可配置选项*都显式写出,让你一眼看清运行 IRMS
(Iterative Ruin and Memetic Search,迭代破坏与模因搜索)算法时有哪些可调项。
除非注释另有说明,下面所有取值都是默认值。
"""

import time

import pypdms
from pypdms import (
    MaxIterations,  # 达到 N 代后停止
    MaxRuntime,  # 达到 S 秒墙钟时间后停止(精确)
    Model,
    NoImprovement,  # 连续 N 代无改善后停止
    SolverParams,
)


def build_graph_manually() -> Model:
    """逐条边构建一张小图。"""
    model = Model()
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
        (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
    ]
    for u, v in edges:
        model.add_edge(u, v)
    return model


def main():
    # ------------------------------------------------------------------ #
    # 1. 构建模型:手动构建(见上)或从文件读取。
    # ------------------------------------------------------------------ #
    # model = build_graph_manually()
    model = Model.from_data(
        pypdms.read("Instances/CNP/realworld/condmat.txt")
    )

    # ------------------------------------------------------------------ #
    # 2. 选择一个停止准则(只选其一)。
    #    三种都列出;实际只使用下面的 `stopping_criterion`。
    # ------------------------------------------------------------------ #
    stop_by_iterations = MaxIterations(500)      # 500 代后停止
    stop_by_runtime = MaxRuntime(3600)             # 3600 秒后停止
    stop_by_no_improve = NoImprovement(200)      # 连续 200 代无改善 -> 停止
    # 你也可以传入任何自定义的 Callable[[float], bool]。
    _ = (stop_by_iterations, stop_by_no_improve)  # 消除“未使用”的 lint 警告
    stopping_criterion = stop_by_runtime

    # ------------------------------------------------------------------ #
    # 3. 可调求解器参数。每个字段都列出其默认值。
    #    (约束:population_size>=2、offspring_count>=1、
    #     transfer_interval>=1、0<partial_ratio<1、0<=beta<=1、
    #     display_interval>0。)
    # ------------------------------------------------------------------ #
    params = SolverParams(
        population_size=6,        # 每个(可行/不可行)种群的规模
        offspring_count=1,        # 每代每个种群生成的后代数
        transfer_interval=50,     # 可行<->不可行交换之间的代数间隔
        partial_ratio=0.95,       # 不可行预算 = floor(budget*partial_ratio)
        beta=0.9,                 # RSC 交叉:保留共享节点的概率
        display_interval=1.0,     # 进度日志行之间的最小秒数
        search="CHNS",            # 局部搜索策略(见下方选项)
        # CHNS 预算覆盖;None -> 原生默认(针对 CNP 调优)。这些主要对 DCNP
        # 有意义(见下方 dcnp()),那里每次 CHNS 都很昂贵;对 CNP 保持 None。
        chns_random_idle_product=None,
        chns_random_min_idle_steps=None,
        chns_random_max_idle_steps=None,
        chns_random_batch_max=None,
        chns_theta=None,
    )
    # `search` 接受:
    #   "CHNS"        -> 随机化批规模与空闲步(默认,自适应)
    #   "CHNS-ADAPT"  -> 基于空闲步的自适应批增长
    #   "CHNS<N>"     -> 固定批规模 N,例如 "CHNS5"

    # ------------------------------------------------------------------ #
    # 4. 运行求解器。solve() 的每个可选参数都列出。
    #    在此传入的 population_size / offspring_count / search 会**覆盖**
    #    `params` 中对应字段(便于快速实验);保持 None 则直接使用 `params`。
    # ------------------------------------------------------------------ #
    result = model.solve(
        budget=2313,                            # 必填:要删除的节点数
        stopping_criterion=stopping_criterion, # 必填:何时停止
        seed=42,                               # 随机种子(0 也有效)
        params=params,                         # 上面的 SolverParams
        population_size=None,                  # 覆盖 params.population_size
        offspring_count=None,                  # 覆盖 params.offspring_count
        search=None,                           # 覆盖 params.search
        display_interval=None,                 # 覆盖 params.display_interval
        display=True,                          # 将进度记录到 logger
        collect_stats=True,                    # 记录逐代轨迹
    )

    # ------------------------------------------------------------------ #
    # 5. 查看结果。Result 的每个字段都列出。
    # ------------------------------------------------------------------ #
    print(f"Best objective value : {result.best_obj_value}")
    # print(f"Removed nodes        : {sorted(result.best_solution)}")
    print(f"Iterations           : {result.num_iterations}")
    print(f"Runtime              : {result.runtime:.2f}s")
    print(f"Best found at        : {result.best_found_at_time:.2f}s")
    print(f"Final population size: {len(result.feasible_population)}")
    if result.stats:
        print(f"Trace entries        : {len(result.stats)}")


def dcnp():
    """在小实例 `karate` 上求解基于距离的 CNP(DCNP)。

    DCNP 固定删除预算 k 和距离阈值 D,最小化剩余图中最短路距离不超过 D 的
    无序节点对数量。
    """
    # ------------------------------------------------------------------ #
    # 1. 从 DIMACS 边表格式的 DCNP 实例构建模型。
    #    pypdms.read() 会自动识别 `p edge n m` / `e u v` 格式。
    # ------------------------------------------------------------------ #
    budget = 30
    distance = 3
    seed = 1

    model = Model.from_data(
        pypdms.read("Instances/DCNP/S1/gnm3(9).txt")
    )

    # ------------------------------------------------------------------ #
    # 2. 选择一个停止准则(只选其一)。
    # ------------------------------------------------------------------ #
    # stop_by_iterations = MaxIterations(20)
    stop_by_runtime = MaxRuntime(3600)
    # stop_by_no_improve = NoImprovement(10)
    # _ = (stop_by_runtime, stop_by_no_improve)
    stopping_criterion = stop_by_runtime

    # ------------------------------------------------------------------ #
    # 3. 可调求解器参数。它们与 CNP 使用的是同一组公共参数;对 DCNP 而言,
    #    它们控制种群、交叉以及 CHNS 局部搜索的行为。
    #
    #    下面每个字段都设为其 DCNP 默认值,因此这个 SolverParams 等价于完全
    #    不传 `params`。DCNP 覆盖了三项 CNP 默认(population_size 6->4、
    #    transfer_interval 50->5,以及更轻量的 CHNS 空闲预算),因为 DCNP 的
    #    目标函数每步都重建 K-hop 树,使得每次局部搜索昂贵得多;更轻的预算让
    #    种群得以在时间内真正完成迭代更替。population_size / transfer_interval
    #    的替换仅在你把它们保留为 CNP 库默认(6 / 50)时生效,所以这里把有效的
    #    DCNP 取值显式写出。
    # ------------------------------------------------------------------ #
    params = SolverParams(
        population_size=4,             # DCNP 默认(CNP 默认为 6)
        offspring_count=1,             # 每代生成的后代数
        transfer_interval=5,           # DCNP 默认(CNP 默认为 50)
        partial_ratio=0.95,            # 不可行预算 = floor(budget*ratio)
        beta=0.9,                      # 交叉时保留共享节点的概率
        display_interval=1.0,          # 进度日志行之间的最小秒数
        search="CHNS",                 # 局部搜索策略
        # CHNS 预算覆盖(None -> 原生默认)。这些取值就是自动应用的 DCNP
        # 默认值;在此显式写出以便调节。
        chns_random_idle_product=100,  # 单次空闲步预算的缩放因子
        chns_random_min_idle_steps=20, # 空闲步下限(大 D 时起约束作用)
        chns_random_max_idle_steps=80,  # 空闲步上限
        chns_random_batch_max=15,      # 每步最大破坏批规模
        chns_theta=0.3,                # 贪心(相对随机)选点的概率
    )

    # ------------------------------------------------------------------ #
    # 4. 运行 DCNP 求解器。`budget` 和 `distance` 对 DCNP 是必填的。
    #    在此传入的 population_size / offspring_count / search 会**覆盖**
    #    `params` 中对应字段;保持 None 则使用 `params`。
    # ------------------------------------------------------------------ #
    start = time.perf_counter()
    result = model.solve(
        problem="DCNP",
        budget=budget,                         # 必填:要删除的节点数
        distance=distance,                     # 必填:D 跳阈值
        stopping_criterion=stopping_criterion, # 必填:何时停止
        seed=seed,                             # 随机种子(0 也有效)
        params=params,                         # 上面的 SolverParams
        population_size=None,                  # 覆盖 params.population_size
        offspring_count=None,                  # 覆盖 params.offspring_count
        search=None,                           # 覆盖 params.search
        display_interval=None,                 # 覆盖 params.display_interval
        display=True,                          # 将进度记录到 logger
        collect_stats=True,                    # 记录逐代轨迹
    )
    wall = time.perf_counter() - start

    # ------------------------------------------------------------------ #
    # 5. 查看结果。
    # ------------------------------------------------------------------ #
    print("\n=== DCNP ===")
    print(f"Nodes / budget k / D : {len(model.nodes)} / {budget} / {distance}")
    print(f"Best D-hop pairs     : {result.best_obj_value}")
    print(f"Removed nodes        : {sorted(result.best_solution)}")
    print(f"Iterations           : {result.num_iterations}")
    print(f"Runtime              : {result.runtime:.2f}s")
    print(f"Best found at        : {result.best_found_at_time:.2f}s")
    print(f"Final population size: {len(result.feasible_population)}")
    if result.stats:
        print(f"Trace entries        : {len(result.stats)}")
    print(f"Wall time            : {wall:.2f}s")


if __name__ == "__main__":
    main()
    # dcnp()
