"""模型入口点，用于构建图并运行 IRMS 求解器。"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Callable, Optional

from .ProgressPrinter import ProgressPrinter
from .Result import Result
from .constants import (
    CNP,
    DCNP,
    DEFAULT_POPULATION_SIZE,
    DEFAULT_TRANSFER_INTERVAL,
    PACKAGE_LOGGER_NAME,
)
from .params import SolverParams

if TYPE_CHECKING:
    from ._pypdms import ProblemData, SolverConfig
    from .stop import StoppingCriterion

logger = logging.getLogger(PACKAGE_LOGGER_NAME)

# DCNP-specific CHNS budget. DCNP's objective rebuilds K-hop trees every step,
# so a single CHNS run is far more expensive than in CNP. The native defaults
# (randomIdleProduct=2000, randomMaxIdleSteps=1000, randomBatchMax=50) make one
# local search take tens of seconds on 300+ node instances, so the population
# never turns over. These lighter caps keep each CHNS run cheap enough that the
# dual population evolves many generations within a few-minute budget. Tuned on
# USAir97 / Circuit / Ecoli (100-500 node instances). Any field left out here
# falls back to the native default.
_DCNP_CHNS_DEFAULTS: dict[str, int | float] = {
    "random_idle_product": 100,
    "random_min_idle_steps": 20,
    "random_max_idle_steps": 80,
    "random_batch_max": 15,
    "theta": 0.3,
}

# DCNP-friendly dual-population defaults. DCNP's per-CHNS cost is high, so a
# large population/exchange interval (the CNP defaults, 6 / 50) means the
# population barely finishes initialization within a time budget and the
# feasible<->infeasible exchange never fires. A smaller population turns over
# faster and a short exchange interval lets the two populations actually mix.
# Applied only when the caller left these at the library defaults.
_DCNP_POPULATION_SIZE = 4
_DCNP_TRANSFER_INTERVAL = 5


def _apply_chns_overrides(
    config: "SolverConfig",
    params: SolverParams,
    dcnp_defaults: Optional[dict[str, "int | float"]] = None,
) -> None:
    """将 CHNS 局部搜索预算写入 ``config.chns``。

    优先级：``params.chns_*``（用户显式指定）> ``dcnp_defaults``（问题相关
    的更优默认）> 原生 C++ 默认（不动 ``config.chns``）。

    Args:
        config: 原生 SolverConfig，其 ``chns`` 子配置会被就地修改
        params: 求解器参数，读取其 ``chns_*`` 覆盖字段
        dcnp_defaults: 问题相关的 CHNS 默认（键为 chns 字段名，不含前缀）
    """
    # chns 字段名 -> params 上对应的覆盖属性名
    field_to_param = {
        "max_idle_steps": "chns_max_idle_steps",
        "theta": "chns_theta",
        "random_batch_max": "chns_random_batch_max",
        "random_idle_product": "chns_random_idle_product",
        "random_min_idle_steps": "chns_random_min_idle_steps",
        "random_max_idle_steps": "chns_random_max_idle_steps",
    }
    defaults = dcnp_defaults or {}
    for chns_field, param_attr in field_to_param.items():
        override = getattr(params, param_attr, None)
        if override is not None:
            setattr(config.chns, chns_field, override)
        elif chns_field in defaults:
            setattr(config.chns, chns_field, defaults[chns_field])


def _normalize_feasible_population(
    feasible_population: list[tuple[set[int], int]],
) -> list[tuple[set[int], int]]:
    """规范化可行解种群，确保解和目标值的数据类型一致。

    Args:
        feasible_population: 可行解种群，每个元素为 (解集合, 目标值) 的元组

    Returns:
        规范化后的种群，按目标值、解大小和解内容排序
    """
    # 将解转换为 set 类型，目标值转换为 int 类型，确保数据类型一致性
    normalized = [
        (set(solution), int(obj_value))
        for solution, obj_value in feasible_population
    ]
    # 按目标值、解大小和解内容排序，确保结果可重复
    normalized.sort(key=lambda item: (item[1], len(item[0]), tuple(sorted(item[0]))))
    return normalized


def _compute_overlap_ratio_matrix(
    feasible_population: list[tuple[set[int], int]],
) -> list[list[float]]:
    """计算可行解种群中各解之间的重叠率矩阵。

    重叠率定义为两个解的交集大小除以第一个解的大小。

    Args:
        feasible_population: 可行解种群，每个元素为 (解集合, 目标值) 的元组

    Returns:
        重叠率矩阵，matrix[i][j] 表示解 i 和解 j 的重叠率
    """
    overlap_ratios: list[list[float]] = []
    for solution, _ in feasible_population:
        denominator = len(solution)
        row: list[float] = []
        for other_solution, _ in feasible_population:
            if denominator == 0:
                # 如果解为空，重叠率设为 0
                row.append(0.0)
            else:
                # 计算交集大小占解大小的比例
                row.append(len(solution & other_solution) / denominator)
        overlap_ratios.append(row)
    return overlap_ratios


class Model:
    """图问题建模和求解类，用于关键节点问题 (Critical Node Problems)。"""

    def __init__(self) -> None:
        """初始化模型实例。"""
        # 节点集合
        self.nodes: set[int] = set()
        # 邻接表表示的图结构
        self.adj_list: list[set[int]] = []
        # 缓存的问题数据对象
        self._problem_data: Optional[ProblemData] = None

    def add_node(self, node: int) -> None:
        """向图中添加节点。

        Args:
            node: 节点 ID，必须为非负整数

        Raises:
            ValueError: 如果节点 ID 不是非负整数
        """
        if not isinstance(node, int) or node < 0:
            raise ValueError("节点 ID 必须是非负整数。")
        self.nodes.add(node)
        # 扩展邻接表以容纳新节点
        while len(self.adj_list) <= node:
            self.adj_list.append(set())
        # 添加节点后需要重新创建问题数据
        self._problem_data = None

    def add_edge(self, u: int, v: int) -> None:
        """向图中添加边。

        如果边的端点不存在，会自动创建节点。

        Args:
            u: 边的第一个端点
            v: 边的第二个端点
        """
        # 确保节点存在
        if u not in self.nodes:
            self.add_node(u)
        if v not in self.nodes:
            self.add_node(v)
        # 扩展邻接表
        while len(self.adj_list) <= max(u, v):
            self.adj_list.append(set())
        # 添加无向边（两个方向都要添加）
        self.adj_list[u].add(v)
        self.adj_list[v].add(u)
        # 添加边后需要重新创建问题数据
        self._problem_data = None

    @staticmethod
    def from_data(problem_data: "ProblemData") -> "Model":
        """从 ProblemData 对象创建 Model 实例。

        Args:
            problem_data: 问题数据对象

        Returns:
            创建的 Model 实例
        """
        model = Model()
        model.nodes = set(problem_data.get_nodes_set())
        model.adj_list = [set(neighbors) for neighbors in problem_data.get_adj_list()]
        model._problem_data = problem_data
        return model

    @property
    def problem_data(self) -> "ProblemData":
        """获取问题数据对象，如果不存在则创建。"""
        if self._problem_data is None:
            self._problem_data = self._create_problem_data()
        return self._problem_data

    def _create_problem_data(self) -> "ProblemData":
        """创建 ProblemData 对象。

        Returns:
            创建的 ProblemData 对象
        """
        from ._pypdms import ProblemData

        # 确定最大节点 ID
        max_node_id = max(self.nodes) if self.nodes else 0
        # 创建问题数据对象，大小为最大节点 ID + 1
        problem_data = ProblemData(max_node_id + 1)
        # 添加所有节点
        for node in self.nodes:
            problem_data.add_node(node)
        # 添加所有边（只添加一次，u < v 避免重复）
        for u in range(len(self.adj_list)):
            for v in self.adj_list[u]:
                if u < v:
                    problem_data.add_edge(u, v)
        return problem_data

    def solve(
        self,
        budget: Optional[int] = None,
        stopping_criterion: Optional["StoppingCriterion"] = None,
        seed: int = 0,
        params: Optional[SolverParams] = None,
        population_size: Optional[int] = None,
        offspring_count: Optional[int] = None,
        search: Optional[str] = None,
        display_interval: Optional[float] = None,
        display: bool = True,
        collect_stats: bool = True,
        problem: str = CNP,
        distance: Optional[int] = None,
    ) -> Result:
        """求解图上的关键节点问题。

        通过 ``problem`` 参数支持两种问题变体：

        - ``"CNP"`` (默认): 预算约束 CNP。给定 ``budget`` ``k``,
          最小化剩余成对连通性 ``Σ |C|(|C|-1)/2``。
        - ``"DCNP"``: 距离关键节点问题。给定 ``budget`` ``k`` 和距离
          ``distance`` ``D``，最小化剩余图中距离不超过 ``D`` 的无序节点对数。

        Parameters
        ----------
        budget
            要移除的节点数（必需，``1 <= budget < |V|``）。
        stopping_criterion
            返回 ``True`` 时求解器停止的可调用对象。
        seed
            随机数生成器种子。``0`` 是有效种子。
        params
            可调求解器参数。默认为 :class:`SolverParams()`。
        population_size, offspring_count, search
            对应 ``params`` 字段的便捷覆盖。
        display_interval
            迭代日志行之间的最小秒数。默认为 ``params.display_interval``。
        display
            如果为 ``True``，将进度记录到包日志器。
        collect_stats
            如果为 ``True``，在 :attr:`Result.stats` 中记录每次迭代的追踪。
        problem
            ``"CNP"`` (默认) 或 ``"DCNP"``。
        distance
            DCNP 距离阈值 ``D``（DCNP 必需，``D >= 1``）。
        """
        if stopping_criterion is None:
            raise ValueError("stopping_criterion 是必需的")

        # 解析参数，应用直接关键字覆盖
        params = self._resolve_params(
            params, population_size, offspring_count, search
        )
        effective_display_interval = (
            display_interval if display_interval is not None
            else params.display_interval
        )

        # 规范化问题名称
        normalized_problem = str(problem).upper().replace("_", "-")
        if normalized_problem == DCNP:
            if not isinstance(budget, int) or budget < 1:
                raise ValueError("budget 必须是正整数")
            if budget >= len(self.nodes):
                raise ValueError(
                    f"预算 ({budget}) 必须小于节点数 ({len(self.nodes)})"
                )
            if not isinstance(distance, int) or distance < 1:
                raise ValueError("distance 必须是 DCNP 的正整数")
            return self._solve_dcnp(
                budget, distance, stopping_criterion, seed, params,
                effective_display_interval, display, collect_stats,
            )

        if normalized_problem != CNP:
            raise ValueError(f"未知问题 {problem!r}; 期望 'CNP' 或 'DCNP'")

        # 验证预算参数
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget 必须是正整数")
        if budget >= len(self.nodes):
            raise ValueError(
                f"预算 ({budget}) 必须小于节点数 ({len(self.nodes)})"
            )

        max_runtime = getattr(stopping_criterion, "max_runtime", None)
        return self._solve_fixed_budget(
            budget, stopping_criterion, seed, params,
            effective_display_interval, display, collect_stats,
            max_runtime if isinstance(max_runtime, (int, float)) else None,
        )

    @staticmethod
    def _resolve_params(
        params: Optional[SolverParams],
        population_size: Optional[int],
        offspring_count: Optional[int],
        search: Optional[str],
    ) -> SolverParams:
        """在 ``SolverParams`` 之上应用直接关键字覆盖。

        使用 :func:`dataclasses.replace` 以便覆盖值被重新验证，
        且调用者的 ``params`` 实例永远不会被修改。

        Args:
            params: 求解器参数对象
            population_size: 种群大小覆盖
            offspring_count: 后代数量覆盖
            search: 搜索策略覆盖

        Returns:
            解析后的求解器参数
        """
        from dataclasses import replace

        if params is None:
            params = SolverParams()

        # 构建覆盖字典，只包含非 None 的值
        overrides: dict[str, Any] = {
            key: value
            for key, value in (
                ("population_size", population_size),
                ("offspring_count", offspring_count),
                ("search", search),
            )
            if value is not None
        }
        return replace(params, **overrides) if overrides else params
    # 求解CNP 问题，调用_run_solver
    def _solve_fixed_budget(
        self,
        budget: int,
        stopping_criterion: Callable[[float], bool],
        seed: int,
        params: SolverParams,
        effective_display_interval: float,
        display: bool,
        collect_stats: bool,
        max_runtime: Optional[float],
    ) -> Result:
        """在固定预算下运行一次 IRMS 搜索，最小化成对连通性 (CNP1)。

        ``max_runtime`` (秒) 为正时，作为硬墙钟截止时间推送到原生求解器，
        以便它可以在种群初始化和单代内停止。

        Args:
            budget: 移除节点的预算
            stopping_criterion: 停止准则
            seed: 随机种子
            params: 求解器参数
            effective_display_interval: 有效显示间隔
            display: 是否显示进度
            collect_stats: 是否收集统计信息
            max_runtime: 最大运行时间

        Returns:
            求解结果
        """
        from ._pypdms import SolverConfig

        # 创建原始图
        original_graph = self.problem_data.create_original_graph(
            budget, seed
        )

        # 配置求解器
        config = SolverConfig()
        config.population_size = params.population_size
        config.offspring_count = params.offspring_count
        config.transfer_interval = params.transfer_interval
        config.seed = seed
        config.partial_ratio = params.partial_ratio
        config.beta = params.beta
        config.display_interval = effective_display_interval
        config.search = params.search
        if isinstance(max_runtime, (int, float)) and max_runtime > 0:
            config.max_runtime = float(max_runtime)

        return self._run_solver(
            original_graph, budget, config,
            stopping_criterion, display, effective_display_interval,
            collect_stats,
        )

    def _solve_dcnp(
        self,
        budget: int,
        distance: int,
        stopping_criterion: Callable[[float], bool],
        seed: int,
        params: SolverParams,
        effective_display_interval: float,
        display: bool,
        collect_stats: bool,
    ) -> Result:
        """求解固定预算 DCNP，采用与 CNP 相同的双种群 IRMS 流程。

        目标值是删除 ``budget`` 个节点后，剩余图中距离不超过
        ``distance`` 的无序节点对数量；值越小越好。

        求解过程与 CNP 完全一致：维护可行（预算 ``k``）与不可行
        （部分预算 ``⌊k * partial_ratio⌋``）两个种群，每代通过
        RSC 交叉加 CHNS 局部搜索产生后代，并按成本 + 多样性排名
        裁剪种群；每隔 ``transfer_interval`` 代把不可行种群的最优解
        补全到完整预算后注入可行种群。
        """
        from ._pypdms import SolverConfig

        # 创建原始 DCNP 图
        original_graph = self.problem_data.create_original_dcnp_graph(
            budget, distance, seed
        )

        # DCNP 单次 CHNS 昂贵，大种群/长交换间隔（CNP 默认 6/50）会让种群
        # 在时间预算内几乎跑不完初始化、交换也永不触发。若调用者未显式改动
        # 这两个值（仍为库默认），替换为更适合 DCNP 的小种群 + 短交换间隔。
        dcnp_population_size = (
            _DCNP_POPULATION_SIZE
            if params.population_size == DEFAULT_POPULATION_SIZE
            else params.population_size
        )
        dcnp_transfer_interval = (
            _DCNP_TRANSFER_INTERVAL
            if params.transfer_interval == DEFAULT_TRANSFER_INTERVAL
            else params.transfer_interval
        )

        # 配置求解器（与 CNP 路径一致）
        config = SolverConfig()
        config.population_size = dcnp_population_size
        config.offspring_count = params.offspring_count
        config.transfer_interval = dcnp_transfer_interval
        config.seed = seed
        config.partial_ratio = params.partial_ratio
        config.beta = params.beta
        config.display_interval = effective_display_interval
        config.search = params.search

        # DCNP 的目标函数每步重建 K-hop 树，单次 CHNS 比 CNP 贵得多。
        # 默认的 randomize idle 预算（可到 1000+ 步）在中等实例上会让
        # 单次局部搜索耗时数十秒、种群跑不动。这里为 DCNP 设更轻量的
        # CHNS 默认预算；params.chns_* 若显式给出则覆盖。
        _apply_chns_overrides(config, params, dcnp_defaults=_DCNP_CHNS_DEFAULTS)

        max_runtime = getattr(stopping_criterion, "max_runtime", None)
        if isinstance(max_runtime, (int, float)) and max_runtime > 0:
            config.max_runtime = float(max_runtime)

        return self._run_solver(
            original_graph, budget, config,
            stopping_criterion, display, effective_display_interval,
            collect_stats, dcnp=True,
        )

    def _run_solver(
        self,
        original_graph,
        budget: int,
        config,
        stopping_criterion: Callable[[float], bool],
        display: bool,
        display_interval: float,
        collect_stats: bool,
        dcnp: bool = False,
    ) -> Result:
        """运行双种群求解器（CNP 与 DCNP 共用）。

        Args:
            original_graph: 原始图对象（CNP_Graph 或 DCNP_Graph）
            budget: 预算
            config: 求解器配置
            stopping_criterion: 停止准则
            display: 是否显示进度
            display_interval: 显示间隔
            collect_stats: 是否收集统计信息
            dcnp: 是否为 DCNP 问题（选择对应的双种群实现）

        Returns:
            求解结果
        """
        from ._pypdms import DCNPDualPopulation, DualPopulation

        population_cls = DCNPDualPopulation if dcnp else DualPopulation

        start_time = time.perf_counter()

        # 将基于时间的准则时钟与实际求解器启动对齐，
        # 以便报告的运行时间和原生截止时间共享一个参考。
        # 否则准则的时钟会在构造时启动（在图设置之前），
        # 缩小有效预算。
        if hasattr(stopping_criterion, "start_time"):
            stopping_criterion.start_time = start_time

        # 不可行解移除的节点少于预算允许的，
        # 上限为 budget-1 以便它们保持严格不可行（且 >=1 以便种群有东西演化）。
        # 当 budget=1 时我们回退到 1，这等于可行预算。
        infeasible_budget = max(1, min(
            math.floor(budget * config.partial_ratio),
            budget - 1,
        ))

        population = population_cls(
            original_graph, budget, infeasible_budget, config,
        )

        printer = ProgressPrinter(
            should_print=display,
            logger=logger,
            display_interval=display_interval,
        )
        printer.start(budget, config.seed)
        printer.initializing_population_message()

        init_result = population.initialize()
        best_solution = set(init_result[0])
        best_obj_value = init_result[1]
        best_found_at_time = time.perf_counter() - start_time

        printer.print_iterations_header()

        iterations = 0
        idle_generations = 0
        stats: list[dict] = []

        # 主求解循环
        while not stopping_criterion(best_obj_value):
            population.advance_one_generation()
            iterations += 1

            # 处理交换事件
            for event in population.drain_exchange_events():
                report = event.report
                printer.exchange(event.iteration, {
                    "exchange_triggered": report.exchange_triggered,
                    "first_population_candidate_obj":
                        report.first_population_candidate_obj,
                    "first_population_improved_best":
                        report.first_population_improved_best,
                })

            # 处理迭代事件
            for iter_event in population.drain_iteration_events():
                elapsed = time.perf_counter() - start_time
                if iter_event.best_objective < best_obj_value:
                    best_obj_value = iter_event.best_objective
                    best_found_at_time = elapsed
                    idle_generations = 0
                else:
                    idle_generations += 1

                if collect_stats:
                    stats.append({
                        "iteration": iter_event.iteration,
                        "elapsed": elapsed,
                        "best_obj_value": best_obj_value,
                        "idle_generations": idle_generations,
                        "population_size": iter_event.population_size,
                    })

                printer.iteration(
                    iter_event.iteration, elapsed, best_obj_value,
                    idle_generations, iter_event.population_size,
                )

        # 获取最终解
        final_sol, final_obj = population.get_best_feasible_solution()
        best_solution = set(final_sol)
        best_obj_value = min(best_obj_value, final_obj)
        runtime = time.perf_counter() - start_time

        # 规范化可行解种群
        raw_pop = population.get_feasible_population()
        feasible_population = _normalize_feasible_population(
            [(set(s), v) for s, v in raw_pop]
        )
        overlap_ratios = _compute_overlap_ratio_matrix(feasible_population)

        result = Result(
            best_solution=best_solution,
            best_obj_value=best_obj_value,
            num_iterations=iterations,
            runtime=runtime,
            best_found_at_time=best_found_at_time,
            stats=stats if collect_stats else None,
            feasible_population=feasible_population,
            feasible_population_overlap_ratios=overlap_ratios,
        )

        printer.end(result)
        return result
