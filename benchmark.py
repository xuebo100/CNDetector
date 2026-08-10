"""
PyPDMS 重复运行基准测试。

在同一实例上用不同种子将 IRMS 求解器运行 ``n`` 次,并汇总报告:

  * 找到的最优目标值,
  * 平均目标值,
  * 每次运行找到其最优解的平均时刻(``Result.best_found_at_time``),以及
  * 平均总运行时间。

所有旋钮通过命令行传入。示例
-------------------------------------------------
    # 10 次,每次 30 秒,Hamilton3000a,预算 300
    python benchmark.py Instances/CNP/realworld/Hamilton3000a.txt \
        --budget 300 --runs 10 --max-runtime 30

    # 以迭代次数为界的 5 次运行,自定义求解器旋钮
    python benchmark.py Instances/CNP/realworld/Bovine.txt \
        --budget 3 --runs 5 --max-iterations 200 \
        --population-size 12 --offspring-count 2 --search CHNS5

    # DCNP:删除 k=20 个节点后最小化距离 D=2 之内的节点对数,
    # 5 次,每次 120 秒(自动应用 DCNP 调优默认)
    python benchmark.py Instances/DCNP/R1/USAir97.txt \
        --problem DCNP --budget 20 --distance 2 --runs 5 --max-runtime 120

    # DCNP:为大 D 实例使用更轻量的 CHNS 预算
    python benchmark.py Instances/DCNP/R1/USAir97.txt \
        --problem DCNP --budget 30 --distance 3 --runs 3 --max-runtime 120 \
        --population-size 3 --chns-random-idle-product 60 \
        --chns-random-min-idle-steps 10
"""

from __future__ import annotations

import argparse
import statistics
from typing import Callable

import pypdms
from pypdms import (
    MaxIterations,
    MaxRuntime,
    Model,
    NoImprovement,
    SolverParams,
)
from pypdms.stop import StoppingCriterion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 PyPDMS 求解器运行 n 次并报告汇总统计。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "instance",
        help="邻接表格式的图文件路径。",
    )
    parser.add_argument(
        "--problem", type=str, default="CNP", choices=["CNP", "DCNP"],
        help="要求解的问题变体。",
    )
    parser.add_argument(
        "--budget", type=int, default=None,
        help="要删除的节点数 (k);必填,须 < |V|。",
    )
    parser.add_argument(
        "--distance", type=int, default=None,
        help="DCNP 距离阈值 D(--problem DCNP 必填,D >= 1)。",
    )
    parser.add_argument(
        "-n", "--runs", type=int, default=10,
        help="连续运行的次数。",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="基准随机种子;第 i 次运行使用 seed + i。",
    )

    # 停止准则(互斥)。未指定时默认为 30 秒运行时间。
    stop = parser.add_mutually_exclusive_group()
    stop.add_argument(
        "--max-runtime", type=float, metavar="SECONDS",
        help="每次运行达到该秒数(墙钟)后停止。",
    )
    stop.add_argument(
        "--max-iterations", type=int, metavar="N",
        help="每次运行达到该代数后停止。",
    )
    stop.add_argument(
        "--no-improvement", type=int, metavar="N",
        help="每次运行在连续 N 代无改善后停止。",
    )

    # 可选求解器旋钮(留空为 None -> 使用 SolverParams 默认值)。
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--offspring-count", type=int, default=None)
    parser.add_argument(
        "--search", type=str, default=None,
        help='局部搜索策略:"CHNS"、"CHNS-ADAPT" 或 "CHNS<N>"。',
    )
    parser.add_argument(
        "--transfer-interval", type=int, default=None,
        help="可行<->不可行交换之间的代数间隔。",
    )
    parser.add_argument("--partial-ratio", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)

    # CHNS 局部搜索预算覆盖(None -> 使用问题相关默认)。这些是 DCNP 的主要
    # 调参旋钮——DCNP 的目标函数每步都重建 K-hop 树,DCNP 已自动应用更轻量默认。
    parser.add_argument("--chns-theta", type=float, default=None)
    parser.add_argument("--chns-random-batch-max", type=int, default=None)
    parser.add_argument("--chns-random-idle-product", type=int, default=None)
    parser.add_argument("--chns-random-min-idle-steps", type=int, default=None)
    parser.add_argument("--chns-random-max-idle-steps", type=int, default=None)

    parser.add_argument(
        "--show-solver-log", action="store_true",
        help="显示每次运行的求解器进度日志(默认关闭)。",
    )
    return parser.parse_args()


def make_criterion_factory(args: argparse.Namespace) -> Callable[[], StoppingCriterion]:
    """返回一个工厂,为每次运行生成*全新的*停止准则。

    每次运行用全新实例很重要:MaxRuntime 会重置其时钟,而
    MaxIterations / NoImprovement 带有可变计数器,绝不能在多次运行间泄漏。
    """
    if args.max_iterations is not None:
        return lambda: MaxIterations(args.max_iterations)
    if args.no_improvement is not None:
        return lambda: NoImprovement(args.no_improvement)
    # 默认:基于运行时间,未指定时为 30 秒。
    seconds = args.max_runtime if args.max_runtime is not None else 30.0
    return lambda: MaxRuntime(seconds)


def build_params(args: argparse.Namespace) -> SolverParams:
    """构建 SolverParams,只覆盖用户显式提供的字段。

    未设置的字段保持 ``None``,以便应用求解器的问题相关默认值
    (尤其是 DCNP 更轻量的种群与 CHNS 预算)。
    """
    overrides = {
        key: value
        for key, value in (
            ("population_size", args.population_size),
            ("offspring_count", args.offspring_count),
            ("search", args.search),
            ("transfer_interval", args.transfer_interval),
            ("partial_ratio", args.partial_ratio),
            ("beta", args.beta),
            ("chns_theta", args.chns_theta),
            ("chns_random_batch_max", args.chns_random_batch_max),
            ("chns_random_idle_product", args.chns_random_idle_product),
            ("chns_random_min_idle_steps", args.chns_random_min_idle_steps),
            ("chns_random_max_idle_steps", args.chns_random_max_idle_steps),
        )
        if value is not None
    }
    return SolverParams(**overrides)


def main() -> None:
    args = parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    # 各问题变体的参数要求。
    if args.problem == "DCNP" and args.distance is None:
        raise SystemExit("--distance is required for --problem DCNP")
    if args.problem != "DCNP" and args.distance is not None:
        raise SystemExit("--distance only applies to --problem DCNP")
    if args.budget is None:
        raise SystemExit(f"--budget is required for --problem {args.problem}")

    model = Model.from_data(pypdms.read(args.instance))
    params = build_params(args)
    make_criterion = make_criterion_factory(args)

    # 按问题变体组装 solve() 的关键字参数。distance 仅 DCNP 用。
    solve_kwargs: dict = {"problem": args.problem, "budget": args.budget}
    if args.problem == "DCNP":
        solve_kwargs["distance"] = args.distance

    objectives: list[float] = []
    best_found_times: list[float] = []
    runtimes: list[float] = []

    # 对 DCNP,若 population_size / transfer_interval 保留为 CNP 库默认值,求解器
    # 内部会替换为 DCNP 调优值;因此显示 "auto",而非用户并未选择的、会造成误导的
    # 库默认值。
    if args.problem == "DCNP" and args.population_size is None:
        pop_display = "auto(DCNP)"
    else:
        pop_display = str(params.population_size)

    # 主约束:预算 k(CNP/DCNP)。
    constraint_line = f"Budget   : {args.budget}"
    objective_label = "objective"

    print(f"Instance : {args.instance}")
    print(f"Problem  : {args.problem}"
          + (f"  distance D={args.distance}" if args.problem == "DCNP" else ""))
    print(constraint_line)
    print(f"Runs     : {args.runs}")
    print(f"Search   : {params.search}  pop={pop_display}  "
          f"offspring={params.offspring_count}")
    print("-" * 64)
    print(f"{'run':>4} {'seed':>6} {objective_label:>14} "
          f"{'best@(s)':>10} {'runtime(s)':>11}")
    print("-" * 64)

    for i in range(args.runs):
        seed = args.seed + i
        result = model.solve(
            stopping_criterion=make_criterion(),
            seed=seed,
            params=params,
            display=args.show_solver_log,
            collect_stats=False,
            **solve_kwargs,
        )
        objectives.append(result.best_obj_value)
        best_found_times.append(result.best_found_at_time)
        runtimes.append(result.runtime)
        print(f"{i:>4} {seed:>6} {result.best_obj_value:>14} "
              f"{result.best_found_at_time:>10.2f} {result.runtime:>11.2f}")

    print("-" * 64)
    print("Summary over", args.runs, "run(s):")
    print(f"  Best objective        : {min(objectives)}")
    print(f"  Worst objective       : {max(objectives)}")
    print(f"  Average objective     : {statistics.mean(objectives):.2f}")
    if args.runs > 1:
        print(f"  Std dev objective     : {statistics.stdev(objectives):.2f}")
    print(f"  Avg time to best (s)  : {statistics.mean(best_found_times):.2f}")
    print(f"  Avg total runtime (s) : {statistics.mean(runtimes):.2f}")


if __name__ == "__main__":
    main()
