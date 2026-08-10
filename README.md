# PyPDMS

一个高性能的**关键节点问题(Critical Node Problem, CNP)** Python 求解器,采用
**IRMS**(*Iterative Ruin and Memetic Search*,迭代破坏与模因搜索)——一种基于
种群的双模因元启发式算法。PyPDMS 将高速的 C++ 内核(通过 pybind11)与简洁的
Python API 结合在一起。

关键节点问题要解决的是:给定一张图和预算 `k`,应删除哪 `k` 个顶点,才能最小化
剩余的成对连通性(即所有剩余连通分量上 `|C|*(|C|-1)/2` 之和)?

PyPDMS 通过统一的入口 `Model.solve` 求解两种变体:

- **CNP**(`problem="CNP"`,默认):给定预算 `k`,最小化剩余成对连通性。
- **DCNP**(`problem="DCNP"`):*基于距离* CNP —— 给定预算 `k` 和距离 `D`,
  最小化剩余图中最短路距离不超过 `D` 的无序节点对数。

## 安装

PyPDMS 使用 Meson + Ninja + pybind11 从源码构建。

```bash
pip install -e . --no-build-isolation
```

修改 C++ 代码后,如需单独重编译原生扩展:

```bash
python buildtools/build_extensions.py --build_type release
```

依赖要求:
- Python ≥ 3.9
- 支持 C++20 的编译器(clang ≥ 17、gcc ≥ 11 或 MSVC 2022)
- `meson`、`ninja`、`pybind11`(由 `build` 依赖组自动引入)

## 快速上手

```python
from pypdms import Model, MaxIterations

model = Model()
edges = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (5, 6), (6, 7), (7, 8), (8, 9), (9, 0),
    (0, 5), (1, 6), (2, 7), (3, 8), (4, 9),
]
for u, v in edges:
    model.add_edge(u, v)

result = model.solve(
    budget=3,
    stopping_criterion=MaxIterations(50),
    seed=42,
)

print(f"Best objective: {result.best_obj_value}")
print(f"Removed nodes:  {sorted(result.best_solution)}")
print(f"Runtime:        {result.runtime:.2f}s")
```

### 从文件读取图

```python
import pypdms
from pypdms import Model, MaxRuntime

problem = pypdms.read("path/to/graph.adj")  # 邻接表格式
model = Model.from_data(problem)
result = model.solve(budget=100, stopping_criterion=MaxRuntime(60), seed=1)
```

`pypdms.read()` 也会自动识别 DIMACS 边表文件(`p edge n m` 加若干 `e u v` 行),
即随包附带的 `Instances/DCNP` 数据所用的格式。

## 基于距离的 CNP(DCNP)

DCNP 在恰好删除 `budget` 个节点后,最小化仍处于 `D` 跳之内的节点对数量。

```python
import pypdms
from pypdms import Model, MaxRuntime

problem = pypdms.read("Instances/DCNP/R1/karate.txt")
model = Model.from_data(problem)

result = model.solve(
    problem="DCNP",
    budget=3,
    distance=2,
    stopping_criterion=MaxRuntime(10),
    seed=1,
)

print(f"Best D-hop pairs: {result.best_obj_value}")
print(f"Removed nodes:    {sorted(result.best_solution)}")
```

DCNP 采用与 CNP **完全相同的双种群模因搜索**(可行 + 不可行两个种群、RSC 交叉、
CHNS 局部搜索、周期性交换)。区别在于成本:DCNP 的目标函数每一步 CHNS 都要重建
K-hop 树,因此单次局部搜索远比 CNP 昂贵(300 节点实例上要数十秒,而 CNP 是毫秒
级)。正因如此,DCNP 使用**一套不同的默认超参数**,并自动应用:

| 旋钮 | CNP 默认 | DCNP 默认 | 原因 |
|------|----------|-----------|------|
| `population_size` | 6 | **4** | 初始化成本 = `种群 x 2 x 单次CHNS`;更小的种群才能在时间预算内跑完初始化并完成迭代更替。 |
| `transfer_interval` | 50 | **5** | 总共只跑几十代,间隔太大则交换永不触发。 |
| CHNS `random_idle_product` | 2000 | **100** | 限制单次的空闲步预算;在 `D=2` 下把单次 CHNS 提速约 3 倍且质量不降。 |
| CHNS `random_min_idle_steps` | 40 | **20** | 大 `D` 时的约束下限,此时每次目标评估都很贵。 |
| CHNS `random_max_idle_steps` | 1000 | **80** | 同为空闲步预算上限。 |
| CHNS `random_batch_max` | 50 | **15** | 每步更小的破坏批规模。 |

`population_size` / `transfer_interval` 的替换**仅在你把它们保留为库默认值时**
生效;传入显式值(或 `SolverParams` 上任意 `chns_*` 字段)即可覆盖。这些默认值是
在 100–500 节点的实例(USAir97、Circuit、Ecoli)上以数分钟预算调出来的。若沿用
旧的 CNP 默认,DCNP 在 USAir97(`k=30, D=3`)下 180 秒内跑 **0** 代——种群始终走
不出初始化;换用 DCNP 默认后即可正常演化。

> **关于大 `D` 的说明。** 当 `D >= 3` 时,成本由 K-hop 树的 BFS 维护主导。剖析
> 发现瓶颈并非目标求和(已是 O(n)),也非贪心选点扫描,而是 `bfsKTree` 这一
> BFS 热路径本身。现已对其做了**精确的数据结构优化**(把邻接表铺平为 CSR、用
> `vector<uint8_t>` 标志数组替代内层每条边的哈希查找),在保持目标值逐字节一致
> 的前提下,把单次 CHNS 提速约 **2.7–3.6 倍**(USAir97 `k=30, D=3`:约 29s → 约
> 11s;`k=20, D=2`:约 11s → 约 3s)。至此 `D >= 3` 的中等实例也能在数分钟预算内
> 正常演化。
>
> 进一步的*完全增量式*目标(删点时只更新受影响的点对而完全不重跑 BFS)理论上
> 可行,但因为删除一个节点会使许多点对的最短路变长,精确增量更新的代价与现在的
> 部分 BFS 相当,收益有限。对更大实例的 `D >= 3`,仍可用更小的
> `population_size=3` 和 `chns_random_min_idle_steps=10` 进一步压低单次成本。

如需进一步调优,CHNS 预算通过 `SolverParams` 暴露:

```python
from pypdms import Model, MaxRuntime, SolverParams

# 例如为大 D 实例把 DCNP 调得更轻量
params = SolverParams(
    population_size=3,
    transfer_interval=5,
    chns_random_idle_product=60,
    chns_random_min_idle_steps=10,
    chns_random_max_idle_steps=60,
    chns_theta=0.3,
)
result = model.solve(problem="DCNP", budget=30, distance=3,
                     stopping_criterion=MaxRuntime(120), seed=1, params=params)
```

### DCNP 进一步加速的方向

CSR + 标志数组的热路径优化已落地(零风险、精确)。若还需更快,按性价比排序,
以下方向尚未实现,供参考:

1. **候选列表 / 邻域限制**(下一个性价比选择)。`findBestNodeToRemove` 目前对
   每个活跃节点做一次精确"试删"评估;可先用便宜的代理(节点度、`treeSize_`、
   或预计算的介数中心性)预筛出 top-M 候选,只对它们精确评估。把 O(n) 次精确
   评估降到 O(M)(M << n)。剖析显示这一步并非当前瓶颈,故加速有限且会带来
   *微小的质量波动*(不再是精确同解)。
2. **并行化**。`findBestNodeToRemove` 的各候选评估相互独立,可用 OpenMP 并行。
   但当前评估通过 `removeNode`/`addNode` **修改**共享图状态,需先实现一个
   *非变异*的"试删"增量评估器(只算目标差值、不改 `intree_`/`treeSize_`),
   属于较大重构。
3. **更激进的问题相关默认**。对更大实例的 `D >= 3`,用更小的
   `population_size=3` 配合更低的 `chns_random_min_idle_steps`(如 10)进一步压低
   单次 CHNS 成本,以换取更多代数。这是纯参数手段,无需改代码。
4. **完全增量式目标**(收益存疑)。删点时只更新受影响的点对而完全不重跑 BFS。
   因为删除一个节点会使许多点对的最短路变长,精确增量更新的代价与现有的部分
   BFS 相当,预计收益有限——除非配合近似(容忍目标值的小误差)。

## 停止准则

PyPDMS 内置三种停止准则:

| 类 | 说明 |
|----|------|
| `MaxIterations(n)` | 运行 `n` 次求解器迭代后停止 |
| `MaxRuntime(s)`    | 运行 `s` 秒墙钟时间后停止 |
| `NoImprovement(n)` | 当最优目标连续 `n` 次迭代未改善时停止 |

停止准则可以是任何符合 `Callable[[float], bool]` 的可调用对象,因此你也可以自定义
(例如目标阈值、外部信号)。

## 求解器参数

传入 `SolverParams` 来调节搜索:

```python
from pypdms import Model, MaxIterations, SolverParams

params = SolverParams(
    population_size=10,      # 每个(可行/不可行)种群的规模
    offspring_count=2,       # 每代每个种群生成的后代数
    transfer_interval=50,    # 可行↔不可行交换之间的代数间隔
    partial_ratio=0.95,      # 不可行预算 = floor(budget * partial_ratio)
    beta=0.9,                # RSC 交叉保留共享节点的概率
    search="CHNS",           # 局部搜索策略
)

result = model.solve(
    budget=20,
    stopping_criterion=MaxIterations(500),
    params=params,
    seed=42,
)
```

`SolverParams` 还提供若干 `chns_*` 字段,用于覆盖 CHNS 局部搜索预算
(`chns_random_idle_product`、`chns_random_min_idle_steps`、
`chns_random_max_idle_steps`、`chns_random_batch_max`、`chns_theta`、
`chns_max_idle_steps`)。它们默认为 `None`(使用原生默认,针对 CNP 调优);对
**DCNP** 最为关键——见上文 DCNP 一节。

### 搜索策略

`params.search` 接受:
- `"CHNS"` —— 随机化批规模与空闲步(默认,自适应)
- `"CHNS-ADAPT"` —— 基于空闲步的自适应批增长
- `"CHNS<N>"` —— 固定批规模 N(例如 `"CHNS5"`)

## 结果

`Model.solve` 返回一个 `Result`,包含:
- `best_solution: set[int]`
- `best_obj_value: int`
- `num_iterations: int`
- `runtime: float`
- `best_found_at_time: float`
- `stats: list[dict] | None` —— 逐迭代轨迹(默认开启)
- `feasible_population: list[tuple[set[int], int]]` —— 最终种群
- `feasible_population_overlap_ratios: list[list[float]]`

## 批量基准测试

`benchmark.py` 在同一实例上用不同种子重复运行求解器多次,并汇总报告最优 / 平均
目标值、平均找到最优的时间以及平均总运行时间。两种问题变体都支持:

```bash
# CNP:Hamilton3000a,预算 300,10 次,每次 30 秒
python benchmark.py Instances/CNP/realworld/Hamilton3000a.txt \
    --budget 300 --runs 10 --max-runtime 30

# DCNP:USAir97,k=20,D=2,5 次,每次 120 秒(自动应用 DCNP 调优默认)
python benchmark.py Instances/DCNP/R1/USAir97.txt \
    --problem DCNP --budget 20 --distance 2 --runs 5 --max-runtime 120
```

`--problem` 选择变体(`CNP` / `DCNP`);DCNP 需要 `--distance`。求解器旋钮
(`--population-size`、`--beta`、各 `--chns-*` 等)均为可选,留空则使用问题相关的
默认值。

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]" --no-build-isolation

# 运行测试
pytest tests/

# 修改后重编译 C++
python buildtools/build_extensions.py --build_type release
```

> **macOS 提示。** 在 Apple Silicon 上重编译并拷贝原生扩展
> (`_pypdms.*.so`)后,`import` 可能被 AMFI 以 SIGKILL(退出码 137)直接杀掉,
> 因为拷贝后的 Mach-O 签名失效。重新做一次即席签名即可修复:
> `codesign -f -s - pypdms/_pypdms.cpython-*-darwin.so`。

## 许可证

MIT —— 见 [`LICENSE`](LICENSE)。
