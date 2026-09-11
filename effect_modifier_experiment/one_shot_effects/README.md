# FedFisher-EM、One-shot DR-Sieve 与共同实验协议

本目录增加两套方法及其共同数据、评估和通信协议：FedFisher-EM 为主要研究方法，
One-shot DR-Sieve 为独立方法和对照。代码实现、数值一致性测试、获准完成的 pilot、
统计有效性和学术创新是不同层面的证据，不能相互替代。

完整科学定义见 [SPECIFICATION.md](SPECIFICATION.md)。本目录继承
[AGENTS.md](AGENTS.md)、[项目维护协议](../../AGENTS.md)和
[调度运行手册](../../docs/codex/SCHEDULER_RUNBOOK.md)。当前任务状态、源码绑定、
测试及 pilot 验收证据只记录在 [SCHEDULER_HANDOFF.md](../../docs/codex/SCHEDULER_HANDOFF.md)，
并以 Git、科学 manifest 和中央调度器只读状态核实；本说明不是运行状态记录。

## 方法与比较范围

| 配方/方法 | T 的用途 | V 的用途 | H 的用途 | 一次上传的内容 |
| --- | --- | --- | --- | --- |
| FedFisher-EM strict Diag | 共同初始化后本地 MLP 训练、站内平均 Fisher；压缩配方还构造方向 | strict 服务器不用 V | 压缩配方在上传前构造评分矩阵 | 参数、Fisher、样本数、方向 U、各变量 B、元数据 |
| FedFisher 旧选择规则配对诊断 | 复用相同本地训练结果 | 仅模拟 evaluator 重放分类准确率选点规则 | evaluator 比较最终模型的精确评分 | strict 部署接口不接收诊断用 V |
| DR-Sieve spline / linear | 三个 nuisance 的嵌套交叉拟合、DR 伪结局、Gram 摘要 | T-only nuisance 生成 V 伪结局，服务器按 V 风险选效应 λ | 固定基函数的精确 C 矩阵块 | T/V 原始求和 G、h、c，样本数，H 的 C，元数据 |
| 同质 R / R-LASSO 共同目标适配 | 复用既有 R 残差回归和站点 CD 摘要 | 不使用；LASSO 用 T 样本数的 EBIC | 线性效应模型的精确 C 矩阵块 | 原始本地 β、precision、样本数、H 的 C、元数据 |
| 共同协议 one-upload FedAvg 聚合对照 | 与 MLP 对照相同的固定本地训练预算 | 不使用 | 独立 evaluator 计算精确评分 | 本地参数、样本数、元数据 |

FedAvg 此处用于分离“同一本地训练结果采用何种聚合”的影响。它是固定预算的一次
参数平均，不能替代既有 [causal_comparison](../causal_comparison/README.md) 中的
多轮 FedAvg。多轮 dsLassoCov-style、旧同质 R 两版本、原始 FedFisher 路径和历史
结果继续保留。新共同 T/V/H 的训练样本量不同，正式比较必须在共同协议下重跑相关
基线，不能拼接历史成绩。本目录不新增标准 X-learner，也不改变已有预测差方法。
legacy variant 在这里标识数据来源；统一的新 H/外部评分可能使 control/local 与
Gaussian/combined 成为重复单元，不能按旧2×2置乱因子实验解释这些新结果。

## 共同约定

- 方法只接收 `Rows(x,a,y,row_ids)` 或 `SitePartitions` 的观测数据。
  `EvaluationTruth` 中的 propensity、潜在结局、CATE、modifier mask 和变量角色
  只供独立模拟 evaluator 使用；服务器不接收个体行或站点回调。
- 每站用独立于 X/A/Y 的随机排列划分 T/V/H：`floor(.70*n)`、`floor(.15*n)`、
  剩余行。所有方法使用相同划分；不按 A/Y 分层或挑选，不用 H 重训，不在选参后合并 T+V。
- 固定目标权重为原站点 `n_total / sum(n_total)`。T、V、H 分别先取站内平均，
  再施加一次目标权重。不能改用 T/V/H 的实际行数重新定义站点混合权重。
- `predict_tau(X)` 始终输出概率尺度效应。结局 MLP 使用 `[1,X,X]` 与 `[0,X,0]`
  的类别 1 概率差；DR/R 直接输出效应工作模型。DR/R 不做范围截断，另报越界比例。
- H 的主扰动固定为站内边际置乱，每个变量 5 次。相同 H、相同置乱计划和相同权重
  才能用于摘要与精确参考的比较。边际置乱不是条件置换。

两种评分分别保存，不能混用：

| 评分 | 定义 | 样本/权重 |
| --- | --- | --- |
| 外部测试评分 | `mean(abs(tau(X)-tau(X_permuted_j)))` | SCMV2 按测试站点置乱，按原 π 汇总站均值；legacy 保留原共同测试集的 pooled endpoint |
| H 平方扰动评分 | `sum_s pi_s * mean_H,repeats((tau(X)-tau(X_permuted_j))**2/2)` | 共同 H，每站每变量 5 次边际置乱 |

SCMV2 外部 CATE RMSE、估计异质性及越界比例也按预声明目标混合计算。
`PEHE = mean((tau_hat-tau_true)^2)`，因此本协议 `sqrt_PEHE = CATE_RMSE`。
legacy 测试对象不伪造 SCM 真值或站点归属；其 CATE RMSE 为不适用。
恢复指标沿用精确边界并列下均匀抽取的规则：`expected_topk_hits`、
`expected_topk_precision`、`probability_all_signals_selected`、`auroc`。
null 场景这些恢复指标不适用，应查看 CATE 误差、估计异质性和变量分数。

## 默认配置与局限

[protocol.py](protocol.py) 是可审查配方的单一入口，固定 pilot 为 seed 0、默认
10 站点、每站 500–1000 行、100 个原始变量、10000 行独立测试数据。

FedFisher-EM 使用浅层或深层 MLP、30 个本地 epoch、batch 64、SGD 学习率 .01、
momentum .9，以及无量化 CPU float64 参数和平均 categorical Fisher。
strict 聚合无阻尼；评分度量单独使用 `score_damping=.001`。
方向 U 的默认构造只抽取 **最多 8 行 T，每个变量 1 次置乱**，采用预先固定的
随机化近似；这不是完整 T 梯度二阶矩的精确分解。H 摘要仍用完整 H 和 5 次置乱。
记录的方向能量误差只针对这组有限训练梯度，不能当作总体误差界。

压缩比较包含 Fisher 梯度、欧氏梯度、固定随机正交三种方向，以及 rank 16/32/64。
同一 rank 的等字节比较采用**实际序列化填充**至公开维度决定的预算，同时保留
未填充字节数；不能把这种比较解释为最优编码效率或跨 rank 的最优信息分配。

DR 主版本固定 `tanh(X)` 上的加性三次 B-spline，每变量保留 5 列和一个公共截距；
linear 消融直接使用 `[1,X]`。固定 slope-only ridge `1e-6` 只产生 adaptive 权重，
不加入最终 group-LASSO 目标。默认 5×5 折 nuisance、30 个 nuisance λ，
30 个效应 λ，用 V 风险选择。它允许本地 Gram 奇异，但必须满足矩阵和 KKT 检查。
加性基函数不包含交互，有限 spline 也不保证精确表示默认真 CATE。
DR 名称不代表 nuisance 正确、模型无偏、变量选择具有 FDR 保证或已证明优于其他方法。

## 模块与接口

| 模块 | 主要责任/入口 |
| --- | --- |
| [types.py](types.py)、[data.py](data.py) | 观测/真值类型隔离；`partition_rows`、`generate_scm`、`generate_legacy` |
| [interventions.py](interventions.py) | 共享置乱计划、合法性检查、指纹 |
| [fisher_models.py](fisher_models.py) | 共同模型坐标、概率效应、梯度、平均 Fisher |
| [fedfisher_em.py](fedfisher_em.py) | `fit_local`、`build_training_payload`、`aggregate_training_payloads`；压缩 `build_payload`、`aggregate` |
| [fisher_summary.py](fisher_summary.py) | T 上方向构造、H 的 B、完整梯度和压缩诊断 |
| [basis.py](basis.py)、[dr_nuisance.py](dr_nuisance.py) | 固定共享基；`fit_dr_pseudo_outcomes(T,V,config,seed)` |
| [dr_sieve.py](dr_sieve.py) | `build_local_payload`；`aggregate(payloads,config)`；`validate_server_result`；`DRSieveModel(beta,basis_config)` |
| [r_baseline.py](r_baseline.py) | `build_local_payload`；`aggregate(payloads,penalty,config)`；共同目标 R 的只读校验 |
| [fedavg.py](fedavg.py) | 固定本地训练预算的一次参数平均聚合对照 |
| [wire.py](wire.py) | bytes 编解码、实际通信记账；只接受数值数组和 JSON 元数据 |
| [evaluation.py](evaluation.py) | 精确外部/H 评分、tie-aware 恢复、CATE 误差及线性化门槛 |
| [protocol.py](protocol.py)、[artifacts.py](artifacts.py) | 固定科学配方、阶段前置条件、不可覆盖且绑定溯源的完成证据 |
| [runner.py](runner.py)、[model_io.py](model_io.py) | 经前台适配器调用的科学流程；模型序列化与只读重建 |
| [validation.py](validation.py)、[audit_validation.py](audit_validation.py) | 读取已保存模型/摘要，校验配置、局部拟合证据、目标、评分、通信和完成条件 |
| [reporting.py](reporting.py) | 汇总调用方已验收的完成记录，生成小型压缩 pilot 配对审查表 |

训练/服务器 API 是库接口，不构成直接运行科学计算的授权。

## 分阶段验证与调度

1. 小型单线程数值测试验证目标/风险/评分恒等式、梯度/KKT、T/V/H 与真值隔离、
   一次上传边界、实际 bytes，以及篡改和失败拒绝。它们不产生正式科学结论。
2. `staged_pilot_plan()` 先列 `scmv2_base`、`legacy_control_iid` 的浅/深层 aggregation
   与 full-gradient linearity，以及 DR linear/spline。聚合诊断区分旧选点规则、
   相同旧二次目标的精确解和新的目标加权 strict 解。
3. 压缩 recipe 必须已有同数据集、同模型的合格完整梯度线性化证据。
   两项工程门槛均为 `.25`：相对 tau RMSE 和相对 score 最大误差，分母下界 `1e-8`。
   门槛失败保留结果并停止该条件的大规模低秩扩展；增加 rank 不能修复泰勒线性化误差。
4. 然后才做少量 scenario/基线检查。正式多 seed 比较需要另行审查的正式任务契约，
   不能将有界 pilot recipe 扩成 300-seed 请求。

提案任务名为 `effect-modifier-one-shot-effects-pilot`，参数只有 `recipe`、`dataset`。
可审查的阶段清单见 [staged-pilot-plan.json](../server_scheduler/one-shot-effects-v1/staged-pilot-plan.json)，
首阶段请求示例保存在同目录的 `examples/`；这些文件不是已发送请求。
必须经独立注册、fixture、operator 准入；本地 outbox 文件不是已提交任务。
使用中央协议 v2 前台适配入口，遵守分配的线程数；不通过旧 launcher 或旧任务名绕过。

结果置于 `/data/del6500/FedFisher/effect_modifier_experiment/outputs/` 的独立
`ose_v1_*_pilot` 命名空间。源码和小型配置在 canonical repo；测试缓存、
可再生成报告和 outbox 在 `/scr/del6500/FedFisher/`。
完成要求 clean-source provenance、配置/身份、payload/model/result 哈希和 `done.json`
全部一致；进程退出 0 不是科学完成证据，部分输出不得覆盖或直接续跑。

`result.json` 的 `models.primary` 保存模型标识、`fit`、`test`、`holdout`；聚合诊断
可以另有 `legacy_validation_selected`、`strict_legacy_uploaded_objective` 和
`matched_fedavg`。`local_audits` 保留站点拟合证据，`diagnostics` 保留旧规则或线性化
诊断，`data_fingerprints` 校验共同数据。`communication` 同时报告 native/upload/
padding bytes 及单列协议/模型广播；`timing` 与方法内部阶段明确计时范围。
外部 `test.scores` 与 `holdout.exact_squared_scores` / `summary_squared_scores`
不能相互替代，后两者的比较才检验一次上传评分摘要。

## 来源与保留实现

本规范整理用户提供的 FedFisher-EM / DR-Sieve 实施要求；SCMV2 是此项目的显式模拟
设计，不附会外部数据来源或新方法论文。原始 FedFisher 的仓库实现入口是
[fisher_avg.py](../../algs/fisher_avg.py)、[run_one_shot_algs.py](../../run_one_shot_algs.py)
和 [compress_fisher.py](../../utils/compress_fisher.py)，它们作为历史算法定义保留。

既有同质 R 的参考为用户提供的 DF R-learner 论文
[doi:10.1093/biomtc/ujag034](https://doi.org/10.1093/biomtc/ujag034)及已在旧模块标注的
[作者 R 代码](https://github.com/Jin-hongLi/DF-R-learner/blob/main/DFRlearn_main_c1n100.R)。
本目录的 R 是共同权重适配，DR-Sieve 则是另一种伪结局回归；二者都不能宣称原封不动
复现论文中的异质 effect-modifier 设定。详细差异见
[既有 R 文档](../causal_comparison/README.md)与本目录科学规范。
