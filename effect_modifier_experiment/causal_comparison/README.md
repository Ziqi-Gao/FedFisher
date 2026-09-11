# 多轮通信与同质 R-learner 比较实验

本目录在现有效应修饰变量实验上增加真实多轮 FedAvg、同质 R-learner
及统一计时。保持现有结局模型的“训练预测模型 → 治疗/对照预测做差 →
置乱原始协变量”路径，不添加标准 X-learner。已有科学结果和运行命名空间
保留；新结果使用独立 `comparison_*_v1_*` 命名空间。

规范见 [AGENTS.md](AGENTS.md)；固定方法、数据与评估配置以
[protocol.py](protocol.py) 为准。所有默认值写入科学 manifest，不能由请求
任意覆盖。变更默认值需要新的配置/源码溯源与相应试运行验证。

## 比较方法

`protocol.CELLS` 包含以下科学单元；`shallow` 与 `deep` 分别表示
`SyntheticMLP` 与 `SyntheticMLPDeep`。

| 科学单元 | 拟合与通信方式 |
| --- | --- |
| `dslassocov` | 保留现有交互项 logistic LASSO、5 折/30 个 λ 交叉验证、FISTA 和梯度/损失汇总；没有一轮截断 |
| `local_logistic_lasso` | 每站本地 logistic LASSO 选参与拟合，上传一次参数，按样本量 FedAvg |
| `one_shot_fedavg_{shallow,deep}` | 每站本地 MLP，上传一次模型，按样本量平均 |
| `one_shot_group_lasso_{shallow,deep}` | 保留显式交互输入的首层 group-LASSO，本地选参后上传一次模型 |
| `fedavg_{shallow,deep}_e{1,5,10,30}_patience` | 真正多轮 FedAvg；训练内验证集选择轮数，随后用全部训练数据重新拟合 |
| `fedavg_{shallow,deep}_e{1,5,10,30}_budget30` | 真正多轮 FedAvg；每站总共遍历训练数据 30 次 |
| `r_unpenalized` | 站内 nuisance 交叉拟合与本地效应估计，上传一次摘要，服务器求共享效应的无惩罚解 |
| `r_adaptive_lasso` | 与上项相同的站内流程；服务器对共享效应斜率增加 adaptive LASSO，并用 EBIC 选参 |

原始 FedFisher 路径的计时接入位于项目的 `main.py`/`train_model.py`。
它不属于这些 CPU 比较单元；其实际运行须另行满足 GPU 或其他所需执行
配置的适配、资源评估、fixture、中央试运行和注册要求。计时开关本身
不构成绕过调度器的许可。旧结果中没有测量的时间不能补填为零，也不能
与新测得的完整成本直接混作同一时间指标。

原始入口仍会在每种算法的训练路径中计算 Diag 和 KFAC 两类 Fisher；
计时记录通过 `actual_path_comparable=false` 标明这一点。按该方法实际
所需组件相加得到的 `necessary_train_component_sum` 只是组件时间之和，
不等于一次独立方法运行的完整耗时，不能直接用作公平比较的主表时间。

## 本地更新次数与公平性

一个 epoch 是遍历一次站内样本，通常包含多次梯度更新。站点大小不同，
同样的 epoch 数不表示相同的 minibatch 更新次数。代码分别记录本地
epoch、实际梯度更新、处理的样本数、聚合轮数和时间。

固定训练量对照使用同一个结构、初始化种子、batch size 64、SGD 学习率
0.01、momentum 0.9，并匹配每站总样本遍历量：

| 每轮本地 epoch 数 E | 聚合轮数 R | 总本地 epoch 数 |
| --- | --- | --- |
| 1 | 30 | 30 |
| 5 | 6 | 30 |
| 10 | 3 | 30 |
| 30 | 1 | 30 |

这匹配的是数据遍历量，不是墙钟时间。每轮本地模型从上一轮全局聚合
结果开始，并重新建立本地 SGD optimizer；momentum 在轮次之间重置。
minibatch 随机序列按站点、轮次和拟合阶段确定。因此 E 的变化也改变
momentum 重置频率与 minibatch 序列，不能将结果解释成严格隔离的
“只有网络通信次数发生变化”。

`patience` 单元在每站训练数据中固定划出约 20% 的验证样本，保留可用
类别的训练样本，以站点验证损失总和除以验证样本总数进行判断。至少
训练 5 轮；改善超过 `min_delta=1e-4` 才重置 patience，连续 10 轮未
达到该改善后停止。保存实际最低验证损失的轮数，再从原始共同初始化
出发，在全部站内训练样本上重新训练该轮数。选轮与重新拟合的全部
时间、更新和通信都计入。

`max_rounds=200` 是选轮过程的异常保护上限；触及上限但未满足 patience
条件的结果不能成为成功完成标记。这里的 `converged` 表示满足预设
验证停止规则，不表示非凸模型达到数值最优。`budget30` 只声称完成固定
训练量，不声称收敛。

四个 E 是预先声明的敏感性比较；学习率等也是固定配置，不能称为已
选择出最优 E/学习率。任何后续选择都应依据训练内验证或独立的调参
试运行预先冻结，禁止根据正式测试 Top-2/AUROC 选择超参数。

## 两个同质 R 版本

参考论文 [DF R-learner，doi:10.1093/biomtc/ujag034](https://doi.org/10.1093/biomtc/ujag034)
以及[作者的 R 实现](https://github.com/Jin-hongLi/DF-R-learner/blob/main/DFRlearn_main_c1n100.R)。
这里是独立的 Python 同质化改写，不能当作原论文异质模型的直接复现。

每站分别估计 `m_k(X)=E[Y|X,site=k]` 和
`e_k(X)=P(A=1|X,site=k)`。二者都使用 logistic LASSO：

- 外层 5 折交叉拟合，折分配与 X/A/Y 的取值无关。
- 每个外层训练折内部进行 5 折选参；外层待预测样本的标签不会进入
  内层选参、标准化或重新拟合。
- 固定 30 个 λ，几何序列从 0.3 到 0.0003，不使用留出样本计算 λ 网格。
  标准化只使用相应内层训练折；最终 nuisance 拟合只使用外层训练折。
- nuisance 截距不惩罚，所有原始 X 斜率接受 L1 惩罚。任何候选拟合
  不满足既有 logistic 优化器的 stationarity 检查时均失败关闭。
- 单类别训练子集使用显式记录的平滑 Bernoulli 常数
  `(sum(target)+0.5)/(n+1)`；不得从其他折或测试数据借用信息。
  若整个站点没有治疗/对照两组，拒绝拟合。
- 交叉拟合后的 propensity 限制在 `[0.05,0.95]`，记录发生裁剪的数量。

对站点 k 构造 `Z_k=(A-e_hat_k(X))*[1,X]`、`r_k=Y-m_hat_k(X)`，
进行无惩罚残差回归，得到本地效应系数 `b_k`。保留参考实现的工作精度：

\[
\hat\sigma_k^2=\|r_k-Z_k b_k\|^2/n_k,\qquad
P_k=Z_k^\top Z_k/\hat\sigma_k^2.
\]

`P_k` 已包含样本量尺度，不能再乘一次 `n_k`。每站只向服务器上传
`site_id,n_k,b_k,P_k`；服务器 API 不接受个体数据或访问站内数据的回调。
用共同参数 beta 明确施加各站效应同质约束：

\[
\min_\beta\quad
\frac{1}{2N}\sum_k(b_k-\beta)^\top P_k(b_k-\beta)
+\lambda\sum_{j=1}^{p}w_j|\beta_j|.
\]

两个版本的 nuisance 和本地摘要计算相同，区别只有最后的效应惩罚：

- 无惩罚版本直接求解
  `(sum_k P_k) beta = sum_k P_k b_k`；不搜索最终阶段的 λ。
- adaptive LASSO 版本用无惩罚共享解构造
  `w_j=(abs(beta_initial,j)+1e-6)^(-0.5)`，效应截距权重为零。
  从仅保留截距时的 KKT 条件得到 `lambda_max`，取 30 个几何递减
  候选至 `1e-4*lambda_max`，并显式加入 λ=0。
  每个候选用服务器本地坐标下降求解并验证 KKT 残差。

适配后的 EBIC 为
`CD discrepancy + (1+s)*log(N) + 2*gamma*log(choose(p,s))`，
其中 `s` 是非零共享斜率数量，`gamma=1`，共同截距只计一个自由度。
不再使用原异质模型的站点分组数作为自由度。EBIC 相差不超过 `1e-12`
时选择较大的 λ。服务器的坐标下降和 λ 搜索不增加跨站通信轮数。

本地样本数必须大于效应参数个数；残差 Gram 与精度矩阵必须正定且
条件数不超过 `1e12`，`RSS/n` 必须大于 `1e-12`。退化、不可识别或不
收敛均失败关闭，不偷偷加入 ridge、使用伪逆、丢弃站点或生成成功标记。

## 评估尺度与数据限制

沿用 10 个不等样本站点、100 个原始协变量、X1/X2 两个真实信号、
IID/non-IID 及四个 Gaussian/置乱因子组合。控制/本地置乱共用训练数据；
Gaussian/组合组共用训练数据。共同测试集和数据生成机制保持不变。

跨方法主指标在概率尺度计算：结局模型使用
`P_hat(Y=1|A=1,X)-P_hat(Y=1|A=0,X)`，并重新构造一致的
`[1,X,X]`、`[0,X,0]` 输入；R 模型直接返回 `beta_0+X beta`。
R 预测不强行裁剪到 `[-1,1]`，也不伪装为分类 logit。

对每个原始 Xj 进行相同的置乱，计算治疗效应预测绝对变化的平均值，
再评价变量恢复。各方法共用置乱种子与样本；站内置乱使用共同测试集
的不相交、完整站点分片，按分片样本数汇总。`permute/tau_prob` 是新比较
主尺度；`zero` 干预以及结局模型的 `tau_logit` 为补充。R 方法不提供
不存在的 logit 指标或分类准确率。

稀疏模型的重要性可能并列。新评估对 Top-2 边界的等分数变量作均匀、
不放回选择，主字段为 `expected_topk_hits`（期望命中数）、
`expected_topk_precision` 和 `probability_all_signals_selected`（两个
信号同时入选的概率）；AUROC 使用并列各计 0.5 的规则。100 个重要性
全为零时，期望命中数为 0.04、precision 为 0.02、两个信号同时入选
的概率为 `1/4950`、AUROC 为 0.5。按下标打破并列的旧指标仅保存在
`deterministic_index_tiebreak_diagnostic` 中，不能据此宣称完美恢复。

现有结局 logit 中共享的交互系数不意味着概率差是线性函数；按结局
标签划分站点及保持标签计数的 Gaussian 设计，也不能保证站内真实
因果效应完全同质。因此 R 方法是共享线性概率效应的工作模型，参考
`Gram/(RSS/n)` 是同方差工作精度，不能直接继承原论文的全部理论或
声称得到异方差稳健推断。这些限制不改变原实验数据。

## 计时、通信与运行边界

所有方法独立核算完整拟合成本，包含预处理、所有选参候选、本地重新
拟合、Fisher/摘要计算和服务器计算。R 两个版本各自执行完整 nuisance
流程；不因实现上可共享中间产物而少计其中一个方法的时间。数据生成
与评估分别记录，不并入主拟合时间；缺少独立阶段的计时明确标记，
不可把未测量解释为零耗时。

记录墙钟时间、当前进程 CPU 时间、细分阶段、终止原因、实际更新数
及通信 payload。单进程串行模拟的本地计算时间不是部署后各站并行
运行时间，也不包含排队或真实网络延迟。CPU/GPU 结果分别呈现；原始
GPU 路径的计时必须显式同步设备。

R 每站一次摘要上传，逻辑编码为两个 int64 加 float64 的系数向量和
完整精度矩阵，服务器集中评估前不需要回传效应参数。模型部署或评估
分发不包含在该训练通信量内。FedAvg 另计训练广播、模型上传及选轮
所需的验证广播/损失上传。所有字节数表示声明的数值 payload，排除
运输协议、序列化 framing、加密和真实网络耗时。

增加代码、禁用注册提案或请求示例不会启动或提交任何计算。运行只能
由中央 ServerScheduler 按独立的新任务名准入，通过现有唯一前台适配器
执行。请求仅选择已审查的 variant/cell 和确定性分片，不指定 CPU/GPU、
路径或任意超参数。提案、参数 schema、fixture 和有界试运行先经评审；
由调度器操作者验证和提交，科学完成与代表性耗时通过验证后再进入
正式种子集合。不得直接调用训练模块、旧 launcher 或新增本地后台任务。

源码和小型配置保存在项目；持久结果位于
`/data/del6500/FedFisher/effect_modifier_experiment/outputs/`；测试、缓存和
临时文件位于 `/scr/del6500/FedFisher/`。结果通过 format-v2 manifest、
源码/config 溯源、语义身份、产物哈希和完成标记共同验证。部分或无效
输出失败关闭；仅校验通过的完成单元可以跳过。任务状态、测试证据及
操作者交接记录集中维护于 `docs/codex/SCHEDULER_HANDOFF.md`。

## 完成结果的汇总

`scripts/report_causal_comparison.py` 是只读科学产物的前台报告入口。
参数为 `--cohort pilot|formal`、`--cells` 后接已完成的方法单元列表，以及
FedFisher `/data` 输出目录或 `/scr` 子树内的 `--output-prefix`。
它不会训练、生成数据、推理或提交任务。

汇总前必须通过所有指定单元、两个 split 和四个 variant 的科学完成
校验；任何缺失或失败都会停止报告，不会悄悄删去不利种子。报告检查
相同方法的配对训练/拟合/模型指纹，以及跨方法数据和置乱一致性。
输出 CSV 包括每组均值、跨种子标准误、各组相对 control 的配对差异和
`combined - Gaussian - local + control` 交互项；JSON 保留逐种子数值。
单种子试运行的标准误记为缺失。拟合时间包含所有训练、选参和聚合
成本，评估时间另列。原始 GPU FedFisher 的侧录时间不混入这份 CPU 表。
