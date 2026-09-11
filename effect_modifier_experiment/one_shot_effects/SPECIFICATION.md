# One-shot effect methods：科学与结果契约

本文件定义 `one-shot-effects-v1` 的 estimand、模型、摘要、失败条件和验收范围。
入口索引和操作边界见 [README.md](README.md)；固定 recipe 以
[protocol.py](protocol.py) 为准。任何参数改动应形成新的配置与溯源，不能按正式结果
反向修改 DGP、基函数、线性化门槛或重要性定义。

## 1. 共同样本、信息与权重

站点 s 原始样本数记为 n_s；N=Σ_s n_s，固定 π_s=n_s/N。观测行仅包含原始
协变量 X、处理 A、二元结局 Y 和唯一 row ID。站内按独立随机排列切分
n_T=floor(.70n_s)、n_V=floor(.15n_s)、n_H=n_s−n_T−n_V。
划分流固定为 `SeedSequence([seed,8300009,site_id])`，不读取 X/A/Y 值。
方法共享全部 T/V/H ID、特征坐标、目标权重和 H 扰动计划。

T 用于拟合及其内部调参；V 仅用于该方法预先声明的选择；H 用于上传前评分摘要。
本版本选完参数保留 T 模型，不合并 T+V、不用 H 重拟合。严格服务器只接收不可执行的
序列化数值摘要，不接受原始行、函数或医院回调。评价代码可以读取独立测试真值及 H
来验证摘要，但不能把这条模拟诊断路径作为部署通信接口。

所有混合平均都采用 Σ_s π_s×站内均值。小数取整后的 n_T/n_V/n_H 不重新定义 π。
对 raw-sum 矩阵施加 π_s/n_split,s；对已经是均值的 Fisher/B/C 只施加 π_s。
本协议没有 privacy、差分隐私或安全聚合保证；“摘要不含原始行”并非完整隐私证明。

## 2. 预测尺度、扰动与评价

所有方法实现 `predict_tau(X)`。二分类模型令

\[
\hat\tau_\theta(x)=\hat p_\theta(Y=1\mid [1,x,x])
                    -\hat p_\theta(Y=1\mid [0,x,0]).
\]

DR/R 返回其效应模型预测；不构造对应 logits、分类准确率或反事实个体真实结局。
DR/R 不把预测截到 [−1,1]，否则二次摘要不再对应评价模型；另外报告越界比例。

H 的每站、每变量 j 有 5 次完整边际置乱。用相同原始行和置乱行定义

\[
I_j(\hat\tau)=\sum_s\pi_s\frac{1}{5n_{H,s}}
 \sum_{r=1}^{5}\sum_{i\in H_s}
 \frac{[\hat\tau(X_i)-\hat\tau(X_i^{(j,r)})]^2}{2}.
\]

主 H 置乱种子为 `8400011+seed+1000003*site_id`，保存计划指纹；
外部测试的对应偏移为8500009，应从实际配置/结果核对派生值，不能靠图表重建另一组扰动。
方向 U 的 T 子样本扰动另有流、用途和重复次数，不能和 H 评分扰动混淆。

外部测试评分是平均**绝对**变化，而非上式平方评分。SCMV2 测试先独立抽取
site_id~Categorical(π)，再生成 X/A/Y；置乱限制在测试站点内，并以原 π 汇总站内
均值。CATE 平方误差、预测均值/二阶矩及越界率使用同一 π；CATE RMSE 为加权均方
误差开方，估计异质性为目标混合下预测的中心二阶矩。每个目标测试站点至少需要2行，
否则拒绝所需的站均值/置乱计算，不能悄悄重归一化成另一个目标分布。

legacy 独立测试保留原共同 pooled 测试集定义；没有 site_id 时不伪造站点归属，
也不标成 SCMV2 的目标混合条件。两种外部扰动口径在结果中明确区分。

恢复报告 `expected_topk_hits`、`expected_topk_precision`、
`probability_all_signals_selected`、`auroc`。边界分数精确并列时，在并列变量中
均匀无放回选取所需名额。两组排名的 Top-2 集合一致概率采用独立均匀并列选择；
常数排名的 Spearman 相关为不适用，不伪造相关系数。null 没有真 modifier，恢复
Top-k/AUROC 不适用；分数排序不是显著性检验或 FDR 控制。

## 3. FedFisher-EM 的严格对角聚合

浅/深层 MLP 沿用仓库架构，在共同原始 `[A,X,A*X]` 坐标初始化。本地不独立标准化。
共同 float32 随机初始化后转换到 CPU float64；本地训练默认30 epochs、batch64、
SGD learning rate .01、momentum .9，固定预算，无 V/H 选点。

令二分类 logit 差为 l_1−l_0，模型概率为 p_0,p_1。本地对角 Fisher 是

\[
F_{s,d}=\frac1{n_{T,s}}\sum_{i\in T_s}
 p_{s,0}(A_i,X_i)p_{s,1}(A_i,X_i)
 \left[\partial_{\theta_d}(l_{s,1}-l_{s,0})(A_i,X_i)\right]^2.
\]

这是模型 categorical Fisher 的精确二分类标签期望，均值分母是 n_T。
它不使用真实 SCM propensity/潜在结局，也不是 observed-label empirical Fisher。
训练的 Y 仍进入本地结局模型拟合。

strict 服务器解

\[
Q(\theta)=\tfrac12\sum_s\pi_s\sum_d F_{s,d}(\theta_d-\theta_{s,d})^2.
\]

正曲率坐标取 `sum(pi*F*theta)/sum(pi*F)`；实现按坐标最大曲率缩放后运算，保留
微小正曲率。仅当该坐标所有站点曲率精确为0时，用 π 加权 FedAvg 坐标作为声明的
最小化解。聚合不量化、不加阻尼、不按 V/test 准确率选择迭代点。有限性和相对
站立性检查必须通过，默认相对容差 `1e-10`。

旧规则诊断另外重放每参数张量 s=15 的量化、无 π 的平均-Fisher 二次目标、
2000 次迭代及每100步严格准确率提升选点；其 V 只属于模拟 evaluator。
同一组量化上传还求解相同旧目标的精确解，从而区分“选点/求解变化”与“π/量化变化”。
该诊断使用新 T/V/H、CPU float64 和新 π；不能声称重现旧 CUDA float32 历史结果。
旧规则全零张量量化的0/0等退化行为显式失败，不静默修复后冒充原算法。

## 4. FedFisher 聚合后评分摘要

定义 g_s(x)=∇_θτ_{θ_s}(x)。Fisher 梯度方案使用评分度量
M_s=diag(F_s)+.001I；欧氏梯度方案使用 M_s=I。固定随机正交方案采用 Fisher
评分度量，但 U 不依赖训练数据。**评分度量阻尼不进入上节聚合目标。**

在 T 的预定子样本上，收集原始及各候选变量扰动行的 M_s^(−1/2)g_s(x)。
默认最多8行、每变量1次置乱，所有候选变量受到相同规则；真实信号索引不可用于
选择行、变量或方向。Fisher/欧氏方案使用两遍流式随机化二阶矩 range finder，
rank 为16/32/64，oversampling默认8；只存 D×(r+k) 而非 D×D。
固定随机方案用固定种子的 Gaussian QR，并把能量审计成本纳入计时。

报告抽样/方向 seed、行数及ID指纹、梯度数量/遍数、Rayleigh 值、已观察能量与
投影残差。`observed_relative_projection_energy_error` 只针对这一有限梯度集合，
既不是完整 T 二阶矩的精确主空间误差，也不是总体或谱误差上界。

U 的列正交，定义

\[
q_s(x)=U_s^T M_s^{-1/2}g_s(x),\qquad
z_s(\theta)=U_s^T M_s^{1/2}(\theta-\theta_s),
\]
\[
\tilde\tau_{s,\theta}(x)=\tau_{\theta_s}(x)+q_s(x)^Tz_s(\theta).
\]

H 上令 a_s(x)=[τ_s(x);q_s(x)]，v_s(θ)=[1;z_s(θ)]，上传

\[
B_{s,j}=\frac1{5n_{H,s}}\sum_{r,i}\frac{
 [a_s(X_i)-a_s(X_i^{(j,r)})][a_s(X_i)-a_s(X_i^{(j,r)})]^T}{2}.
\]

一次 payload 包含 θ_s、F_s、n_s、n_T、U_s、所有 B_sj 和完整协议元数据。
服务器先得到 θ_F，再算 `sum_s pi_s v_s(theta_F)' B_sj v_s(theta_F)`。
该分数对**压缩后的站点局部线性近似**精确，不是完整非线性最终模型的精确分数，
也不是本地模型重要性的简单平均。

评价先在相同 H 和置乱上计算最终非线性模型精确分数，再用完整参数梯度与
θ_F−θ_s 的方向导数计算未压缩泰勒近似，最后才比较低秩方案。完整梯度诊断不上传
各变量 D×D 矩阵；它是 evaluator 计算。保存每变量误差、最大误差、tau RMSE、
排名相关、并列敏感 Top-2 一致性、恢复、欧氏/Fisher 参数距离和摘要/直接低秩误差。

## 5. DR-Sieve：伪结局与共享基

每站分别拟合 e_s(X)=P(A=1|X,s)、μ_s0(X)=E[Y|A=0,X,s] 和
μ_s1(X)=E[Y|A=1,X,s]。两个 outcome regression 按处理组单独拟合，不能把
R-learner 的 m_s(X)=E[Y|X,s] 同时当成 μ_s0、μ_s1。

\[
\psi_i=\hat\mu_1(X_i)-\hat\mu_0(X_i)
 +\frac{A_i}{\hat e(X_i)}[Y_i-\hat\mu_1(X_i)]
 -\frac{1-A_i}{1-\hat e(X_i)}[Y_i-\hat\mu_0(X_i)].
\]

T 内默认5折外层交叉拟合；每次拟合仅用对应外层训练行。内部 logistic LASSO
默认5折 CV、固定30个 λ 从 .3 到 .0003，内层训练集标准化、按行数汇总验证
logloss，绝对 `1e-12` 并列选更大λ，然后仅在外层训练集标准化/重拟合。
CV/helper 复用既有 R 模块，不改旧方法。V 的 nuisance 另在全部 T 上拟合；
V 的 A/Y 只进入 ψ 和验证风险。H/test 永不进入 nuisance。

e 预测裁到 [.05,.95] 后形成 AIPW，记录裁剪数量和范围。任何外层训练补集或
全 T 缺一个治疗组均失败；非恒定 target 行数小于内层折数失败。非空且恒定的
outcome target 使用 `(sum(Y)+.5)/(n+1)`，包括单类内层训练折，报告发生位置。
常数处理只用本次训练行；不丢弃站点、不读真实 propensity、不把优化不收敛当成功。

主基函数固定为 u_j=tanh(X_j) 上的三次 B-spline。结点向量是
`[-1,-1,-1,-1,-1/3,1/3,1,1,1,1]`，6个基函数中每组删除最后一个 index5，
留下 q_j=5，并加一个公共截距，因此维度 D_φ=1+5p。端点取 clamped 连续极限。
linear 消融直接为 `[1,X]`，q_j=1。两者都不做额外中心化/缩放或经验分位数选结点。
基函数配置在看数据前固定，不通过正式测试选择主版本。

## 6. DR-Sieve：服务器目标、选择和精确评分

站点 T/V 分别上传 raw sums `G=sum(phi phi')`、`h=sum(phi psi)`、
`c=sum(psi²)` 和各自 n。服务器构造

\[
G_T=\sum_s\pi_sG_{T,s}/n_{T,s},\quad
h_T=\sum_s\pi_sh_{T,s}/n_{T,s}.
\]

先按固定规则解
`[G_T+1e-6*diag(0,1,...,1)] beta_initial=h_T`，检查矩阵方程残差。
公共截距不加 ridge；这个 ridge 只用于 adaptive 权重
`omega_j=(norm(beta_initial,j)+1e-6)^(-.5)`。
最终目标不包含 ridge：

\[
\tfrac12\beta^TG_T\beta-h_T^T\beta+
 \lambda\sum_{j=1}^p\omega_j\sqrt{q_j}\|\beta_j\|_2.
\]

λ_max 是仅截距模型的最大组梯度范数除以 `omega_j*sqrt(q_j)`。
默认30个几何间隔λ从 λ_max 到 `1e-4*lambda_max`，**不额外追加0**。
仅当 λ_max 精确0时重复0至声明的路径长度并按规则取首项。
优化为带重启的单调 FISTA；每个候选必须满足组 KKT 与 proximal-gradient
残差检查。默认最大20000次、容差 `1e-7*max(1,norm(h_T,inf))`；达到安全上限
且不满足残差即失败。初始化线性方程阈值为 `1e-9*max(1,norm(h_T,inf))`。

V 风险为没有1/2系数的

\[
R_V(\beta)=\sum_s\frac{\pi_s}{n_{V,s}}
 [\beta^TG_{V,s}\beta-2h_{V,s}^T\beta+c_{V,s}].
\]

风险绝对 `1e-12` 内并列取较大 λ。保留所选 T 系数，不作 T+V/H refit。
本地 Gram 允许奇异；上传矩阵必须有限、对称，增广训练/验证矩应为 PSD，
H 块也应为 PSD；矩阵形状、计数、共享基和配置不一致均拒绝。

H 使用相同置乱定义
`C_sj=mean(delta_phi delta_phi')/2`。加性基下只有变量 j 对应的 q_j×q_j 块
可能非零，截距差为0，因此只上传 p 个小块。
服务器分数为 `sum_s pi_s beta_j' C_sj beta_j`，对同一个**未裁剪**基函数模型、
H 和扰动精确；没有参数泰勒近似误差。数值测试需与完整 C 及逐行直接评分一致。

结果包含 `coefficient_path`、每 λ KKT/优化残差/目标/风险、adaptive 权重、
初始化残差、选择位置和实际通信。只读 validator 通过摘要算术复核整条候选路径、
V 选点、选中模型和 H 分数，不重新训练或读取个体行。

## 7. 共同协议基线

同质 R adapter 保留既有 T-only nuisance m/e、局部无惩罚残差回归和 CD precision：
`Z=(A-e)[1,X]`、`beta_s=(Z'Z)^(-1)Z'(Y-m)`、
`P_s=Z'Z/(RSS_s/n_Ts)`。它仍要求本地 OLS 可识别、条件数和残差方差合格，
这项限制不适用于 DR-Sieve。

共同目标是

\[
\sum_s\frac{\pi_s}{2n_{T,s}}
 (\hat\beta_s-\beta)^TP_s(\hat\beta_s-\beta)+\lambda\sum_jw_j|\beta_j|.
\]

令 N_T=Σn_Ts，wrapper 在调用未改变的旧服务器前，把每站 precision 乘以
`N_T*n_s/(N*n_Ts)`，旧 solver 再除 N_T，故最终恰为 π_s/n_Ts，权重只用一次。
无惩罚版本直接解共享二次目标；adaptive-LASSO 沿用旧初始无惩罚系数权重和
**追加λ=0** 的 EBIC 路径，EBIC 的 N 为 N_T，不含 V/H。这与 DR 的 V 风险选择不同。
V 对这两个 R 版本不用。H 只提供线性 C 块，不进入估计/选择。

R 新 wire/method 名是 `r_common_target_v1`。只读校验检查目标权重、无惩罚解、
所选 KKT/EBIC、H 分数和记录的路径证书；旧服务器未返回每个候选系数，故不宣称
从原始摘要逐个重算整条 R EBIC 路径。旧文件及历史结果不修改。

FedAvg 共同协议对照训练设置与 MLP locals 一致，服务器取 `sum_s pi_s theta_s`，
不计算 Fisher、不使用 V。它用于一次聚合对照；已有多轮 FedAvg/dsLasso 的配方
继续在旧 comparison 中保留。本协议的单次平均不能被描述为不限通信的训练基线。

## 8. SCMV2 与 legacy

SCMV2 先 S、再 X、再 A、再 Y，不按实现的 Y 重新分站或固定标签计数。
默认10站点，各站 n_s 独立均匀抽自整数500…1000，p=100，独立测试10000行。

\[
g=.8\tanh X_3+.5\tanh X_4,\quad b=.25+.5\operatorname{sigmoid}(g),
\]
\[
e=\operatorname{clip}\{\operatorname{sigmoid}(.8X_3-.6X_5),.05,.95\},
\quad\tau=.1\sin X_1+.1\tanh X_2,
\]
\[
\mu_0=b-\tau/2,\quad\mu_1=b+\tau/2,\quad
A\sim\mathrm{Bernoulli}(e),\quad Y\sim\mathrm{Bernoulli}(\mu_A).
\]

μ_a 必须在 [.15,.85]；检查失败拒绝生成，禁止事后裁剪 μ 来改变 CATE。
X1/X2 是 modifier，X3 是预后变量兼治疗混杂变量，X4 是预后变量，X5 影响治疗，
其他变量在结构方程中为噪声。相关场景的噪声可与信号统计相关，不能将“结构噪声”
等同“统计独立”。所有站点共享 outcome/treatment/effect 函数。

| 场景 | X | CATE/目的 |
| --- | --- | --- |
| base | N(0,I) | 上述加性 τ |
| covariate_shift | N(m_s,I) | 函数同 base，隔离协变量均值偏移 |
| correlated_shift | N(m_s,Σ)，Σ_jk=.5^abs(j-k) | 同 base，增加相关性 |
| interaction | 同 base | `.2*tanh(X1)*tanh(X2)`，明确的加性 sieve 错设压力测试 |
| null | 同 base | τ=0；不报告恢复 AUROC/Top-k |

固定均值设计是 `linspace(-.75,.75,num_sites)` 的各站标量复制到每个原始坐标；
单站取0。不根据哪种方法表现更好调整站点均值、关联结构或效果强度。
即使默认 τ 加性，有限 tanh-spline 也只是近似工作模型；interaction 单个场景失效
不代表所有 DR 估计理论失效。条件重采样、oracle nuisance 和 KFAC 都不属于第一版
默认协议，应另设配置与标识；旧按 Y 分站的数据不能被假定为精确站内 Gaussian。

legacy adapter 复用原生成器的站点与共同测试行，只增加共同 T/V/H 划分。
四个 variant 和 iid/noniid 来源定义保留，不能用新的统一评分协议复写旧含义。
新方法结果使用新的命名空间，不直接拼接旧训练总体的成绩。
这里 legacy variant 是数据来源标签；新的 H 始终站内置乱，legacy 外部评分始终
使用共同 pooled 测试集。因而 control/local_permutation、site_gaussian_shift/
site_gaussian_local_permutation 可能形成相同数据与评分协议的重复单元；不能把
这些新结果解读成旧2×2置乱因子实验，也不能从重复标签推导新的因子效应。

## 9. 字节、计时与验收阶段

wire 是 canonical JSON 头和 little-endian 数值数组的 bytes。通信计数包括全部
θ/F/U/B、G/h/c/C、样本数、配置和 metadata，不能只统计模型参数或矩阵。
一个站点一个部署 payload；服务器本地λ搜索/优化不增加消息。共同协议分发和最终
模型下发另列 bytes。模拟 evaluator 的 raw access 不应记成服务器“一次上传能力”。

相同模型/数据/rank 的三种压缩使用公开维度的统一实际上传预算：
`round_up_1024(8*(D*(rank+2)+p*(rank+1)^2)+65536)`。
不足预算的 payload 实际填充并计费，保存 native/padded bytes 与 padding；
超出预算应失败而非截断数据。这是等 rank/等实际 bytes 的协议控制，并非最优压缩率、
最优可变 rank 分配或网络时延实验。

计时同时记录 wall 和 process CPU：本地训练、nuisance CV/refit、Fisher、U构造、
H摘要、序列化、服务器求解/选参/评分和 evaluator 诊断分别保留可识别范围。
runner 的 `train` 总阶段包括所有本地拟合与上传摘要构造；`aggregation` 包括
服务器 λ 搜索、选择和评分。DR/R 的内部细分在各方法审计中，不能把通用 `tuning`
阶段为0解释成“没有调参”；该阶段还单独记录旧 V 选点诊断和匹配本地 FedAvg 参考。
嵌套阶段不重复相加成总耗时；模拟顺序执行总时间不是跨站并行部署延迟。
公共数据准备与不同方法比较的计时边界应由 runner 一致记录；未测量真实网络时间。

阶段1做小型单线程 CPU 数值/数据/通信契约测试。阶段2准备有界 seed0 pilot：
聚合旧规则与 strict 配对、完整梯度线性化、DR 摘要恒等式和 SCM 概率/真值隔离。
阶段3压缩要读取同模型、同数据集、同源完整梯度证据，并同时满足
相对 tau RMSE≤.25、相对最大评分误差≤.25，分母最低1e−8。它们是预先声明的
工程扩展门槛，不是统计定理。失败保留记录，不跳过失败条件直接扩大 rank/seed。

`staged_pilot_plan()` 是有限 rollout，不自动穷举 recipe×dataset。
后续少量场景检查完成并经调度/科学验收后，才另行准备正式种子集合和比较契约。
提案、fixture、小测试或存在 outbox 都不是 operator 许可或 pilot 完成证据。
所有新科学计算通过独立注册的 ServerScheduler 协议v2前台路径；不提交完整300-seed，
不使用旧任务名/launcher，不修改中央状态或其他任务资源。

持久结果绑定 clean commit、source tree/config/provenance hash 和 seed/recipe/dataset；
包括实际 site payload、model、result、started、done 清单与哈希。只读 status 必须
通过任务特定语义校验。部分输出、异源数据和重复完成均失败关闭，不能覆盖修补后
把退出码当成科学成功。

## 10. 来源与推论边界

本文件整理用户提供的实施规范和本仓库实际函数。FedFisher 历史参照为
[原始聚合代码](../../algs/fisher_avg.py)及
[原始一次聚合入口](../../run_one_shot_algs.py)，本版本 strict 算法的差异已逐项列出。
既有 R 数值组件与参考来源见
[rlearner.py](../causal_comparison/rlearner.py)和
[同质 R 文档](../causal_comparison/README.md)，对应用户提供的
[DF R-learner 论文 DOI](https://doi.org/10.1093/biomtc/ujag034)及
[作者公开 R 实现](https://github.com/Jin-hongLi/DF-R-learner/blob/main/DFRlearn_main_c1n100.R)。
DR-Sieve 与 SCMV2 不声称来自该 R 论文，也不伪造另外的原论文引用。

通过软件验证只能支持所检验的数值恒等式、数据边界和优化证据。它不能证明真实
数据的无混杂/重叠假设、nuisance 一致性、effect 工作模型正确、选变量的显著性或
FDR 控制。FedFisher-EM 的线性化/压缩质量和严格聚合恢复表现、DR-Sieve 的相对
性能与稳定性都需要预声明实验；任何创新、优越性或理论主张必须另有证据。
