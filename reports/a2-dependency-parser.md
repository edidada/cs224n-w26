# CS224n Winter2026 A2 实验报告（待远程训练验收）

来源为[官方八页手册](https://web.stanford.edu/class/cs224n/assignments_w26/a2.pdf)和[官方starter](https://web.stanford.edu/class/cs224n/assignments_w26/a2.zip)，ZIP SHA256 `68854395ef802cb974222e4d0a843bf8c1ca04d8c49de1ebe88fa5129d72ca56`。手册实际下载507426字节、SHA256 `a6a3c3e6141c0ded7940bc41476717f0ad3ead77199b9deab099cbafaa62cf52`，完整读取，依存图另实际查看。原始ZIP导入为cf16c607，独立分支`edidada/a2-dependency-parser`，独立YAML为`a2-dependency-parser.yml`。本项目是官方发行包镜像，不是官方GitHub starter fork。

## 1. Word2Vec推导

令词表大小为M，U为d×M且各列为外部词向量，中心向量v为d×1。令z=Uᵀv，p=softmax(z)，y为真实外部词o的一位有效向量。

(a) 交叉熵`-Σ y_w log p_w`只有w=o项非零，故等于`-log p_o`。

(b) 写`J=-u_oᵀv+log Σ exp(u_wᵀv)`。对v求导得`∂J/∂v=-u_o+Σ p_w u_w=U(p-y)`，形状d×1。梯度为零的完整条件是`U(p-y)=0`，即残差落在U零空间；不能只写成p=y，因为U可能秩不足。梯度下降把中心向量拉向真实外部词，并减去当前预测分布的平均外部词方向，降低过度预测的关联。

(c) L2归一化去掉模长。若同方向词的模长编码情感强度，短语向量求和时这种信息有用：一维示例+2与-1相加为正，分别归一化后相加为0。若模长只反映无关频率，而分类只需方向，去掉它有利于避免无关权重。对α>0的`u_x=αu_y`，归一化后二者相同；零向量不适用此运算。

(d) `∂J/∂U=[∂J/∂u_1,...,∂J/∂u_M]`，形状d×M，亦即`v(p-y)ᵀ`。

(e) 对任意列u_w求导，真实列为`(p_o-1)v`，其他列为`p_w v`；统一形式`(p_w-y_w)v`。来自线性项的导数只出现在真实列，log-sum-exp项给每列p_w v。

(f) Leaky ReLU在x<0时导数α，x>0时导数1；不讨论0处不可微情形。

(g) 对`σ(x)=(1+exp(-x))^-1`用链式法则得到`exp(-x)/(1+exp(-x))²=σ(x)(1-σ(x))`。

## 2. 优化与正则化

Adam的动量m是梯度的指数滑动平均，历史方向会平滑单批次噪声和反向震荡，持续一致的方向得以保留。第二矩v记录逐参数梯度平方的平均；在其他条件相同时，较小v带来较大的有效学习率，较大v抑制剧烈更新，缓解不同参数尺度不一致。此处讨论题给简化式，实际PyTorch Adam另含偏差修正和数值稳定项。

Dropout的每个掩码位期望为1-p，因此为保持期望有`γ(1-p)h_i=h_i`，取γ=1/(1-p)，p<1。训练时随机丢弃激活，减少对固定搭配单元的依赖；评估时关闭随机丢弃，使用完整网络得到稳定预测。反向dropout已在训练时缩放，评估不应再乘一次同样的系数。

## 3. 依存分析

### (a) 给定句子的完整转移

下表缩写R=ROOT、p=presented、f=findings、c=conference；buffer为尚未处理的词。依存图中conference直接依附presented，at为conference的case依存，不能误把它附在findings下。

| 步 | 操作 | 操作后stack | 操作后buffer | 新依存 |
| --- | --- | --- | --- | --- |
| 0 | 初始化 | R | I p my f at the NLP c | — |
| 1 | S | R I | p my f at the NLP c | — |
| 2 | S | R I p | my f at the NLP c | — |
| 3 | LA | R p | my f at the NLP c | p→I |
| 4 | S | R p my | f at the NLP c | — |
| 5 | S | R p my f | at the NLP c | — |
| 6 | LA | R p f | at the NLP c | f→my |
| 7 | RA | R p | at the NLP c | p→f |
| 8 | S | R p at | the NLP c | — |
| 9 | S | R p at the | NLP c | — |
| 10 | S | R p at the NLP | c | — |
| 11 | S | R p at the NLP c | 空 | — |
| 12 | LA | R p at the c | 空 | c→NLP |
| 13 | LA | R p at c | 空 | c→the |
| 14 | LA | R p c | 空 | c→at |
| 15 | RA | R p | 空 | p→c |
| 16 | RA | R | 空 | R→p |

(b) 长度n需要2n步：每个词恰好SHIFT一次，并恰好被一次弧操作移出stack；ROOT不被移除。

(c)(d) 实现独立buffer拷贝、SHIFT/LEFT-ARC/RIGHT-ARC及共享partial-parse对象的批处理，保持输入句子和输出顺序。空输入/空句子直接返回空依存；只从unfinished列表移除已完成对象，不销毁原始partial-parses记录。

(e)(i) 令z_i=Σ_j x_j W_ji+b_i，则`∂h_i/∂x_j=1[z_i>0] W_ji`，不讨论z_i=0。

(e)(ii) softmax Jacobian为`∂p_j/∂l_i=p_j(δ_ji-p_i)`，代入交叉熵链式法则得`∂CE/∂l_i=p_i-y_i`，真实类为p_i-1、其余为p_i。

(e)(iii) 直接使用可训练的embedding参数与两组权重/偏置，手工索引并拼接36个50维特征，再执行1800→200 ReLU→dropout→3 logits。权重Xavier uniform、偏置uniform；不使用nn.Linear/nn.Embedding，CrossEntropyLoss接收未softmax的logits。Adam学习率0.0005，batch1024，完整10epochs，按最高dev UAS保存检查点，然后仅用该检查点评估test。

## 远程验收方案与当前状态

PyTorch2.1.2 CPU、Python3.11、NumPy1.26.4；固定随机种子20261125及哈希种子0。数据与原始utils哈希清单为a2-source.sha256，官方现有测试保持原样。新增验证脚本仅存集中`ai_agent_scripts/py/verify_cs224n_parser.py`，固定来源b81b3fcb1b01e89f1b8abd8d2f5e627989801c59，经带来源/哈希的本任务ZIP发布，不暴露整个私有脚本库。

Actions首先运行官方转移/批处理与模型sanity checks，再验证手工lookup的重复索引梯度、独立线性参考前向、全部参数梯度、dropout模式切换、错误softmax受控负例和小型优化。完整训练不带-d，保留十轮loss/dev UAS、最佳模型、测试依存和精确提交/run；最后从原始test.conll独立重算UAS，逐句检查单父、无自环和无环树。依据手册目标检查末轮训练loss<0.08、最佳dev UAS>0.87，test报告实际值，不预设成绩。

当前尚无本分支训练结果，best dev/test UAS均待真实Actions产出，不把AST检查、toy训练或旧GPT2项目结果当作本A2通过。本地未运行训练、模型测试或课程实验；仅源码/语法/哈希检查。自学说明与代码披露AI辅助，不冒充课堂独立提交或私有评分结果。
