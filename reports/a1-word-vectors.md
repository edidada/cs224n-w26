# CS224n Winter2026 A1：Exploring Word Vectors（代码及数据探针待远端验证）

来源为[官方A1发行包](https://web.stanford.edu/class/cs224n/assignments_w26/a1.zip)，92860字节，SHA256 `adbae7acc45bdcdd4bf21c35f25116ca1fc9698140cdcacec458295f9dffe87e`。原始导入cf16c607；当前仓库是官方ZIP镜像，不是独立官方GitHub starter的fork。独立分支`edidada/a1-word-vectors`，独立YAML为a1-word-vectors.yml。

## 实现

在原notebook填写四个函数：按字典序去重；逐文档统计上下文窗口内的有向共现计数；使用n_iter10的TruncatedSVD返回UΣ；仅绘制指定词并标注。共现不跨文档边界，不计当前位置本身，但同一词出现在不同位置的共现会累计到对角线；两种方向均计数，矩阵对称。空语料、零窗口、重复词和边界窗口由独立枚举oracle核对，不通过过滤空token或词频改变原150篇预处理定义。

原四个sanity/test单元源码逐字保持一致，原始README/env及参考图片另固定Git blob SHA。新增验证脚本只存集中`ai_agent_scripts/py/verify_cs224n_vectors.py`，固定来源6b4d45d4fc9af242709b81ba7b951fde48fe3573；通过只含本任务源码及provenance的工具ZIP供Actions调用。

## 数据与真实输出计划

完整执行原notebook全部代码单元，保留GloVe wiki-gigaword **400000×200** 全词表，不能以少量候选词或50维模型替代。原150篇IMDB采样不改变；首运行在Actions解析stanfordnlp/imdb实际commit并写入执行副本，再按该commit加载。GloVe原gzip核对gensim官方元数据MD5，并记录实际SHA256；没有提前编造未知资产哈希。

工件应含完整执行notebook、sanity/co-occurrence/GloVe三幅实际图、固定数据revision与模型大小、原始近邻/距离/类比探针，以及独立检查JSON。验证附录只添加到执行副本，明确不属于原始提交题目；原import列表不新增库。环境固定Python3.11、NumPy1.26.4、SciPy1.13.1、Gensim4.3.3、datasets2.18.0等，模型与notebook实验只在Actions执行。

## 分析题状态

2.4类比表达式为woman + grandfather - man，近邻不是严格的家谱逻辑；2.5(a)亲属主题的混合解释，以及2.9语料偏见和反事实数据增强的概念说明已填写。其余依赖实际图像或数值的解释仍待审核：

- 1.5：按实际图与官方参考，找两组聚类及两例预期未聚类；不能凭词义猜图上位置。
- 2.1：比较共现与GloVe图，若图有差异按原题使用官方参考图说明。
- 2.2：已列多义词探针，实际top10是否包含两个词义需人工核查，不能把任何近邻列表都当作满足。
- 2.3：从预先定义的同义/反义词组中选择实际满足反义词距离更近的一组，记录精确距离后解释语料分布。
- 2.5(b)/2.6(b)：必须实际验证正确及错误类比的top1，不能只写预期答案。
- 2.6(a)：观察原手/手套/脚实例的实际错误排名再解释。
- 2.7/2.8：实际观察给定职业及独立职业的性别偏移近邻，再讨论统计偏见；结果不表示群体固有能力或合适角色。

因此即使本次执行CI成功，也仅证明代码/输出链路与数值探针成立；仍需实际看图、复核词义及补齐书面分析，再验证最终HEAD，不能登记整个A1完成。本轮未在本地运行notebook、模型或课程测试，自学代码披露AI辅助，不冒充课堂独立提交。
