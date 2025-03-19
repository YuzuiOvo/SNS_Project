# Gold-Prediction
Data文件夹代表数据 Scripts文件夹代表每一个功能部分的代码
step1：Scripts里的gold prediction 是获取过去一年的黄金价格和美元指数，并且存为CSV，相对应的是Data里的dxy idex和 gold prices
step2：Scripts里的data analysis 合并并检查黄金与美元指数数据的质量,看一下数据有没有问题 有没有缺失 等等，对应Data里的 merged data 这个没什么用
step3：Scripts里的data visualization 是数据可视化 运行这个脚本可以生成一个折线图，看一下我的数据有没有问题 没有对应的Data文件
step4：Scripts feature engineering是创建移动平均线、滞后等特征，生成最终数据集，对应的是final dataset
step5：split dataset 我把所有数据划分了测试集和训练集，80%是训练集，20%用来测试一下的
所以整个顺序是 获取数据----检查数据质量----数据可视化（把数据放到折线图中看看有没有骤减骤升等）---特征训练----划分数据集训练集
下一步应该是根据数据，选择合适模型开始训练了，对80%的训练集训练，最后用20%的测试集去验证。

