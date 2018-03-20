# MIL
## data
saved_models   模型保存目录
searchqa       数据目录
quasart        数据目录

## code
preprocess.py   数据分词等函数
data/           数据处理
    data.py      数据类 词典，数据集，采样类
    vector.py    将样本数值化函数，和批数据处理函数
    utils.py     加载数据，构建词典，计数，计时等工具类
model.py       模型 wraper
milmodel.py    模型
layers         自定义层
main.py          程序入口
run.sh          数据预处理，训练，预测的全过程
