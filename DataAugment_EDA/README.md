本程序参考代码：[EDA_NLP_for_Chinese](https://github.com/zhanlaoban/eda_nlp_for_Chinese)

## 使用方式
本代码暂时不考虑上传到pip上，本代码只需要进入chinese-text-eda目录下后，执行：
```shell script
python setup.py install
```
即可安装使用。
也可以将代码复制到项目录下使用。

## 中文文本分类数据增强
在中文文本分类过程中需要使用数据增强的方式能够在数据量少的情况起到一定效果。

### 固定格式数据增强
该种方式提供只需要将待增强的数据处理成如下格式，然后在书写一个py脚本，在命令行运行即可。
```text
label   sentence
```
其中数据的标签在前，文本在后，标签和文本之间使用\t（tab）分割。

使用方式如下，python 文件为example.py
```python
from ChineseTextEDA.eda import SimpleEDAEnhance

sed = SimpleEDAEnhance()
sed.simple_eda_enhance()

```
然后在控制台中输入相关参数即可。控制台输入样例如下：
```shell script
python example.py --input train.txt --output eda_train.txt --num_aug 2 --alpha 0.2
```
相关参数说明：
- --input: 原始数据的输入文件, must
- --output: 增强数据后的输出文件, optional，如果未填写，会在input目录下生成一个eda开头的结果
- --num_aug: 一条数据增强几条数据，optional, default 9
- --alpha: 每条语句中将会被改变的单词数占比, optional, default 0.1

更简单的方式是：随机创建一个python脚本如test.py，然后导入example即可，如下：
```python
from ChineseTextEDA import example
```
然后在命令行输入指令即可。

### 自定义格式数据增强
有时数据格式相对复杂，这时需要我们将增强的方法嵌入到数据处理的程序中，这时可以参考如下方法, 案例代码如下，这也是实例化EDA这个类
```python
eda = EDA(num_aug=num_aug,stop_words=stop_words, stop_words_type="hit")
enhance_result = []
with open(input, 'r', encoding='utf8') as reader:
    for index, line in enumerate(reader):
        line = line.strip()
        if not line:
            continue
        parts = line.split('\t')
        label = parts[0]
        sentence = parts[1]
        aug_sentences = eda.eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha,
                                p_rd=alpha)
        for aug_sentence in aug_sentences:
            enhance_result.append(f"{label}\t{aug_sentence}")
```
其中，EDA初始化类的参数如下：
- num_aug: 一条数据增强到多少条，optional, default 9
- stop_words: 增强过程使用的停用词, optional, default use hit提供的停用词
- stop_words_type: 停用词类型，optional, select scope: ["hit", "cn", "baidu", "scu"] 

EDA类中的数据增强方法的分数如下：
- sentence: must, 待增强的语句;
- alpha_sr: default=0.1,近义词替换词语的比例
- alpha_ri: default=0.1,随机插入词语个数占语句词语数据的比例
- alpha_rs: default=0.1,随机交换词语个数占语句词语数据的比例
- p_rd: default=0.1，随机删除词语个数占语句词语数据的比例 

一个简单的测试如下：
```python
from ChineseTextEDA.eda import EDA

eda = EDA()
res = eda.eda("我们就像蒲公英，我也祈祷着能和你飞去同一片土地")
print(res)
```
结果如下：
```text
['我们 就 像 蒲公英 ， 我 也 天主 着 能 和 你 飞去 同 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 着 能 和 你 飞去 同 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 和 能 着 你 飞去 同 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 着 聚花 能 和 你 飞去 同 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 着 能 和 飞去 你 同 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 能 和 你 飞去 同 一片 土地', '我们 就 像 ， 我 也 祈祷 着 能 和 你 飞去 同 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 着 能 和 你 飞去 同 假如 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 着 能 和 你 直奔 同 一片 土地', '我们 就 像 蒲公英 ， 我 也 祈祷 着 能 和 你 飞去 同 一片 土地']
```