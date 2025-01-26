## 执行主文件
```python
python main.py rules/topv2_weather.cfg
```

包含两个参数，数据集名称（dataset_name) + 数据数量（num）。由于异步比较草率，实际生成的数量可能会略大于num（但
生成nl时候有校验操作，最终数量又有可能不够num）。
由于像parse、translate_format(虽然该叫convert哈哈)等函数与具体的数据集挂钩，所以需要传递dataset_name的名称（目前我
只实现了topv2需要的。其他名称需要另行实现。

## 主函数的结构分为如下几步：
```python
derivation_texts: list[str] = generate_expressions(n=num)  # 通过CFG的规则生成初步的逻辑表达式
al_exps: list[Assertion | Formula] = parse_derivations(derivation_texts)  # 将CFG的规则转成断言逻辑的语法
# 存储为断言（Assertion）或带有逻辑连接符的事实序列（Formula，如A1 ^ A2）
gen_labels: list[str] = translate_format(al_exps)  # 统一从断言逻辑的格式展开成topv2数据集需要的格式
# 这里就已经是label了（对于绝大多数数据集而言）
dataset: CustomDataset = generate_nl(gen_labels)  # 根据逻辑表达式的label生成自然语言，并组织成数据集
# CustomDataset就是个继承torch.utils.data.Dataset的抽象类，没什么特殊的。然后其中的每个Example的格式都是
# self.input和self.output
```

对于部分数据集，如topv2，需要额外的后处理以修复标签。这些处理可以在`postprocess`中实现。所以你在主函数里能看到`fix_labels_topv2`
函数的出现。


## CFG的格式
绝大多数都是自定义的（毕竟parse等函数都是自己写）。就提醒下几处：
1. 非终止符用$打头，纯大写或下划线。终止符最好用纯小写
2. 需要调取随机值生成器的，用`_generate`作为后缀。如`date_time_generate`会调取概念`date time`的生成器。
没有自定义生成器时，将默认调取gpt进行（注意temperature不要太高，现在是0.3）。
3. 置空默认用null（虽然我记得这个没有嵌入代码中，是自定义的，但不妨统一，以防万一）。
4. 其余应该都是自定义的，不是固定在代码里的，包括`intent:`, `date_time:`, `[]`等各种前后缀。
5. `_generate`生成回来的值，其空格会被替换为`*space`，以免和CFG规则中的空格产生歧义。你在parse等函数的时候要注意转回去