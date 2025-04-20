# AI+心理学问题求助

## 研究目的
我的目的是基于一个已有的人格情景判断测验（测量大五人格），通过大语言模型设计文生图prompt，并基于文生图模型转为图片的人格情景判断测验。这里的图片是一种图片序列，视觉叙事。这过程还需要充分发挥图片相比文字之外的优势。

## 介绍（当前已有的工作）
1. step1: 通过实体与关系提取，构建出了每道题目的情景题干的事件图谱。

这一步的目的是对情景进行结构化的表达，为了更好地进行信息提取与增加。
这里的图谱有两种节点，属性节点（att）；客体节点（obj）。两种边，属性边（|），没有value，连接属性节点与客体节点；关系边(-)，有value，连接客体节点与客体节点。

*Input*: str: situation（原始题目）
*Output*: DiGraph_original

2. step2: 以5种连接模式检索知识，交由大语言模型识别是否为特质激活线索。

这一步的目的是识别出题目激活特质的关键信息。
这里的连接模式有5种，[att|obj, obj-obj, att|obj-obj, obj-att|obj, att|obj-att|obj]。
通过搜索图谱，可以获得相关的知识，进而交由大语言模型进行激活特质相关的知识的识别。

*Input*: DiGraph_original
*Output*: list[dict[connection_pattern: content]]

3. step3: 以视觉叙事语法（VNG）为原则，对情景的事件图谱进行分镜设计

这一步的目的是将原始情景转化为若干分镜，VNG的原则大概是E（背景）-I（初始动作）-P（高潮）。
这一步通过大语言模型，把第一步的事件图谱分为了三个分镜图谱，以支持进一步的图像序列制作。

*Input*: DiGraph_original
*Output*: dict[vng_type: DiGraph_vng_type]

## 示例
1. step1
- input
'Ye is sitting in the middle of a crowded movie theater. Shortly after the film has started, Ye realize that Ye made a mistake in the cinema and ended up in the wrong film'

- output
DiGraph with 7 nodes and 6 edges
[38;5;4mℹ  Nodes:[0m
object_1 {'type': 'object_node', 'value': 'Ye'}
object_2 {'type': 'object_node', 'value': 'movie theater'}
object_3 {'type': 'object_node', 'value': 'film'}
attribute|2|1 {'type': 'attribute_node', 'value': 'crowded'}
attribute|1|1 {'type': 'attribute_node', 'value': 'sitting in middle'}
attribute|3|1 {'type': 'attribute_node', 'value': 'wrong film'}
attribute|3|2 {'type': 'attribute_node', 'value': 'has started'}
[38;5;4mℹ  Edges:[0m
object_1 -> object_2 {'type': 'relation_edge', 'value': 'inside'}
object_1 -> object_3 {'type': 'relation_edge', 'value': 'watching'}
attribute|2|1 -> object_2 {'type': 'attribute_edge'}
attribute|1|1 -> object_1 {'type': 'attribute_edge'}
attribute|3|1 -> object_3 {'type': 'attribute_edge'}
attribute|3|2 -> object_3 {'type': 'attribute_edge'}

2. step2
- input
DiGraph with 7 nodes and 6 edges
[38;5;4mℹ  Nodes:[0m
object_1 {'type': 'object_node', 'value': 'Ye'}
object_2 {'type': 'object_node', 'value': 'movie theater'}
object_3 {'type': 'object_node', 'value': 'film'}
attribute|2|1 {'type': 'attribute_node', 'value': 'crowded'}
attribute|1|1 {'type': 'attribute_node', 'value': 'sitting in middle'}
attribute|3|1 {'type': 'attribute_node', 'value': 'wrong film'}
attribute|3|2 {'type': 'attribute_node', 'value': 'has started'}
[38;5;4mℹ  Edges:[0m
object_1 -> object_2 {'type': 'relation_edge', 'value': 'inside'}
object_1 -> object_3 {'type': 'relation_edge', 'value': 'watching'}
attribute|2|1 -> object_2 {'type': 'attribute_edge'}
attribute|1|1 -> object_1 {'type': 'attribute_edge'}
attribute|3|1 -> object_3 {'type': 'attribute_edge'}
attribute|3|2 -> object_3 {'type': 'attribute_edge'}

- output
[{'type': 'att|obj-att|obj',
  'content': ['sitting in middle', 'Ye', 'watching', 'wrong film', 'film']},
 {'type': 'obj-att|obj',
  'content': ['Ye', 'inside', 'crowded', 'movie theater']}]

3. step3
- input
DiGraph with 7 nodes and 6 edges
[38;5;4mℹ  Nodes:[0m
object_1 {'type': 'object_node', 'value': 'Ye'}
object_2 {'type': 'object_node', 'value': 'movie theater'}
object_3 {'type': 'object_node', 'value': 'film'}
attribute|2|1 {'type': 'attribute_node', 'value': 'crowded'}
attribute|1|1 {'type': 'attribute_node', 'value': 'sitting in middle'}
attribute|3|1 {'type': 'attribute_node', 'value': 'wrong film'}
attribute|3|2 {'type': 'attribute_node', 'value': 'has started'}
[38;5;4mℹ  Edges:[0m
object_1 -> object_2 {'type': 'relation_edge', 'value': 'inside'}
object_1 -> object_3 {'type': 'relation_edge', 'value': 'watching'}
attribute|2|1 -> object_2 {'type': 'attribute_edge'}
attribute|1|1 -> object_1 {'type': 'attribute_edge'}
attribute|3|1 -> object_3 {'type': 'attribute_edge'}
attribute|3|2 -> object_3 {'type': 'attribute_edge'}

- output
VNG-E:
DiGraph with 4 nodes and 3 edges
[38;5;4mℹ  Nodes:[0m
object_1 {'type': 'object_node', 'value': 'Ye'}
object_2 {'type': 'object_node', 'value': 'movie theater'}
attribute|1|1 {'type': 'attribute_node', 'value': 'sitting in middle'}
attribute|2|1 {'type': 'attribute_node', 'value': 'crowded'}
[38;5;4mℹ  Edges:[0m
object_1 -> object_2 {'type': 'relation_edge', 'value': 'inside'}
attribute|1|1 -> object_1 {'type': 'attribute_edge'}
attribute|2|1 -> object_2 {'type': 'attribute_edge'}


VNG-I:
DiGraph with 3 nodes and 2 edges
[38;5;4mℹ  Nodes:[0m
object_1 {'type': 'object_node', 'value': 'Ye'}
object_3 {'type': 'object_node', 'value': 'film'}
attribute|3|2 {'type': 'attribute_node', 'value': 'has started'}
[38;5;4mℹ  Edges:[0m
object_1 -> object_3 {'type': 'relation_edge', 'value': 'watching'}
attribute|3|2 -> object_3 {'type': 'attribute_edge'}


VNG-P:
DiGraph with 3 nodes and 2 edges
[38;5;4mℹ  Nodes:[0m
object_1 {'type': 'object_node', 'value': 'Ye'}
object_3 {'type': 'object_node', 'value': 'film'}
attribute|3|1 {'type': 'attribute_node', 'value': 'wrong film'}
[38;5;4mℹ  Edges:[0m
object_1 -> object_3 {'type': 'relation_edge', 'value': 'watching'}
attribute|3|1 -> object_3 {'type': 'attribute_edge'}
