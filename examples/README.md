# SituationProcessor 使用说明

`SituationProcessor` 是一个用于处理单个情景的类，它整合了事件图提取、线索提取和故事板提取功能。

## 功能特点

- 提取事件图（Event Graph）
- 提取线索（Cues）
- 提取故事板（Storyboard）

## 使用方法

```python
from src.pipeline import SituationProcessor

# 准备情景文本
situation = """
You are working on a group project with three other students. One team member
consistently fails to complete their assigned tasks on time, causing delays
in the project's progress. The deadline is approaching, and the team's grade
depends on everyone's contribution.
"""

# 初始化 SituationProcessor
processor = SituationProcessor(situ=situation, trait='C')

# 提取事件图
G = processor.extract_event_graph()

# 提取线索（可以复用已有的事件图）
cues = processor.extract_cues(G)

# 提取故事板（可以复用已有的事件图）
vng_Gs = processor.extract_storyboard(G)
```

## 参数说明

### 初始化参数

- `situ`: 情景文本字符串
- `trait`: 性格特质，如 'O', 'C', 'E', 'A', 'N'，默认为 'O'
- `model`: 使用的LLM模型，默认为 'claude-3-5-sonnet-latest'

### 方法参数

- `extract_event_graph()`: 无参数，返回 networkx.DiGraph 格式的事件图
- `extract_cues(G=None)`: 接受可选的事件图参数，如果不提供则会自动提取，返回线索列表
- `extract_storyboard(G=None)`: 接受可选的事件图参数，如果不提供则会自动提取，返回故事板字典

## 返回值说明

- `extract_event_graph()`: 返回 networkx.DiGraph 格式的事件图
- `extract_cues()`: 返回线索列表，每个线索是一个字典
- `extract_storyboard()`: 返回故事板字典，键为故事板名称，值为 networkx.DiGraph 格式的图

## 示例

请参考 `situation_processor_example.py` 文件，了解如何使用 `SituationProcessor` 类。
