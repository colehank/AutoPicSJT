from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import colormaps as cm

from ..utils.graph_utils import map_knowledge
plt.rcParams['font.family'] = 'Comic Sans MS'
plt.rcParams['font.family'] = 'Times New Roman'


def draw_G(
    G,
    ax=None,
    figsize=(6, 6),
    title='',
    node_fontsize=10,
    edge_fontsize=10,
    node_size=800,
    dpi=300,
    show_node_value=True,
    show_edge_value=True,
    object_node_color='skyblue',
    attribute_node_color='pink',
    relation_edge_color='black',
    attribute_edge_color='lightgray',
    object_node_edge_color='white',
    attribute_node_edge_color='white',
    colors=None,
    layout='spring',
    attribute_node_shape='o',
    object_node_shape='o',
):
    """
    绘制场景图 G, 并提供以下可配置功能:
      - 是否显示节点的 value (例如 "private email")
      - 是否显示关系边的 value (注意只有 relation_edge 有值)
      - 可设置不同类型节点(object_node 与 attribute_node)的颜色
      - 可设置不同类型边(relation_edge 与 attribute_edge)的颜色
      - 可设置节点本身的边框颜色
      - 当传入 colors 参数时，启动指定节点、边及标签的颜色绘制。colors 格式为:
          {
              'node': { node_id: (fill_color, edge_color), ... },
              'edge': { edge_id: edge_color, ... },
              'label': {
                  node_or_edge_id: label_color, ...   # 节点id或边 (u, v)
              }
          }
      - 可选择绘制图的布局(如 'spring', 'circular', 'kamada_kawai', 'shell')

    :param G: nx.DiGraph 对象
    :param figsize: 图形大小
    :param title: 图标题
    :param node_fontsize: 节点标签字体大小
    :param edge_fontsize: 边标签字体大小
    :param node_size: 节点大小
    :param dpi: 图像 dpi
    :param show_node_value: 是否显示节点的 value (True: 显示; False: 不显示)
    :param show_edge_value: 是否显示关系边的 value (仅对 relation_edge 有效)
    :param object_node_color: object_node 节点的默认填充颜色
    :param attribute_node_color: attribute_node 节点的默认填充颜色
    :param relation_edge_color: relation_edge 边的默认颜色
    :param attribute_edge_color: attribute_edge 边的默认颜色
    :param object_node_edge_color: object_node 节点的默认边框颜色
    :param attribute_node_edge_color: attribute_node 节点的默认边框颜色
    :param colors: 指定节点、边及标签颜色的字典, 格式见上文说明
    :param layout: 图的布局方式, 可选 'spring', 'circular', 'kamada_kawai', 'shell'
    :return: matplotlib.figure.Figure 对象
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    # 根据参数选择布局
    plt.close('all')
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=3)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    elif layout == 'shell':
        pos = nx.shell_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=0.8)

    returnfig = False
    if ax is None:
        returnfig = True
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # 分别提取不同类型的边
    relation_edges = [
        (u, v) for u, v, data in G.edges(data=True) if data.get('type') == 'relation_edge'
    ]
    attribute_edges = [
        (u, v) for u, v, data in G.edges(data=True) if data.get('type') == 'attribute_edge'
    ]

    # 处理 relation_edge 的颜色
    relation_edge_colors = []
    for u, v in relation_edges:
        data = G.get_edge_data(u, v)
        edge_key = data.get('id', (u, v))
        if colors and 'edge' in colors:
            edge_color = colors['edge'].get(edge_key, relation_edge_color)
        else:
            edge_color = relation_edge_color
        relation_edge_colors.append(edge_color)

    # 绘制关系边 (object->object)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=relation_edges,
        ax=ax,
        arrowstyle='-|>',
        arrowsize=20,
        width=2,
        edge_color=relation_edge_colors,
        min_source_margin=15,
        min_target_margin=15,
        connectionstyle='arc3, rad=.1',
    )

    # 处理 attribute_edge 的颜色
    attribute_edge_colors = []
    for u, v in attribute_edges:
        data = G.get_edge_data(u, v)
        edge_key = data.get('id', (u, v))
        if colors and 'edge' in colors:
            edge_color = colors['edge'].get(edge_key, attribute_edge_color)
        else:
            edge_color = attribute_edge_color
        attribute_edge_colors.append(edge_color)

    # 绘制属性边 (attribute->object)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=attribute_edges,
        ax=ax,
        arrowstyle='-|>',
        arrowsize=20,
        width=2,
        edge_color=attribute_edge_colors,
        min_source_margin=15,
        min_target_margin=15,
    )

    # 根据节点类型提取不同的节点
    object_nodes = [
        n for n, d in G.nodes(data=True) if d.get('type') == 'object_node'
    ]
    attribute_nodes = [
        n for n, d in G.nodes(data=True) if d.get('type') == 'attribute_node'
    ]

    # 绘制 object 节点 (客体节点)
    if colors and 'node' in colors:
        object_fill_colors = []
        object_border_colors = []
        for node in object_nodes:
            if node in colors['node']:
                fill, border = colors['node'][node]
            else:
                fill, border = object_node_color, object_node_edge_color
            object_fill_colors.append(fill)
            object_border_colors.append(border)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=object_nodes,
            ax=ax,
            node_size=node_size,
            node_color=object_fill_colors,
            node_shape=object_node_shape,
            edgecolors=object_border_colors,
            linewidths=1.5,
        )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=object_nodes,
            ax=ax,
            node_size=node_size,
            node_color=object_node_color,
            node_shape=object_node_shape,
            edgecolors=object_node_edge_color,
            linewidths=1.5,
        )

    # 绘制 attribute 节点
    if colors and 'node' in colors:
        attribute_fill_colors = []
        attribute_border_colors = []
        for node in attribute_nodes:
            if node in colors['node']:
                fill, border = colors['node'][node]
            else:
                fill, border = attribute_node_color, attribute_node_edge_color
            attribute_fill_colors.append(fill)
            attribute_border_colors.append(border)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=attribute_nodes,
            ax=ax,
            node_size=node_size,
            node_color=attribute_fill_colors,
            node_shape=attribute_node_shape,
            edgecolors=attribute_border_colors,
            linewidths=1.5,
        )
    else:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=attribute_nodes,
            ax=ax,
            node_size=node_size,
            node_color=attribute_node_color,
            node_shape=attribute_node_shape,
            edgecolors=attribute_node_edge_color,
            linewidths=1.5,
        )

    # 绘制节点标签
    if show_node_value:
        labels = {n: d.get('value', n) for n, d in G.nodes(data=True)}
        node_texts = nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            ax=ax,
            font_size=node_fontsize,
            font_color='black',  # 默认颜色
        )
        # 如果指定了 label 的颜色，则更新节点标签颜色
        if colors and 'label' in colors:
            for n, text in node_texts.items():
                label_color = colors['label'].get(n, 'black')
                text.set_color(label_color)

    # 绘制边标签 (仅对 relation_edge 绘制边的 value)
    if show_edge_value:
        edge_labels = {
            (u, v): d.get('value', '')
            for u, v, d in G.edges(data=True)
            if d.get('type') == 'relation_edge'
        }
        edge_texts = nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=edge_fontsize,
            label_pos=0.5,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.5),
        )
        # 如果指定了 label 的颜色，则更新边标签颜色
        if colors and 'label' in colors:
            for edge, text in edge_texts.items():
                label_color = colors['label'].get(edge, 'black')
                text.set_color(label_color)

    # 调整坐标轴范围
    x_vals, y_vals = zip(*pos.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_padding = (x_max - x_min) * 0.2
    y_padding = (y_max - y_min) * 0.2
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.close('all')

    if returnfig:
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.axis('off')
        fig.tight_layout()
        plt.close('all')
        return fig



def draw_Gs(Gs):
    plt.close('all')
    vng_map = {
        'E': 'Establisher',
        'I': 'Initial',
        'Pr': 'Prolongation',
        'P': 'Peak',
    }
    if len(Gs) == 1:
        fig_width = 4
        fig_height = 4
        fig_size = (fig_width, fig_height)
        fig, axs = plt.subplots(1, 1, figsize=(fig_size), dpi=300)
        vng = list(Gs.keys())[0]
        draw_G(
            Gs[vng], ax=axs, title=vng_map[vng],
            node_fontsize=8, edge_fontsize=8,
        )
        axs.spines['top'].set_visible(False)
        axs.spines['bottom'].set_visible(False)
        axs.spines['left'].set_visible(False)
        axs.spines['right'].set_visible(False)
        return fig
    else:

        fig_width = 4*len(Gs)
        fig_height = 4
        fig_size = (fig_width, fig_height)
        fig, axs = plt.subplots(1, len(Gs), figsize=(fig_size), dpi=300)

        for i, (vng, G) in enumerate(Gs.items()):
            draw_G(
                G, ax=axs[i], title=vng_map[vng],
                node_fontsize=8, edge_fontsize=8,
            )
            axs[i].spines['top'].set_visible(False)
            axs[i].spines['bottom'].set_visible(False)
            axs[i].spines['left'].set_visible(False)
            if i == len(Gs) - 1:
                axs[i].spines['right'].set_visible(False)

        plt.tight_layout()
        return fig

def draw_G_cue_highlight(G, cues, cmap='hsv', title=None, **kwargs):
    """
    绘制场景图G，并高亮显示指定的cues

    参数:
    G: nx.DiGraph 对象，场景图
    cues: 需要高亮显示的cues列表
    cmap: 颜色映射名称，默认为'hsv'
    title: 图标题，默认为'SituCues'
    **kwargs: 传递给draw_G的其他参数

    返回:
    matplotlib.figure.Figure 对象
    """

    def get_colors_by_cues(G, cues, cmap='gist_ncar'):
        def custom_cmap(value):
            colormap = cm.get_cmap(cmap)
            return colormap(value)

        colors_cue = [custom_cmap(i / len(cues)) for i in range(len(cues))]
        gh_id = [map_knowledge(G, cue['content'], cue['type']) for cue in cues]
        colors = {'edge': {}, 'label': {}}
        for i, cue in enumerate(gh_id):
            color = colors_cue[i]
            # 检查cue是否为None
            if cue is None:
                continue
            for j in cue:
                if isinstance(j, tuple):  # 处理关系边
                    colors['edge'][j] = color
                    colors['label'][j] = color
                if 'attribute' in j:  # 处理属性边
                    colors['edge'][(j, f'object_{j.split("|")[-2]}')] = color
                    colors['label'][(j, f'object_{j.split("|")[-2]}')] = color
        return colors

    # 获取高亮颜色
    colors = get_colors_by_cues(G, cues, cmap=cmap)

    # 设置默认参数
    default_kwargs = {
        'attribute_edge_color': 'black',
        'attribute_node_shape': '^',
        'object_node_color': 'lightgray',
        'attribute_node_color': 'lightgray',
        'title': title,
    }

    # 更新参数
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    # 调用draw_G函数绘制图
    return draw_G(G, colors=colors, **kwargs)
