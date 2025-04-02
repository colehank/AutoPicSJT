import matplotlib.pyplot as plt
import networkx as nx

def draw_G(
    G, 
    ax=None,
    figsize=(6, 6), 
    title='',
    node_fontsize=10, 
    edge_fontsize=10, 
    node_size=400,
    dpi=300,
    show_node_value=True, 
    show_edge_value=True,
    object_node_color='skyblue', 
    attribute_node_color='pink',
    relation_edge_color='black', 
    attribute_edge_color='lightgray',
    layout='spring'
):
    """
    绘制场景图 G, 并提供以下可配置功能: 
      - 是否显示节点的 value(例如 "private email")
      - 是否显示关系边的 value(注意只有 relation_edge 有值)
      - 可设置不同类型节点(object_node 与 attribute_node)的颜色
      - 可设置不同类型边(relation_edge 与 attribute_edge)的颜色
      - 可选择绘制图的布局(如 'spring', 'circular', 'kamada_kawai', 'shell')
    
    :param G: nx.DiGraph 对象
    :param figsize: 图形大小
    :param title: 图标题
    :param node_fontsize: 节点标签字体大小
    :param edge_fontsize: 边标签字体大小
    :param node_size: 节点大小
    :param show_node_value: 是否显示节点的 value(True: 显示 value; False: 不显示)
    :param show_edge_value: 是否显示关系边的 value(仅对 relation_edge 有效)
    :param object_node_color: object_node 节点的颜色
    :param attribute_node_color: attribute_node 节点的颜色
    :param relation_edge_color: relation_edge 边的颜色
    :param attribute_edge_color: attribute_edge 边的颜色
    :param layout: 图的布局方式, 可选 'spring', 'circular', 'kamada_kawai', 'shell'
    :return: matplotlib.figure.Figure 对象
    """
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
    relation_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get("type") == "relation_edge"]
    attribute_edges = [(u, v) for u, v, data in G.edges(data=True) if data.get("type") == "attribute_edge"]

    # 绘制关系边(object->object), 采用指定颜色
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=relation_edges,
        ax=ax,
        arrowstyle='-|>',
        arrowsize=20,
        width=2,
        edge_color=relation_edge_color,
        min_source_margin=15,
        min_target_margin=15,
        connectionstyle='arc3, rad=.1'
    )

    # 绘制属性边(attribute->object), 采用指定颜色
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=attribute_edges,
        ax=ax,
        arrowstyle='-|>',
        arrowsize=20,
        width=2,
        edge_color=attribute_edge_color,
        min_source_margin=15,
        min_target_margin=15
    )

    # 根据节点类型提取不同的节点
    object_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "object_node"]
    attribute_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "attribute_node"]

    # 绘制 object 节点(客体节点)——蓝色
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=object_nodes,
        ax=ax,
        node_size=node_size,
        node_color=object_node_color,
        node_shape='o',
        edgecolors='gray'
    )

    # 绘制 attribute 节点——粉色
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=attribute_nodes,
        ax=ax,
        node_size=node_size,
        node_color=attribute_node_color,
        node_shape='o',
        edgecolors='gray'
    )

    # 根据参数决定是否绘制节点标签, 显示节点的 value
    if show_node_value:
        labels = {n: d.get("value", n) for n, d in G.nodes(data=True)}
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            ax=ax,
            font_size=node_fontsize,
        )

    # 根据参数决定是否绘制边标签, 仅对 relation_edge 绘制边的 value
    if show_edge_value:
        edge_labels = {
            (u, v): d.get("value", "")
            for u, v, d in G.edges(data=True)
            if d.get("type") == "relation_edge"
        }
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=edge_labels,
            ax=ax,
            font_size=edge_fontsize,
            label_pos=0.5,
            bbox=dict(facecolor='white', edgecolor='none', pad=0.5)
        )
        
    x_vals, y_vals = zip(*pos.values())
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_padding = (x_max - x_min) * 0.2
    y_padding = (y_max - y_min) * 0.2
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    
    if returnfig:
        return fig
    
def plot_vng_sg(Gs):
    plt.close('all')
    vng_map = {
        'E': 'Establisher',
        'I': 'Initial',
        'Pr': 'Prolongation',
        'P': 'Peak'
    }
    fig_width = 4*len(Gs)
    fig_height = 4
    fig_size = (fig_width, fig_height)
    fig, axs = plt.subplots(1, len(Gs), figsize=(fig_size), dpi=300)

    for i, (vng, G) in enumerate(Gs.items()):
        draw_G(G, ax=axs[i], title=vng_map[vng], node_fontsize=8, edge_fontsize=8)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        if i == len(Gs) - 1: 
            axs[i].spines['right'].set_visible(False)

    plt.tight_layout()
    return fig