from __future__ import annotations

import networkx as nx
from wasabi import msg


def print_G(G):
    """
    Print the graph structure in a readable format.
    :param G: The graph object containing nodes and edges.
    """
    print(G)
    msg.info('\nNodes:')
    for node, data in G.nodes(data=True):
        print(node, data)

    msg.info('\nEdges:')
    for u, v, data in G.edges(data=True):
        print(u, '->', v, data)


def build_G(scene_graph_dict):
    """
    根据给定的字典数据构建 NetworkX DiGraph

    :param scene_graph_dict: 字典格式的数据，包含 "nodes" 和 "edges" 两个键，
                             "nodes" 为 [node_id, node_attrs] 的列表，
                             "edges" 为 [source, target, edge_attrs] 的列表。
    :return: 构建好的 nx.DiGraph 对象
    """
    G = nx.DiGraph()

    # 添加节点
    for node_entry in scene_graph_dict.get('nodes', []):
        node_id, node_attrs = node_entry
        G.add_node(node_id, **node_attrs)

    # 添加边
    for edge_entry in scene_graph_dict.get('edges', []):
        src, dst, edge_attrs = edge_entry
        G.add_edge(src, dst, **edge_attrs)

    return G


def get_node_id(G, value):
    """
    在图 G 中查找具有指定 value 的节点, 返回该节点的 id.

    :param G: nx.DiGraph 对象
    :param target_value: 要查找的节点 value
    :return: 匹配节点的 id, 如果找不到则返回 None
    """
    found = False
    for node, data in G.nodes(data=True):
        val = data.get('value')
        if val == value:
            return node
    if not found:
        msg.warn(f'No node found with value: {value}')
    return None


def get_edge_id(G, value):
    """
    在图 G 中查找具有指定 value 的边, 返回该边的 id.

    :param G: nx.DiGraph 对象
    :param target_value: 要查找的边 value
    :return: 匹配边的 id (如果未设置则返回 (u, v)), 如果找不到则返回 None
    """
    found = False
    for u, v, data in G.edges(data=True):
        if data.get('value') == value:
            return data.get('id', (u, v))
    if not found:
        msg.warn(f'No edge found with value: {value}')
    return None
