import io

from PIL import Image
from graphviz import Digraph


def _trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root, format='svg', rankdir='LR', shapes_only=True):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)

    Note: save with: dot.render('graph', view=True)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = _trace(root)
    dot = Digraph(
        format=format,
        graph_attr={'rankdir': rankdir},
        # node_attr={'rankdir': 'TB'}
    )

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label = f"data {n.data.shape} | grad {n.grad.shape}" if shapes_only else f"data {n.data} | grad {n.grad}",
            shape='record'
        )
        if n._op:
            dot.node(
                name=str(id(n)) + n._op,
                label=n._op
            )
            dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def view_dot(dot):
    png_str = dot.pipe(format='png')  # Get the PNG data as bytes
    byte_io = io.BytesIO(png_str)
    img = Image.open(byte_io)
    img.show()
