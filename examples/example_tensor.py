from catenia import Tensor
from catenia.utils.utils import draw_dot, view_dot


a = Tensor(-4.0)
b = Tensor(2.0)
c = a + b * 4
d = c / 2

y = d ** 2
y.backward()

print(y)

dot = draw_dot(y, shapes_only=False)
view_dot(dot)
# dot.render(directory='output', view=True)
# dot.view()
