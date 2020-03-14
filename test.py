from graphviz import Digraph

dot = Digraph(comment='The Round Table')
dot.node('A', 'king')
dot.view()
