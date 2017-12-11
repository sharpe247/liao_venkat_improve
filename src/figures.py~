from config.settings import *
from graphviz import Digraph

def make_graph(graph, comment):
	node_name_dict ={1:'WRKY22', 2:'MPK4', 3:'FLS2', 4:'MKK2', 5:'PR1', 6:'MEK1', 7:'MKK4', 8:'MKK3', 9:'WRKY25'}
	dot =Digraph(comment)
	for i, j in  node_name_dict.items():
		dot.node(str(i), str(j))
	edges =[str(k)+ str(i) for i, j in graph.items() for k in j]
	dot.edges(edges)
	dot.render('MAPK.jpg', view=True)

if __name__ == '__main__':
	#the graphs are hard coded at the moment
	graph_1 ={1: set([]), 2: set([9]), 3: set([2]), 4: set([6]), 5: set([2]), 6: set([1]), 7: set([1]), 8: set([]), 9: set([])}
	comment_1='Inferred BN with max number of parents =1'
	graph_2 ={1: set([]), 2: set([8, 7]), 3: set([6]), 4: set([]), 5: set([4]), 6: set([2, 5]), 7: set([1, 5]), 8: set([]), 9: set([8, 7])}
	comment_2='Inferred BN with max number of parents =2'
	graph_3 ={1: set([3]), 2: set([1, 7, 9]), 3: set([]), 4: set([6]), 5: set([6]), 6: set([3]), 7: set([3, 5]), 8: set([4, 5]), 9: set([7])}
	comment_3='Inferred BN with max number of parents =3'
	#make_graph(graph_1, comment_1)
	#make_graph(graph_2, comment_2)
	make_graph(graph_3, comment_3)

