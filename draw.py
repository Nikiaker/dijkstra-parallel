import subprocess
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import sys
 
def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]
 
seed = 0
random.seed(seed)
np.random.seed(seed)
 
vertecies = ""
a = ""
b = ""
if len(sys.argv) == 4:
    vertecies = sys.argv[1]
    a = sys.argv[2]
    b = sys.argv[3]
command = vertecies + " " + a + " " + b
 
proc = subprocess.Popen('graph.exe', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
out, err = proc.communicate(input = command.encode())
 
arr = str.split(out.decode())
lis = list(divide_chunks(arr, 4))
print(lis)
 
G = nx.Graph()
 
for x in lis:
    G.add_edge(x[0], x[1], weight = x[2], color=x[3])
 
pos = nx.spring_layout(G)
edge_labels = {(u,v):d["weight"] for u,v,d in G.edges(data = True)}
colors = [G[u][v]['color'] for u,v in G.edges()]
 
nx.draw(G, pos = pos, with_labels = True, node_color = "blue", edge_color=colors, font_color = "white")
nx.draw_networkx_edge_labels(G, pos = pos, edge_labels=edge_labels, label_pos=0.5, rotate = False, font_size=10)
plt.show()
