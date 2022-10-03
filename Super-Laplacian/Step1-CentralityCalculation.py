import numpy as np
import networkx as nx
import pandas as pd

for branche in [1,2,3]:
    if branche==0:
        adj = np.load("PEMS325adj.npy")
        print(adj.shape)

        for aaa in range (adj.shape[0]):
            for bbb in  range(adj.shape[1]):
                if  adj[aaa,bbb]>0.001:
                    adj[aaa,bbb]=1.

        print(adj.shape)

        with open('edge.txt', 'w') as fileobject:
            for aaa in range(0,adj.shape[0]):
                for bbb in range(0, adj.shape[1]):
                    if adj[aaa,bbb]>0.000001:
                        fileobject.write(str(aaa)+" "+str(bbb)+" "+str(adj[aaa,bbb])+"\n")

    if branche==1:
        G = nx.Graph()
        with open("edge.txt") as text:
            for line in text:
                vertices = line.strip().split(" ")
                source = int(vertices[0])
                target = int(vertices[1])
                G.add_edge(source, target)
        print(G.number_of_nodes())

        b = nx.betweenness_centrality(G)

        listbet=[0 for x in range(0,325)]
        yuanshinodelist=[x for x in range(0,325)]
        print(G.nodes())
        for v in G.nodes():
            listbet[v]=b[v]

        index_list = sorted(range(len(listbet)), key=lambda i: listbet[i])[-64:]  # sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:2]
        pro_list = np.array(listbet)[index_list]
        print("indexs={}".format(index_list))
        print("pro values={}".format(pro_list))
    if branche == 2:
        indexs=[25, 297, 202, 204, 223, 34, 215, 168, 90, 2, 246, 141, 236, 165, 166, 54, 60, 237, 118, 124, 289, 254, 186, 273, 48, 312, 189, 224, 187, 222, 132, 150, 181, 321, 323, 247, 283, 72, 226, 227, 4, 280, 269, 230, 188, 27, 142, 257, 19, 36, 156, 148, 147, 128, 121, 20, 160, 31, 149, 102, 218, 219, 26, 217]

        indexs2 = indexs
        G = nx.Graph()
        with open("edge.txt") as text:
            for line in text:
                vertices = line.strip().split(" ")
                source = int(vertices[0])
                target = int(vertices[1])
                G.add_edge(source, target)

        with open("layer.txt",'w') as onelever:
            for node1 in G.nodes():
                if node1 not in indexs:
                    shortpathlength=999
                    leibie=indexs[-1]
                    for bbb in indexs:
                        try:
                            p1=nx.shortest_path_length(G, source=node1, target=bbb)
                        except nx.NetworkXNoPath:
                            tmp=1
                            continue
                        if p1<shortpathlength:
                            shortpathlength=p1
                            leibie=bbb
                    onelever.write(str(node1)+","+str(leibie)+"\n")


