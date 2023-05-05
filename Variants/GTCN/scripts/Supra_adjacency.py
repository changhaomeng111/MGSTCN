import networkx as nx
import numpy as np
import  os


class Supra_adjacency(object):
    def __init__(self,adj_npy=""):
        self.adj_npy=adj_npy

def construct_supra_adjacency(dir,l=16,adj_npy=""):
        G = nx.Graph()
        adj = adj_npy
        for i in range(0,adj.shape[0]):
            for j in range(0, adj.shape[1]):
                if (adj[i,j])>=0.00000:
                    link_weight = float(adj[i,j])
                    G.add_weighted_edges_from([(i, j, link_weight)])
                    G.add_weighted_edges_from([(j, i, link_weight)])

        size=G.number_of_nodes()
        b = nx.betweenness_centrality(G,weight=True)
        listbet=[0 for x in range(0,size)]
        for v in G.nodes():
            listbet[v]=b[v]

        index_list = sorted(range(len(listbet)), key=lambda i: listbet[i])[-1*l:]  # sorted(range(len(p)), key=lambda i: p[i], reverse=True)[:2]
        indexs=index_list
        with open(dir+"layer.txt",'w') as LayersFile:
            for node1 in G.nodes():
                if node1 not in indexs:
                    shortpathlength=999999
                    layerid=indexs[0]
                    for bbb in indexs:
                        try:
                            p1=nx.shortest_path_length(G, source=node1, target=bbb,weight=True)
                        except nx.NetworkXNoPath:
                            continue
                        if p1<shortpathlength:
                            shortpathlength=p1
                            layerid=bbb
                    LayersFile.write(str(node1)+","+str(layerid)+"\n")

        file1=open(dir+"layer.txt")
        k=len(indexs)
        for layerid in indexs:
            globals()["list"+str(layerid)]=[]

        while 1:
            line1=file1.readline()
            if not line1:
                file1.close()
                break
            node,cluster=line1.split(",")

            i=0
            while i<k:
                if str.strip(cluster)==str(indexs[i]):
                    globals()["list" + str(indexs[i])].append(int(node))
                i+=1

        j=0
        while j<k:
            globals()["list" + str(indexs[j])].append(indexs[j])
            j+=1

        countlist=[]
        for aaa in indexs:
            countlist.append(globals()["list"+str(aaa)])

        def FindA(inputlist1, inputlist2, i, j):

            A1length = len(inputlist1)
            A1 = np.zeros((A1length, A1length))
            for nodeindex in range(A1length):

                for nodeindex2 in range(A1length):
                    if nodeindex != nodeindex2:
                        try:
                            p1 = nx.shortest_path_length(G, source=inputlist1[nodeindex], target=inputlist1[nodeindex2])
                            A1[nodeindex, nodeindex2] = 1/p1 if p1>0.0000001  else 0
                        except nx.NetworkXNoPath:
                            A1[nodeindex, nodeindex2] = 0
            A2length = len(inputlist2)
            A2 = np.zeros((A2length, A2length))
            for nodeindex in range(A2length):
                for nodeindex2 in range(A2length):
                    if nodeindex != nodeindex2:
                        try:
                            p1 = nx.shortest_path_length(G, source=inputlist2[nodeindex], target=inputlist2[nodeindex2])
                            A2[nodeindex, nodeindex2] = 1/p1 if p1>0.0000001  else 0
                        except nx.NetworkXNoPath:
                            A2[nodeindex, nodeindex2] = 0
            A1toA2 = np.zeros((A1length, A2length))
            try:
                p1 = nx.shortest_path_length(G, source=inputlist1[-1], target=inputlist2[-1])
            except nx.NetworkXNoPath:
                p1 = 0

            for nodeindex in range(A1length):
                A1toA2[nodeindex, A2length - 1] = 1/p1 if p1>0.0000001  else 0

            A2toA1 = np.zeros((A2length, A1length))
            try:
                p1 = nx.shortest_path_length(G, source=inputlist2[-1], target=inputlist1[-1])
            except nx.NetworkXNoPath:
                p1 = 0

            for nodeindex in range(A2length):
                A2toA1[nodeindex, A1length - 1] = 1/p1 if p1>0.0000001  else 0

            if not os.path.exists(dir+"adj"):
                os.makedirs(dir+"adj")

            # distances = A1[~np.isinf(A1)].flatten()
            # std = distances.std() if distances.std()>0.0000001  else 1
            # A1 = np.exp(-np.square(A1 / std))

            # distances = A2[~np.isinf(A2)].flatten()
            # std = distances.std() if distances.std()>0.0000001  else 1
            # A2 = np.exp(-np.square(A2 / std))
            # A2[A2 < 0.1] = 0
            #
            # distances = A1toA2[~np.isinf(A1toA2)].flatten()
            # std = distances.std() if distances.std()>0.0000001  else 1
            # A1toA2 = np.exp(-np.square(A1toA2 / std))
            # A1toA2[A1toA2 < 0.1] = 0
            #
            # distances = A2toA1[~np.isinf(A2toA1)].flatten()
            # std = distances.std() if distances.std()>0.0000001  else 1
            # A2toA1 = np.exp(-np.square(A2toA1 / std))
            # A2toA1[A2toA1 < 0.1] = 0

            np.save(dir+"adj/A" + str(i), A1)
            np.save(dir+"adj/A" + str(j), A2)
            np.save(dir+"adj/A" + str(i) + "-" + "A" + str(j), A1toA2)
            np.save(dir+"adj/A" + str(j) + "-" + "A" + str(i), A2toA1)

        os.remove(dir+"layer.txt")
        nodeorder_list=[]
        for i in range(0,k):
            nodeorder_list+=countlist[i]
            for j in range(i+1,k):
                    FindA(countlist[i],countlist[j],i,j)
        return nodeorder_list
