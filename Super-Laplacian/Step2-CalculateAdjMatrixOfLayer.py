import networkx as nx
import numpy as np

file1=open("./layer.txt")
k=64
indexs = [25, 297, 202, 204, 223, 34, 215, 168, 90, 2, 246, 141, 236, 165, 166, 54, 60, 237, 118, 124, 289, 254, 186,
          273, 48, 312, 189, 224, 187, 222, 132, 150, 181, 321, 323, 247, 283, 72, 226, 227, 4, 280, 269, 230, 188, 27,
          142, 257, 19, 36, 156, 148, 147, 128, 121, 20, 160, 31, 149, 102, 218, 219, 26, 217]

for aaa in indexs:
    globals()["list"+str(aaa)]=[]

while 1:
    line1=file1.readline()
    if not line1:
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




def FindA(inputlist1,inputlist2,i,j):
    G = nx.Graph()
    with open("edge.txt") as text:
        for line in text:
            vertices = line.strip().split(" ")
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)

    A1length = len(inputlist1)
    A1 = np.zeros((A1length, A1length))
    for nodeindex in range(A1length):
        for nodeindex2 in range(A1length):
            if nodeindex != nodeindex2:
                try:
                    p1 = nx.shortest_path_length(G, source=inputlist1[nodeindex], target=inputlist1[nodeindex2])
                    A1[nodeindex, nodeindex2] = p1
                except nx.NetworkXNoPath:
                    A1[nodeindex, nodeindex2] = 0

    A2length=len(inputlist2)
    A2 = np.zeros((A2length,A2length))
    for nodeindex in range(A2length):
        for nodeindex2 in range(A2length):
            if nodeindex != nodeindex2:
                try:
                    p1 = nx.shortest_path_length(G, source=inputlist2[nodeindex], target=inputlist2[nodeindex2])
                    A2[nodeindex,nodeindex2]=p1
                except nx.NetworkXNoPath:
                    A2[nodeindex, nodeindex2] =0

    A1toA2 = np.zeros((A1length, A2length))
    for nodeindex in range(A1length):
        for nodeindex2 in range(A2length):
                try:
                    p1 = nx.shortest_path_length(G, source=inputlist1[nodeindex], target=inputlist2[-1])
                    A1toA2[nodeindex, nodeindex2] = p1
                except nx.NetworkXNoPath:
                    A1toA2[nodeindex, nodeindex2] = 0


    A2toA1= np.zeros((A2length, A1length))
    for nodeindex in range(A2length):
        for nodeindex2 in range(A1length):
                try:
                    p1 = nx.shortest_path_length(G, source=inputlist2[nodeindex], target=inputlist1[-1])
                    A2toA1[nodeindex, nodeindex2] = p1
                except nx.NetworkXNoPath:
                    A2toA1[nodeindex, nodeindex2] = 0

    np.save("./adj/A"+str(i), A1)
    np.save("./adj/A" + str(j), A2)
    np.save("./adj/A" + str(i)+"-"+"A" + str(j), A1toA2)
    np.save("./adj/A" + str(j)+"-"+"A" + str(i), A2toA1)

for i in range(0,k):
    for j in range(i+1,k):
            FindA(countlist[i],countlist[j],i,j)