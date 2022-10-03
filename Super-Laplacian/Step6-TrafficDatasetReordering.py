import  numpy as np

k=64
indexs = [25, 297, 202, 204, 223, 34, 215, 168, 90, 2, 246, 141, 236, 165, 166, 54, 60, 237, 118, 124, 289, 254, 186,
          273, 48, 312, 189, 224, 187, 222, 132, 150, 181, 321, 323, 247, 283, 72, 226, 227, 4, 280, 269, 230, 188, 27,
          142, 257, 19, 36, 156, 148, 147, 128, 121, 20, 160, 31, 149, 102, 218, 219, 26, 217]
orderlist=[]
for lable in indexs:
    orderlist.append(int(lable))
    file1 = open("../layer.txt")
    while 1:
        line1=file1.readline()
        if not line1:
            break
        node,cluster=line1.split(",")
        if str.strip(cluster)==str(lable):
            orderlist.append(int(node))
    file1.close()

sz_flow = np.load("./PEMSBAY.npy")
data= np.mat(sz_flow)

data2=np.zeros((data.shape[0],len(orderlist)))
for aaa in range(0,len(orderlist)):
    for bbb in range(0,data.shape[0]):
        data2[bbb,aaa]=data[bbb,orderlist[aaa]]
print(data2.shape)
np.save("PEMSBAYOrder_"+str(k),data2)



