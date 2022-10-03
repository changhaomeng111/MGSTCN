# MGSTCN
(1) Experimental environment:
        scipy>=0.19.0
        numpy>=1.12.1
        pandas>=0.19.2
        tensorflow>=1.3.0



(2)For the three variants:
The MGTCN requires replacing the files in the model in the MGSTCN directory and modifying the corresponding references in train.py.
The GSTCN needs to replace the input dataset in MGSTCN with the original non-layered data and the corresponding adjacency matrix.
The GTCN needs to replace the input dataset in MGTCN with the original non-layered data and the corresponding adjacency matrix.
