# MGSTCN
(1) Experimental environment:
        scipy>=0.19.0
        numpy>=1.12.1
        pandas>=0.19.2
        tensorflow>=1.3.0

(2) DataSets
Download pems-bay data from 
export fileid=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq
export filename=data/pems_bay/pems-bay.h5
wget -O $filename 'https://drive.google.com/uc?export=download&id='$fileid

The Didi dataset involves a protection protocol, which needs to be downloaded from the official website or the official email address of the link provided in the paper.


(3)For the three variants:
The MGTCN requires replacing the files in the model in the MGSTCN directory and modifying the corresponding references in train.py.
The GSTCN needs to replace the input dataset in MGSTCN with the original non-layered data and the corresponding adjacency matrix.
The GTCN needs to replace the input dataset in MGTCN with the original non-layered data and the corresponding adjacency matrix.
