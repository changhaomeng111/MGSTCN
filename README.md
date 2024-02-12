# MGSTCN
(1) Experimental Environment:

        scipy>=0.19.0
        numpy>=1.12.1
        pandas>=0.19.2
        tensorflow>=1.3.0
        networkx>=2.5.1
  
The PEMS04 and PEMS08 datasets can be run directly.
The PEMS-BAY and Didi datasets need to be run by placing ".\data\PEMS-BAY or Didi\train.py" in the data directory in the same level as the 'MGSTCN/model'.
Note that the PEMS-Bay and Didi datasets need to replace utils.py in the lib directory.

(2) DataSets

"PEMS04 and PEMS08": 
See the data folder in the MGSTCN directory.

The npz files required for PEMS-BAY and Didi are provided below and can be added directly to the "MGSTCN\data\PEMS-BAY or \Didi" directory to run.

"PEMS-BAY":

The raw data is obtained as follows
Download pems-bay data from 
export fileid=1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq
export filename=data/pems_bay/pems-bay.h5
wget -O $filename 'https://drive.google.com/uc?export=download&id='$fileid

"Didi Dataset":

The Didi raw dataset involves a protection protocol, which needs to be downloaded from the official website or the official email address of the link provided in the paper.

It is worth noting that for the Didi-Xi'an dataset, the experimental parameters in this paper are different from those set in the original paper of HGCN. The batch size used in the original HGCN was 64, but the batch size used in 7 baselines of this paper was 32. There are differences in the preprocessing process based on the Didi-Xi'an dataset (this part of HGCN code has not been made public), so the experimental results are different from those in the original paper.

(3) For Three Variants:

The MGTCN, GSTCN and GTCN running require the data directory to be added to the root directory of the corresponding variant model.

(4) Super-Laplacian

The different Super Laplacian matrix are generated by adjusting the input parameters. For example, l=8,16,32,64 where l indicates the number of layers.
