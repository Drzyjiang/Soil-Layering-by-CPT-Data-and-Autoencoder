**RUN SCRIPT INSTRUCTIONS**

**1. Install libraries**

In Powershell, run following commands to install external libraries.

- Numpy

python -m pip install \--user numpy

- Pandas

python -m pip install \--user pandas

- Scipy

python -m pip install \--user scipy

- Scikit-learn

python -m pip install \--user scikit-learn

- Matplotlib

python -m pip install \--user matplotlib

- Pytorch

python -m pip install \--user torch

- Plotly

python -m pip install \--user plotly

**2. Set up Input files**

Put CPT profile .csv files into folder \\CPT Profiles. The header line
of each .csv file must precisely match the following:

Depth (ft), Cone resistance (tsf), Sleeve friction (tsf), Pore pressure
u2 (psi)

**3. Set up configuration files**

- cptProfileOffsetRows.txt

Input integers separated by comma. The number of integers must match the
CPT profile file (.csv) count. For example, with three .csv files,

0, 0, 0

- hyperParameters.txt

Update the integers on the right side of equal signs:

numberGlobalClasses = 3

numberGlobalLayers = 4

embed_dim = 64

n_epochs = 1500

- plotFlags.txt

Update Boolean (True/False) on the right side of equal signs:

plotImportFlag = True

plotTsneFlag = True

plotReconstructedFlag = True

plotLayeringFlag = True

**4. Execution**

In folder \\Code, Run following command in Powershell or equivalent:

python Soil_Layering_by_Cone_Penetration_Test_Data_and_Autoencoder.py

To complete execution, close all figures.

**5. Tune hyperparameters**

Apply the Elbow Method on the following plots to tune hyperparameter
*numberGlobalClasses:*

- Within-cluster sum of squares plot

- Silhouette plot

Apply the Elbow Method on the following plots to tune hyperparameter
*numberGlobalLayers:*

- Accuracy vs Number of max_leaf_nodes plot
