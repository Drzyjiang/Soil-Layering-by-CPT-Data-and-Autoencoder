**EXECUTION INSTRUCTIONS**

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

**2. Set up Input files**

Put CPT profile .csv files into folder \\CPT Profiles. The header line
of each .csv file must precisely match the following:

Depth (ft), Cone resistance (tsf), Sleeve friction (tsf), Pore pressure
u2 (psi)

**3. Set up configuration files**

- cptProfileParameters.txt

Update the integers on the right side of equal signs:

- cptProfileOffetRows: how to rows to offset for each CPT profile. List
  of integers, separated by coma. Positive value denotes shifting
  profile to lower elevation; negative value denotes shifting profile to
  higher elevation. Length of list must equal .csv file numbers.
  Example,

> cptProfileOffetRows = 0, 0, 0

- minLayerThickness: minimum thickness of a layer to be considered.
  Float. Use zero as no limitations. Example,

> minLayerThickness = 0.0

- hyperParameters.txt

Update the integers on the right side of equal signs:

- numberGlobalClasses: how many distinctive layers to be considered.
  Integer. Example,

numberGlobalClasses = 3

- numberGlobalLayers: how many layers to split for each profile.
  Integer. Example,

> numberGlobalLayers = 4

- embed_dim: latent vector dimensions. Integer, default is 32. Example,

> embed_dim = 32

- n_epochs: number of times of training. Integer, default is 1500.
  Example,

n_epochs = 1500

- n_heads: number of heads in Transformer Encoder. Integer, default
  is 4. Example,

> n_heads = 4

- n_layers: number of Transformer layers. Integer, default is 2.
  Example,

> n_layers = 2

- lossEarlyTerminationCriterion: early termination criterion for
  training in terms of total loss. Float, default is 0.000001. Example,

> lossEarlyTerminationCriterion = 0.000001

- plotFlags.txt

Update Boolean (True/False) on the right side of equal signs:

- plotImportFlag: flag of whether to plot raw data. Boolean, default is
  False. Example,

> plotImportFlag = False

- plotTsneFlag: flag of whether to plot t-SNE plots of latent vectors.
  Boolean, default is True. Example,

> plotTsneFlag = True

- plotReconstructedFlag: flag of whether to plot comparison between raw
  and reconstructed data. Boolean, default is True. Example,

plotReconstructedFlag = True

- plotLayeringFlag: flag of whether to plot final layering results.
  Boolean, default is True. Example,

plotLayeringFlag = True

- plotTsneGroupingFlag: flag of whether to plot grouping results on
  t-SNE plots. Boolean, default is True. Example,

> plotTsneGroupingFlag = True

**4. Execution**

In folder \\Code, Run following command in Powershell or equivalent:

python Soil_Layering_by_Cone_Penetration_Test_Data_and_Autoencoder.py

To complete execution, close all figures.

**5. Tune hyperparameters**

Apply the Elbow Method on the following plots to tune hyperparameter
*numberGlobalClasses:*

- Within-cluster sum of squares plot

- t-SNE plot of classifications on all latent vectors

Apply the Elbow Method on the following plot to tune hyperparameter
*numberGlobalLayers:*

- Accuracy vs Number of max_leaf_nodes plot
