**RUN SCRIPT INSTRUCTIONS**

**1. Install libraries**

In Powershell, run following commands to install external libraries.

- Numpy

[python -m pip install \--user numpy]{.mark}

- Pandas

[python -m pip install \--user pandas]{.mark}

- Scipy

[python -m pip install \--user scipy]{.mark}

- Scikit-learn

[python -m pip install \--user scikit-learn]{.mark}

- Matplotlib

[python -m pip install \--user matplotlib]{.mark}

- Pytorch

[python -m pip install \--user torch]{.mark}

- Plotly

[python -m pip install \--user plotly]{.mark}

**2. Set up Input files**

Put CPT profile .csv files into folder \\CPT Profiles. The header line
of each .csv file must precisely match the following:

[Depth (ft), Cone resistance (tsf), Sleeve friction (tsf), Pore pressure
u2 (psi)]{.mark}

**3. Set up configuration files**

- cptProfileOffsetRows.txt

Input integers separated by comma. The number of integers must match the
CPT profile file (.csv) count. For example, with three .csv files,

[0, 0, 0]{.mark}

- hyperParameters.txt

Update the integers on the right side of equal signs:

[numberGlobalClasses = 3]{.mark}

[numberGlobalLayers = 4]{.mark}

[embed_dim = 64]{.mark}

[n_epochs = 1500]{.mark}

- plotFlags.txt

Update Boolean (True/False) on the right side of equal signs:

[plotImportFlag = True]{.mark}

[plotTsneFlag = True]{.mark}

[plotReconstructedFlag = True]{.mark}

[plotLayeringFlag = True]{.mark}

**4. Execution**

In folder \\Code, Run following command in Powershell or equivalent:

[python
Soil_Layering_by_Cone_Penetration_Test_Data_Based_on_Autoencoder.py]{.mark}

To complete execution, close all figures.

**5. Tune hyperparameters**

Apply the Elbow Method on the following plots to tune hyperparameter
*numberGlobalClasses:*

- Within-cluster sum of squares plot

- Silhouette plot

Apply the Elbow Method on the following plots to tune hyperparameter
*numberGlobalLayers:*

- Accuracy vs Number of max_leaf_nodes plot
