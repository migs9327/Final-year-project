# Supporting Information
This repository contains the supporting information for the project "Application of Apple M1 Silicon to Graph Neural Networks for Molecular Property Prediction".

Details of the virtual environment configurations used to achieve GPU-accelerated training using the PiNN and Spektral libraries on the M1 Mac are given in the two "requirements.txt" files.

The folder "PiNN_scripts" contains the modified network and model classes used in combination with the "Training_PiNet_TF2" notebook for migrating PiNet's training to TF2. 

Other folders contain the trained models for all epoch scales mentioned in the project report, alongside GPU metrics and their predictions on the test data set.

The "Data Analysis" notebook contains the code used for generating all plots contained in the project report.

The "PiNet_tffunction_decorator_error" notebook reproduces the error encountered when the exact implementation of PiNN's potential model was recreated in a custom training loop. Lastly "learning_a_LJ_potential" was an attempt at learning a toy problem, but was not reported ultimately due to low prediction accuracy.
