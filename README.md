# T-SE
This repository contains the code to build T-SE (Thresholded-Squeeze-and-Excitation) Networks, as well as some scripts to analyze the results. The implementation is derived from the following repository: https://github.com/moskomule/senet.pytorch.git.

Main.py
-------
Python script that takes as argument and ID (from 0 to 149). Trains and test the SE-Resnet20 architecture with 30 thresholds linearly distributed between 0 and 0.95. Each threshold is used five times to be able to average the results and get a standard deviation for each threshold. The script outputs 150 files describing the energy consumption of the model for each setup, and 150 files describing accuracies, channels switched off and avoided computations for each threshold.

Main baseline.py
----------------
Python script that takes as argument and ID (from 0 to 4). Trains and tests the baseline Resnet-20 architecture 5 times to be able to average the results and get a standard deviation. The script outputs 5 files describing the energy consumption of the model and 5 files describing accuracies for each iteration of the experiment.

Results analysis.py
-------------------
Python scripts that analyzes the output files from Main.py. The results must be previously gathered in a zipped folder containing two folders. The first folder should be named "Emissions" and contain the 150 output files related to energy consumption. The second folder should be named "Output" and should contain the 150 output text files with results related to accuracies, switched off channels and avoided computations. The scripts exports the results to an Excel file and generated several graphs for analysis.

Results analysis baseline.py
----------------------------
Python scripts that analyzes the output files from Main baseline.py. The results must be previously gathered in a zipped folder containing two folders. The first folder should be named "Emissions" and contain the 5 output files related to energy consumption. The second folder should be named "Output" and should contain the 5 output text files with results related to accuracies. The scripts exports the results to an Excel file.

requirements.txt
----------------
File containing the requirements for all experiments and scripts from this repository.


