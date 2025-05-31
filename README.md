# T-SE
This repository contains the code to build T-SE (Thresholded-Squeeze-and-Excitation) Networks, as well as some scripts to analyze the results. The implementation is derived from the following repository: https://github.com/moskomule/senet.pytorch.git.

Main.py
-------
Python script that takes as argument and ID (from 0 to 149). Trains and test the SE-Resnet20 architecture with 30 thresholds linearly distributed between 0 and 0.95. Each threshold is used five times to be able to average the results and get a standard deviation for each threshold. The script outputs 150 files describing the energy consumption of the model for each setup, and 150 files describing accuracies, channels switched off and avoided computations for each threshold.

Main baseline.py
----------------
Python script that takes as argument and ID (from 0 to 4).
