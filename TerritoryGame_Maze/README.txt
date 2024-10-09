***This program consists of three main folders: `TerritoryGame_Maze`, `Territory_complex`, 
and the `Territory_complex_random` folder within `Territory_complex`. 
The three main functions are located within these respective folders.***

The video of the results from the third training session is located in the `Territory_complex_random` folder.

This project uses a virtual environment. To set up the required libraries, use the following dependencies:

1. **Python Version**: 3.8
2. **Main Libraries**:
   - TensorFlow 2.10.0
   - Gym 0.26.1
   - MoviePy 1.0.3
   - Keras 2.10.0
   - NumPy 1.24.4
   - Pandas 2.0.3

For the full list of dependencies, please refer to the provided `requirements.txt` file or `conda` environment file.

2.The operation of this program depends on the import of the following libraries.
import pygame
import sys
import random
import gym
from gym import spaces
import numpy as np
import os
import tensorflow 
import logging
import time
import scipy.spatial
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers

3.To run this code, you must modify the local paths to ensure consistency.
4.This code has already saved the results from the previous two training sessions. If you need to retrain the agents from those sessions, the saved models will be overwritten.
5.To run the third maze training, please locate the `main` function in the `Territory_complex_random` folder, modify the local path accordingly, and then execute the code.
6.**Note:** After running the code, it will automatically save a pygame video, which may cause a delay of about five minutes after the execution ends. If you do not need the video, please delete the code related to the frame storage list and video saving.
7.If you want to use TensorBoard to review the functions, you need to clear the data in the `logs-analysis` folder, leaving only the data from the current run. Otherwise, multiple functions may overlap. However, each data entry in this code is timestamped, so the data will not be confused.