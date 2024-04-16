
import os
import torch
from torchvision import datasets
import torchvision
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
from PIL import ImageFile
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True
plt.ion()
torch  