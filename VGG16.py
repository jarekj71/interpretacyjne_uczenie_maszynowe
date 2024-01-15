# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import json
import numpy as np
import torch
from torchvision import models
import shap
from matplotlib import pyplot as plt
#%%
mean = [0.485, 0.456, 0.406] # średnie dla kanałów w zakresie 0,1
std = [0.229, 0.224, 0.225]


def normalize(image):
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    #reshapeing
    return torch.tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()
#%%

model = models.vgg16(pretrained=True).eval()
X, y = shap.datasets.imagenet50()
X /= 255
to_explain = X[[11, 15]]


url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
fname = shap.datasets.cache(url)
with open(fname) as f:
    class_names = json.load(f)
#%%

e = shap.GradientExplainer((model, model.features[7]), normalize(X))
shap_values, indexes = e.shap_values(
    normalize(to_explain), ranked_outputs=2, nsamples=50
)

index_names = np.vectorize(lambda x: class_names[str(x)][1])(indexes)

# plot the explanations
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

shap.image_plot(shap_values, to_explain, index_names)


