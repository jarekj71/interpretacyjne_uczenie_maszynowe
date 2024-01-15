#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 09:46:01 2024

@author: jarekj
"""

#%%
import json

import numpy as np
import torch
import torchvision

import shap

#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torchvision.models.mobilenet_v2(pretrained=True, progress=False)
model.to(device)
model.eval()
X, y = shap.datasets.imagenet50()
# y nie jest wykorzystywany

#%%

# Getting ImageNet 1000 class names
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]
print("Number of ImageNet classes:", len(class_names))
# print("Class names:", class_names)

#%%

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

'''
n number
| c color
| h height rows
V w width cols

rotacja c na koniec
n
h
w
c

'''

#funkcja sprawdza czy mamy 4 wymiary (wiele obrazów) czy trzy wimiary (jeden obraz)

def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
    elif x.dim() == 3:
        x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
    return x


def nchw_to_nhwc(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 4:
        x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
    elif x.dim() == 3:
        x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
    return x

#%%% Pipes 
# Rotacja jest potrzebna aby zastosować transformację obrazu w sposób zwektoryzowany
transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Lambda(lambda x: x * (1 / 255)),
    torchvision.transforms.Normalize(mean=mean, std=std),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

# transformacja odwrotna służy do wykonania denormalizacji
inv_transform = [
    torchvision.transforms.Lambda(nhwc_to_nchw),
    torchvision.transforms.Normalize(
        mean=(-1 * np.array(mean) / np.array(std)).tolist(),
        std=(1 / np.array(std)).tolist(),
    ),
    torchvision.transforms.Lambda(nchw_to_nhwc),
]

#połączenie w pipe
transform = torchvision.transforms.Compose(transform)
inv_transform = torchvision.transforms.Compose(inv_transform)


#%%
#predykcja obrazu wymaga zamiany w tensor i wysłanie do device

def predict(img: np.ndarray) -> torch.Tensor:
    img = nhwc_to_nchw(torch.Tensor(img))
    img = img.to(device)
    output = model(img)
    return output

#%%
# Check that transformations work correctly
Xtr = transform(torch.Tensor(X))
out = predict(Xtr[7:8])
classes = torch.argmax(out, axis=1).cpu().numpy()
print(f"Classes: {classes}: {np.array(class_names)[classes]}")


#%% explanation

topk = 4
batch_size = 50
n_evals = 10000

masker_blur = shap.maskers.Image("blur(64,64)", Xtr[0].shape)
# funkcja predict, masker output list aklas
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

#%%

shap_values = explainer(
    Xtr[7:8],
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:3],
)

#%%
shap_values.data = inv_transform(shap_values.data).cpu().numpy()[0]
shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]

#%%
shap.image_plot(
    shap_values=shap_values.values,
    pixel_values=shap_values.data,
    labels=shap_values.output_names,
    true_labels=[class_names[643]],
)