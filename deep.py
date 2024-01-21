#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:15:09 2024

@author: jarekj
"""

#%%
import json
import numpy as np
import torch
import torchvision
from torchvision import models
import shap
from matplotlib import pyplot as plt




#%%
#zbiór danych załączony do shap losowy zbiór obrazów z kolekcji imageNet, który został 
#użyty to przeszukania zasobów Google w celu znalezienia podobnych obrazów z prawem wykorzstania

X, y = shap.datasets.imagenet50()
# y nie jest wykorzystywany



# Nazwy klas
url = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
with open(shap.datasets.cache(url)) as file:
    class_names = [v[1] for v in json.load(file).values()]

# test czy się udało
print("Number of ImageNet classes:", len(class_names))
#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%%
mean = [0.485, 0.456, 0.406] # średnie dla kanałów seici pretrenowanej w zakresie 0,1
std = [0.229, 0.224, 0.225]


def standarize(image): # organizacja wewnętrzna tensora
    if image.max() > 1:
        image /= 255
    image = (image - mean) / std
    #reshapeing
    return torch.Tensor(image.swapaxes(-1, 1).swapaxes(2, 3)).float()

def standarize2(image):
    if image.max() > 1:
        image /= 255
    tensor = torch.Tensor(X)
    tensor = tensor.permute(0,3,1,2)
    n = torchvision.transforms.Normalize(mean,std)
    tensor = n.forward(tensor)
    return tensor
    
#%%
sX = standarize(X)
sX2 = standarize2(X)

#%%
# weights to pretrained
model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1', progress=False).eval()
model.to(device) # co to jest?

#%%

e = shap.GradientExplainer(model, sX)
shap_values, indexes = e.shap_values(sX[[11, 15]], ranked_outputs=2, nsamples=50)


index_names = np.vectorize(lambda x: class_names[x])(indexes)
# zmiana z tensora na image
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
shap.image_plot(shap_values, X[[11, 15]], index_names)
#%% inny model, konkretna features
model = models.vgg16(weights='VGG16_Weights.IMAGENET1K_V1').eval()
model.to(device)

#%%
e = shap.GradientExplainer((model,model.features[7]), sX)
shap_values, indexes = e.shap_values(sX[[11, 15]], ranked_outputs=2, nsamples=50)
index_names = np.vectorize(lambda x: class_names[x])(indexes)
shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
shap.image_plot(shap_values, X[[11, 15]], index_names)

#%%
###############################################################################

model = models.mobilenet_v2(weights='MobileNet_V2_Weights.IMAGENET1K_V1', progress=False).eval()
model.to(device) # co to jest?

#%%

n_out = 3
batch_size = 50
n_evals = 2000

masker_blur = shap.maskers.Image("blur(64,64)", sX[0].shape)
# funkcja predict, masker output list aklas

def predict(tensor):
    tensor = torch.Tensor(tensor) # nie do końca wiem czemu
    tensor = tensor.to(device)
    out = model(tensor)
    return out

out = predict(sX[7:8])
explainer = shap.Explainer(predict, masker_blur, output_names=class_names)

#%%
shap_values = explainer(
    sX[7:8],
    max_evals=n_evals,
    batch_size=batch_size,
    outputs=shap.Explanation.argsort.flip[:n_out]
)


#%%

#shap_values.values = [val for val in np.moveaxis(shap_values.values[0], -1, 0)]
labels=shap_values.output_names
true_labels=[class_names[out.argmax().numpy()]]

s_values = shap_values.values[0]
s_values = np.swapaxes(s_values,-1,0)
shap.image_plot(s_values,X[7:8][0],labels)

len(s_values)


plt.imshow(s_values[0])
#%%
import gc
gc.collect()
