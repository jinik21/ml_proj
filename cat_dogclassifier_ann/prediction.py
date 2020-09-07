#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 03:12:41 2020

@author: root
"""
import tensorflow as tf
model= tf.keras.models.load_model("64x3-CNN4.model")
categories=["Cat","Dog"]
from keras.preprocessing import image
import numpy as np
test_image=image.load_img('dataset/prediction/chk4.jpg',target_size=(256,256))#use 64,64 for other models in this directory
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image, axis=0)
result=model.predict(test_image)
print(categories[int(result[0][0])])