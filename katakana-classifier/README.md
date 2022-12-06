# KATAKANA Classifier



## Structure
```
learn-ml/katakana-classifier/2_notebook/
├── 1_data
│   ├── train_data.npy
│   └── train_label.npy
├── 2_notebook
│   ├── common
│   │   ├── License.txt
│   │   ├── activations.py
│   │   ├── gradient.py
│   │   ├── layers.py
│   │   ├── loss.py
│   │   ├── model.py : Conv2DNN Class
│   │   ├── optimizer.py
│   │   └── util.py
│   ├── katakana_model_bn_params.pickle
│   ├── katakana_model_hp_params.pickle
│   ├── katakana_model_params.pickle
│   ├── scratch_conv2dnn.ipynb : Train Conv2DNN with scratch
│   ├── submit_katakana.ipynb
│   ├── tensorflow_conv2dnn.ipynb : Train Conv2DNN by using tensorflow
│   └── util.py
└── README.md
```



## How to run
run `scratch_conv2dnn.ipynb` or `tensorflow_conv2dnn.ipynb`



## Reference
- https://www.kaggle.com/code/elcaiseri/mnist-simple-cnn-keras-accuracy-0-99-top-1
