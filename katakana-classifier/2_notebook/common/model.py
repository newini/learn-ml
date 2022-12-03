# coding: utf-8
import numpy as np
from collections import OrderedDict
from common.layers import Convolution, MaxPooling, ReLU, BatchNormalization, Affine, SoftmaxWithLoss

def heNormalInitializer(n1):
    """
    Heの初期値を利用するための関数
    返り値は、見かけの標準偏差
    """    
    return np.sqrt(2/n1)

def calculateOutputSize(input_size, pad, size, stride):
    return (input_size + 2*pad - size) // stride + 1
    
class Conv2DNN:
    def __init__(
        self, input_dim=(1, 28, 28), 
        weight_init_std=0.01,
        **kwargs
    ):
        """
        input_dim : tuple, 入力の配列形状(チャンネル数、画像の高さ、画像の幅)
        weight_init_std ： float, 重みWを初期化する際に用いる標準偏差
        """
                        
        channel_size = input_dim[0]
        input_size = input_dim[1]
        layer_params = kwargs['layer_params']
        self.layer_name_list = kwargs['layer_params'].keys()
        
        # 重みの初期化, レイヤの生成, layer追加を同時にやる
        self.params = {}
        self.layers = OrderedDict()
        std = weight_init_std
        i = 0
        for key in self.layer_name_list:
            if 'Conv' in key:
                units = layer_params[key]['channel']*layer_params[key]['size']**2
                std = heNormalInitializer(units)
                self.params[f'W{i}'] = std * np.random.randn(layer_params[key]['channel'], channel_size, layer_params[key]['size'], layer_params[key]['size'])
                self.params[f'b{i}'] = np.zeros(layer_params[key]['channel'])
                self.layers[key] = Convolution(self.params[f'W{i}'], self.params[f'b{i}'], layer_params[key]['stride'], layer_params[key]['pad'])
                output_size = calculateOutputSize(input_size, layer_params[key]['pad'], layer_params[key]['size'], layer_params[key]['stride'])
                channel_size = layer_params[key]['channel']
                params = self.layers[key].W.size + self.layers[key].b.size
                print(f'{key}\t\t{(None, channel_size, output_size, output_size)}\t{params}')
                i += 1
            elif 'Pool' in key:
                self.layers[key] = MaxPooling(pool_h=layer_params[key]['size'], pool_w=layer_params[key]['size'], stride=layer_params[key]['stride'], pad=layer_params[key]['pad'])
                output_size = calculateOutputSize(input_size, layer_params[key]['pad'], layer_params[key]['size'], layer_params[key]['stride'])
                print(f'{key}\t\t{(None, channel_size, output_size, output_size)}')
            elif 'ReLU' in key:
                self.layers[key] = ReLU()
            elif 'BatchNorm' in key:
                gamma = np.ones(channel_size)
                beta = np.zeros(channel_size)
                self.layers[key] = BatchNormalization(gamma, beta)
                print(f'{key}\t{(None, channel_size, output_size, output_size)}')
            elif 'Flatten' in key:
                output_size = channel_size * output_size * output_size
                print(f'{key}\t\t{(None, output_size)}')
            elif 'Affine' in key:
                units = layer_params[key]['hidden_size']
                std = heNormalInitializer(units)
                self.params[f'W{i}'] = std * np.random.randn(input_size, layer_params[key]['hidden_size'])
                self.params[f'b{i}'] = np.zeros(layer_params[key]['hidden_size'])
                self.layers[key] = Affine(self.params[f'W{i}'], self.params[f'b{i}'])
                output_size = layer_params[key]['hidden_size']
                params = self.layers[key].W.size + self.layers[key].b.size
                print(f'{key}\t\t{(None, output_size)}\t\t{params}')
                i += 1
                
            input_size = output_size

        self.last_layer = SoftmaxWithLoss()
    
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝播法）
        Parameters
        ----------
        x : 入力データ
        t : 教師データ
        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        i = 0
        for key in self.layer_name_list:
            if (
                ('Conv' in key) or ('Affine' in key)
            ):
                grads[f'W{i}'], grads[f'b{i}'] = self.layers[key].dW, self.layers[key].db
                i += 1
        
        return grads