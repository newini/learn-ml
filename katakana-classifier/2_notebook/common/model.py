# coding: utf-8
import numpy as np
from collections import OrderedDict
from common.activations import *
from common.layers import Convolution, MaxPooling, ReLU, BatchNormalization, Affine, SoftmaxWithLoss, Dropout

def heNormalInitializer(fan_in):
    """
    Heの初期値を利用するための関数
    返り値は、見かけの標準偏差
    """    
    return np.sqrt(2/n_i)

def GlorotUniform(fan_in, fan_out):
    return np.sqrt(6 / (fan_in + fan_out))
    
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
                        
        self.input_dim = input_dim
        channel_size = self.input_dim[0]
        input_size = self.input_dim[1]
        self.layer_params = kwargs['layer_params']
        layer_params = self.layer_params
        
        # 重みの初期化, レイヤの生成, layer追加を同時にやる
        self.params = {}
        self.bn_params = {} # for Batch Normalization
        self.layers = OrderedDict()
        std = weight_init_std
        i = 0
        for key in self.layer_params.keys():
            if 'Conv' in key:
                fan_in = channel_size*layer_params[key]['size']**2
                fan_out = layer_params[key]['channel']*layer_params[key]['size']**2
                # std = heNormalInitializer(fan_in)
                std = GlorotUniform(fan_in, fan_out) 
                self.params[f'W{i}'] = std * np.random.randn(layer_params[key]['channel'], channel_size, layer_params[key]['size'], layer_params[key]['size'])
                self.params[f'b{i}'] = np.zeros(layer_params[key]['channel'])
                self.layers[key] = Convolution(self.params[f'W{i}'], self.params[f'b{i}'], layer_params[key]['stride'], layer_params[key]['pad'])
                output_size = calculateOutputSize(input_size, layer_params[key]['pad'], layer_params[key]['size'], layer_params[key]['stride'])
                channel_size = layer_params[key]['channel']
                params = self.layers[key].W.size + self.layers[key].b.size
                i += 1
            elif 'Pool' in key:
                self.layers[key] = MaxPooling(pool_h=layer_params[key]['size'], pool_w=layer_params[key]['size'], stride=layer_params[key]['stride'], pad=layer_params[key]['pad'])
                output_size = calculateOutputSize(input_size, layer_params[key]['pad'], layer_params[key]['size'], layer_params[key]['stride'])
            elif 'ReLU' in key:
                self.layers[key] = ReLU()
            elif 'BatchNorm' in key:
                self.params[f'gamma{i}'] = np.ones(channel_size)
                self.params[f'beta{i}'] = np.zeros(channel_size)
                self.bn_params[f'moving_mean{i}'] = np.zeros(channel_size)
                self.bn_params[f'moving_var{i}'] = np.zeros(channel_size)
                self.layers[key] = BatchNormalization(self.params[f'gamma{i}'], self.params[f'beta{i}'], moving_mean=self.bn_params[f'moving_mean{i}'], moving_var=self.bn_params[f'moving_var{i}'])
                i += 1
            elif 'Dropout' in key:
                self.layers[key] = Dropout(layer_params[key]['dropout'])
            elif 'Flatten' in key:
                output_size = channel_size * output_size * output_size
            elif 'Affine' in key:
                fan_in = input_size
                fan_out = layer_params[key]['hidden_size']
                # std = heNormalInitializer(fan_in)
                std = GlorotUniform(fan_in, fan_out) 
                self.params[f'W{i}'] = std * np.random.randn(input_size, layer_params[key]['hidden_size'])
                self.params[f'b{i}'] = np.zeros(layer_params[key]['hidden_size'])
                self.layers[key] = Affine(self.params[f'W{i}'], self.params[f'b{i}'])
                output_size = layer_params[key]['hidden_size']
                params = self.layers[key].W.size + self.layers[key].b.size
                i += 1
            # elif 'Softmax' in key:
            #     self.layers[key] = softmax()
                
            input_size = output_size

        self.last_layer = SoftmaxWithLoss()
    
    def summary(self):
        channel_size = self.input_dim[0]
        input_size = self.input_dim[1]
        layer_params = self.layer_params
        for key in self.layer_params.keys():
            if 'Conv' in key:
                units = layer_params[key]['channel']*layer_params[key]['size']**2
                output_size = calculateOutputSize(input_size, layer_params[key]['pad'], layer_params[key]['size'], layer_params[key]['stride'])
                channel_size = layer_params[key]['channel']
                params = self.layers[key].W.size + self.layers[key].b.size
                print(f'{key}\t\t{(None, channel_size, output_size, output_size)}\t{params}\t{self.layers[key].W.shape}, {self.layers[key].b.shape}')
            elif 'Pool' in key:
                output_size = calculateOutputSize(input_size, layer_params[key]['pad'], layer_params[key]['size'], layer_params[key]['stride'])
                print(f'{key}\t\t{(None, channel_size, output_size, output_size)}')
            elif 'ReLU' in key:
                pass
            elif 'BatchNorm' in key:
                gamma_size = self.layers[key].gamma.shape
                params = self.layers[key].gamma.size * 4 # gamma, beta, moving_mean, moving_var have the same size
                print(f'{key}\t{(None, channel_size, output_size, output_size)}\t{params}\t{gamma_size}, {gamma_size}, {gamma_size}, {gamma_size}')
            elif 'Dropout' in key:
                print(f'{key}\t{(None, channel_size, output_size, output_size)}')
            elif 'Flatten' in key:
                output_size = channel_size * output_size * output_size
                print(f'{key}\t\t{(None, output_size)}')
            elif 'Affine' in key:
                units = layer_params[key]['hidden_size']
                output_size = layer_params[key]['hidden_size']
                params = self.layers[key].W.size + self.layers[key].b.size
                print(f'{key}\t\t{(None, output_size)}\t\t{params}\t{self.layers[key].W.shape}, {self.layers[key].b.shape}')
                
            input_size = output_size
        
    def predict(self, x, train_flg=True):
        for key, layer in self.layers.items():
            if (
                ('BatchNorm' in key) or ('Dropout' in key)
            ):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x, t, train_flg=True):
        """
        損失関数
        x : 入力データ
        t : 教師データ
        """
        y = self.predict(x, train_flg)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        
        acc = 0.0
        for i in range( int(np.ceil(x.shape[0] / batch_size)) ):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def gradient(self, x, t, train_flg=True):
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
        self.loss(x, t, train_flg)

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
        for key in self.layer_params.keys():
            if (
                ('Conv' in key) or ('Affine' in key)
            ):
                grads[f'W{i}'], grads[f'b{i}'] = self.layers[key].dW, self.layers[key].db
                i += 1
            elif ('BatchNorm' in key):
                grads[f'gamma{i}'], grads[f'beta{i}'] = self.layers[key].dgamma, self.layers[key].dbeta
                # not need for moving_mean and moving_var but save 
                self.bn_params[f'moving_mean{i}'], self.bn_params[f'moving_var{i}'] = self.layers[key].moving_mean, self.layers[key].moving_var
                
                i += 1
        
        return grads