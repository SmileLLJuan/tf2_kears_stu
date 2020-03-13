#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time   : 2020/3/13 16:16
# @Author :llj
import numpy as np
import keras
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
def random_data(num,pix_num=784):
    a_list=[np.random.randint(0,255,pix_num) for i in range(num)]
    return np.array(a_list)
def random_label(num,num_class=10):
    a_list=[np.random.randint(0,num_class,1)[0] for i in range(num)]
    return a_list
num_class=10
class LeNet():
    def __init__(self):#输入图片大小(n,28,28,1)
        (self.train_data, self.train_label), (self.test_data, self.test_label) = (random_data(1000),random_label(1000)),(random_data(100),random_label(100))
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
        # self.train_label = self.train_label.astype(np.int32) # [60000]
        # self.test_label = self.test_label.astype(np.int32) # [10000]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
        print("self.test_label[:2]",self.test_label[:2])
        self.train_data=self.train_data.reshape(-1,28,28,1)
        self.test_data=self.test_data.reshape(-1,28,28,1)
        self.train_label_onehot = to_categorical(self.train_label,num_classes=num_class) # [60000]
        self.test_label_onehot = to_categorical(self.test_label,num_classes=num_class) # [60000]

        self.create_model()

    def create_model(self):
        model=keras.Sequential()
        model.add(keras.layers.Conv2D(filters=32,kernel_size=(5,5),padding='valid',input_shape=(28,28,1),activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

        model.add(keras.layers.Conv2D(filters=64, kernel_size=(5, 5), padding='valid',activation='relu'))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128,activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10,activation='relu'))

        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model=model
    def train_model(self):
        self.model.fit(self.train_data,self.train_label_onehot,batch_size=128,epochs=10,verbose=1)
    def evlauate_model(self):
        score=self.model.evaluate(self.test_data,self.test_label_onehot,batch_size=128,verbose=1)
        print("Test loss:{},Test accuracy:{}".format(score[0],score[1]))
class AlexNet():
    def __init__(self):#输入图片大小(n,227,227,3)
        self.num_class=10
        (self.train_data, self.train_label), (self.test_data, self.test_label) = (random_data(1000,227*227*3),random_label(1000,self.num_class)),(random_data(100,227*227*3),random_label(100,self.num_class))
        self.train_data =self.train_data.astype(np.float32) / 255.0  # [60000, 28, 28, 1]
        print(self.train_data.shape)
        self.test_data = self.test_data.astype(np.float32) / 255.0  # [10000, 28, 28, 1]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]
        self.train_data=self.train_data.reshape(-1,227,227,3)
        self.test_data=self.test_data.reshape(-1,227,227,3)
        print(self.train_data.shape)

        self.train_label_onehot = to_categorical(self.train_label,num_classes=self.num_class) # [60000]
        self.test_label_onehot = to_categorical(self.test_label,num_classes=self.num_class) # [60000]
        self.create_model()

    def create_model(self):
        input=keras.Input(shape=(227,227,3))
        conv1=keras.layers.Conv2D(filters=96,kernel_size=(11,11),padding='valid',strides=4,activation='relu')(input)
        conv1=keras.layers.MaxPool2D(pool_size=(3,3),strides=2)(conv1)

        conv2 = keras.layers.Conv2D(filters=256, kernel_size=(5,5), padding='same', strides=1)(conv1)
        conv2 = keras.layers.Activation("relu")(conv2)
        conv2 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(conv2)

        conv3 = keras.layers.Conv2D(filters=384, kernel_size=(3,3), padding='same', strides=1)(conv2)
        conv3 = keras.layers.Activation("relu")(conv3)

        conv4 = keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', strides=1)(conv3)
        conv4 = keras.layers.Activation("relu")(conv4)

        conv5 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', strides=1)(conv4)
        conv5 = keras.layers.Activation("relu")(conv5)
        conv5 = keras.layers.MaxPool2D(pool_size=(3, 3), strides=2)(conv5)

        fc_1=keras.layers.Flatten()(conv5)
        fc_2=keras.layers.Dense(4096)(fc_1)
        fc_3=keras.layers.Dense(10,activation='softmax')(fc_2)

        model=keras.Model(inputs=input,outputs=fc_3)
        model.summary()#查看网络结构
        model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model=model
    def train_model(self):
        self.model.fit(self.train_data,self.train_label_onehot,batch_size=128,epochs=10,verbose=1,callbacks=[EarlyStopping])
    def evlauate_model(self):
        score=self.model.evaluate(self.test_data,self.test_label_onehot,batch_size=128,verbose=1)
        print("Test loss:{},Test accuracy:{}".format(score[0],score[1]))

def main():
    ln = AlexNet()
    print(ln.train_data.shape)
    print(ln.train_label_onehot.shape, ln.train_label_onehot[:2])
    ln.train_model()
    ln.evlauate_model()

if __name__=="__main__":
    main()