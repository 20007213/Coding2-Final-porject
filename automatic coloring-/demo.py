# -*- coding: utf-8 -*-

import scipy
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from model import Pix2Pix
from PIL import Image
import cv2

def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)

def predict_single_image(pix2pix, image_path, save_path):
    pix2pix.generator.load_weights('./weights/generator_weights.h5')
    image_B = imread(image_path)
    image_B = scipy.misc.imresize(image_B, (pix2pix.nW, pix2pix.nH))
    images_B = []
    images_B.append(image_B)
    images_B = np.array(images_B)/127.5 - 1.
    generates_A = pix2pix.generator.predict(images_B)
    generate_A = generates_A[0]
    generate_A = np.uint8((np.array(generate_A) * 0.5 + 0.5) * 255)
    generate_A = Image.fromarray(generate_A)
    generated_image = Image.new('RGB', (pix2pix.nW, pix2pix.nH))
    generated_image.paste(generate_A, (0, 0, pix2pix.nW, pix2pix.nH))
    generated_image.save(save_path, quality=95)
    pass


def convert_to_gray_single_image(image_path, save_path, resize_height=256, resize_weidth=256): 
    img = Image.open(image_path)
    img_color = img.resize((resize_weidth, resize_height), Image.ANTIALIAS)
    img_gray = img_color.convert('L')
    img_gray = img_gray.convert('RGB')
    img_gray.save(save_path, quality=95)


if __name__ == '__main__':
    Source_files_path = './images/test.jpg'
    Black_and_white_pictures_path = './images/Black_and_white_test.jpg'
    Automatically_generated_pictures_path = './images/Automatically_generated_test.jpg'

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 打开摄像头

    while True:
        # get a frame
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)  # 摄像头是和人对立的，将图像左右调换回来正常显示
        # show a frame
        cv2.startWindowThread()
        cv2.namedWindow("capture")
        cv2.imshow("capture", frame)  # 生成摄像头窗口

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 如果按下q 就截图保存并退出
            cv2.imwrite(Source_files_path, frame)  # 保存路径
            break

    cap.release()
    cv2.destroyAllWindows()

    img = Image.open(Source_files_path)   # 读取图片
    img = img.resize((256, 256), Image.ANTIALIAS).convert("L")   # 调整图片像素为256*256并转化为黑白图片
    img.save(Black_and_white_pictures_path)   # 存储图片

    gan = Pix2Pix()
    # gan.train(epochs=1200, batch_size=4, sample_interval=10, load_pretrained=True)

    predict_single_image(gan, Black_and_white_pictures_path, Automatically_generated_pictures_path)
    print(f"程序运行完成")
