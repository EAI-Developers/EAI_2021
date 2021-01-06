import cv2 as cv
import numpy as np

Object = {'Enemies':[[198, 89, 179], [200, 72, 72], [84, 184, 153], [180, 122, 48]], 'Me':[[[210, 164, 74]]]}

def get_pre_map(image):
    image = np.transpose(image[0][0].numpy(), (1, 2, 0))
    
    A = np.bitwise_or(image == Object['Enemies'][0], image == Object['Enemies'][1])
    B = np.bitwise_or(image == Object['Enemies'][2], image == Object['Enemies'][3])
    C = np.bitwise_or(A, B)
    C = np.array(C, dtype='uint8') * 255
    
    kernel = np.ones((7, 5), np.uint8)
    
    dilation = cv.dilate(C, kernel, iterations = 1)[:,:,0]
    attention = cv.resize(dilation, (20, 27), interpolation = cv.INTER_AREA)
    
    Me = image == Object['Me'][0]
    Me = np.array(Me, dtype='uint8') * 255
    dilation1 = cv.dilate(Me, kernel, iterations = 1)[:,:,0]
    attention1 = cv.resize(dilation1, (20, 27), interpolation = cv.INTER_AREA)
    
    Res = {'Me':attention1[:, :, np.newaxis], 'Enemies':attention[:, :, np.newaxis]}
    return Res
  