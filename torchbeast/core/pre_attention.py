import cv2 as cv
import numpy as np

Object = {'Enemies':[[198, 89, 179], [200, 72, 72], [84, 184, 153], [180, 122, 48]], 'Me':[[[210, 164, 74]]]}

def get_pre_map(image):
    image = np.transpose(image[0][0].numpy(), (1, 2, 0))
    kernel = np.ones((7, 5), np.uint8)
    
    A = np.bitwise_or(image == Object['Enemies'][0], image == Object['Enemies'][1])
    B = np.bitwise_or(image == Object['Enemies'][2], image == Object['Enemies'][3])
    C = np.bitwise_or(A, B)
    C = np.array(C, dtype='uint8') * 255
    dilation = cv.dilate(C, kernel, iterations = 1)[:,:,0]
    attention = cv.resize(dilation, (20, 27), interpolation = cv.INTER_AREA)
    
    Me = image == Object['Me'][0]
    Me = np.array(Me, dtype='uint8') * 255
    dilation1 = cv.dilate(Me, kernel, iterations = 1)[:,:,0]
    attention1 = cv.resize(dilation1, (20, 27), interpolation = cv.INTER_AREA)
    
    E1 = image == Object['Enemies'][0]
    E1 = np.array(E1, dtype='uint8') * 255
    dilationE1 = cv.dilate(E1, kernel, iterations = 1)[:,:,0]
    attentionE1 = cv.resize(dilationE1, (20, 27), interpolation = cv.INTER_AREA)
    
    E2 = image == Object['Enemies'][1]
    E2 = np.array(E2, dtype='uint8') * 255
    dilationE2 = cv.dilate(E2, kernel, iterations = 1)[:,:,0]
    attentionE2 = cv.resize(dilationE2, (20, 27), interpolation = cv.INTER_AREA)
    
    E3 = image == Object['Enemies'][2]
    E3 = np.array(E3, dtype='uint8') * 255
    dilationE3 = cv.dilate(E3, kernel, iterations = 1)[:,:,0]
    attentionE3 = cv.resize(dilationE3, (20, 27), interpolation = cv.INTER_AREA)
    
    E4 = image == Object['Enemies'][3]
    E4 = np.array(E4, dtype='uint8') * 255
    dilationE4 = cv.dilate(E4, kernel, iterations = 1)[:,:,0]
    attentionE4 = cv.resize(dilationE4, (20, 27), interpolation = cv.INTER_AREA)
    
    A = np.bitwise_or(image == Object['Enemies'][0], image == Object['Enemies'][1])
    B = np.bitwise_or(image == Object['Enemies'][2], image == Object['Enemies'][3])
    C = np.bitwise_or(A, B)
    Me = image == Object['Me'][0]
    All = np.bitwise_or(C, Me)
    All = np.array(All, dtype='uint8') * 255
    dilationAll = cv.dilate(All, kernel, iterations = 1)[:,:,0]
    attentionAll = cv.resize(dilationAll, (20, 27), interpolation = cv.INTER_AREA)
    
    Res = {'Me':attention1[:, :, np.newaxis], 'Enemies':attention[:, :, np.newaxis], 'All':attentionAll, 'E1':attentionE1, 'E2':attentionE2, 'E3':attentionE3, 'E4':attentionE4}
    return Res