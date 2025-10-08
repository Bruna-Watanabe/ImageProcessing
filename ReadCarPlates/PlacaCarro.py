from matplotlib import pyplot as plt
from pytesseract import pytesseract
import numpy as np
import imutils
import cv2
import os
from Histograma import Histograma
from janela import MoveWindow

imgsPath = 'imgs'

global imgs
imgs = []

def PegaImagens():
    imgNames = os.listdir('imgs')
    print(imgNames)
    for i in imgNames:
        imgs.append(cv2.imread(f'imgs/{i}'))

    print(f'peguei {len(imgs)} imagens')

def ResizeImg(img):
    (h, w) = img.shape[:2] # Get original height and width

    # Define a target width and calculate the corresponding height
    target_width = 600
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    new_dimensions_aspect_ratio = (target_width, target_height)    

    resized = cv2.resize(img, new_dimensions_aspect_ratio)
    return resized

def showMultipleImages(imgsArray, titlesArray):
    for i in range(len(imgsArray)):
        cv2.imshow(titlesArray[i], imgsArray[i])

def LePlaca(img):
    pathToTesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    pytesseract.tesseract_cmd = pathToTesseract

    #extract text from image
    text = pytesseract.image_to_string(img, config='tessedit_char_whitelist=0123456789')
    print(f'leu: {text}')
    return text

def MostraImagem(nome, img):
    cv2.imshow(nome, img)
    MoveWindow()
    cv2.waitKey(0)

def LimpaImagem(img):
    #Binarização com limiar
    # img = cv2.imread('ponte.jpg')

    imgOriginal = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # suave = cv2.GaussianBlur(img, (1, 1), 0) # aplica blur
    suave = img

    # kernel = np.ones((4, 4), np.uint8)
    # img_dil2 = cv2.dilate(img, kernel, iterations=2)
    # img_dil4 = cv2.dilate(img, kernel, iterations=4)
    # img_dil6 = cv2.dilate(img, kernel, iterations=6)
    # img_dil8 = cv2.dilate(img, kernel, iterations=8)
    # img_dil10 = cv2.dilate(img, kernel, iterations=10)
    # imgsArray = [img, img_dil2, img_dil4, img_dil6, img_dil8, img_dil10]
    # titlesArray = ['Original', 'Dilate lv. 2', 'Dilate lv. 4', 'Dilate lv. 6', 'img_dil8','img_dil10']
    # showMultipleImages(imgsArray, titlesArray)


    ## (T, bin) = cv2.threshold(equalizada, 130, 255, cv2.THRESH_BINARY)
    ## (T, binI) = cv2.threshold(equalizada, 130, 255, cv2.THRESH_BINARY_INV)
    ## cv2.bitwise_and(img, img, mask = binI)
    
    # resultado1 = np.vstack([np.hstack([suave, bin])])
    # resultado2 = np.vstack([np.hstack([binI, cv2.bitwise_and(img, img, mask = binI)])])
    # cv2.imshow("Binarização da imagem", resultado1)
    # cv2.waitKey(0)
    # cv2.imshow("Binarização da imagem", resultado2)
    # cv2.waitKey(0)

    #reducao de ruido
    filtrado = cv2.bilateralFilter(suave, 11, 17, 17)

    #threshhold
    # bin2 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    threshold = cv2.adaptiveThreshold(filtrado, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    
    thresholdSemFiltro = cv2.adaptiveThreshold(suave, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    edged = cv2.Canny(filtrado, 30, 200) #Edge detection

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 10, True)
        if len(approx) == 4:
            location = approx
            break

    print(location)
    
    try:
        mask = np.zeros(img.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0,255,-1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        placa = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)




        
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
        # contornos, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(imgOriginal, objetos, -1, (255, 0, 0), 1)
        resultado = np.vstack([np.hstack([img, thresholdSemFiltro, threshold])])
        resultado2 = np.vstack([np.hstack([imgOriginal, placa])])

        # cv2.drawContours(imgOriginal, contornos, -1, (0, 255, 0), 2)
        
        # resultado = np.vstack([np.hstack([img, bin1,imgTophat])])
        # resultado2 = np.vstack([np.hstack([bin1, bin2])])
        # cv2.imshow("Binarização da imagem", resultado1)
        # MoveWindow()
        # cv2.waitKey(0)
        LePlaca(threshold)
        MostraImagem("img", resultado2)
        # MostraImagem("original", imgOriginal)
    
    except:
        print('erro')
    ## LePlaca(binI)

    ## resultado1 = np.vstack([np.hstack([suave, equalizada, binI])])
    ## cv2.imshow("Binarizacao da imagem", resultado1)    
    ## MoveWindow()
    ## Histograma(img)
    ## Histograma(equalizada)
    ## cv2.waitKey(0)


    #threshhold adaptativo ficou uma bosta
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
    # suave = img
    # bin1 = cv2.adaptiveThreshold(suave, 255,
    # cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    # bin2 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    # resultado1 = np.vstack([np.hstack([img, suave])])
    # resultado2 = np.vstack([np.hstack([bin1, bin2])])
    # cv2.imshow("Binarização da imagem", resultado1)
    # cv2.waitKey(0)
    # cv2.imshow("Binarização da imagem", resultado2)
    # cv2.waitKey(0)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converte
    # suave = img
    # _, bin = cv2.threshold(suave, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # resultado1 = np.vstack([
    # np.hstack([img, suave])
    # ])
    # resultado2 = np.vstack([
    # np.hstack([bin, bin])
    # ])
    
    # cv2.imshow("Binarização da imagem", resultado1)
    # cv2.waitKey(0)
    # cv2.imshow("Binarização da imagem", resultado2)
    # cv2.waitKey(0)

# PegaImagens()

# img = cv2.imread('imgs/image15.jpg')
# LimpaImagem(ResizeImg(img))

def Main():
    PegaImagens()
    for i in imgs:
        i = ResizeImg(i)
        LimpaImagem(i)


Main()