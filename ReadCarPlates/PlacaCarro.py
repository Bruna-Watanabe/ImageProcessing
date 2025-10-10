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

def ResizeImg(img, target_width = 600):
    (h, w) = img.shape[:2] # Get original height and width

    # Define a target width and calculate the corresponding height
    ratio = target_width / float(w)
    target_height = int(h * ratio)
    new_dimensions_aspect_ratio = (target_width, target_height)    

    resized = cv2.resize(img, new_dimensions_aspect_ratio)
    return resized

def StackImgs(imgs):
    return np.vstack([np.hstack(imgs)])

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

def MostraImagem(nome, img, resize_width = 600):
    img = ResizeImg(img, resize_width)
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
    # aplica blur
    
    #mesma imagem com filtro de abertura
    img_car_opening = cv2.morphologyEx(imgOriginal, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    img_car_opening = cv2.cvtColor(img_car_opening, cv2.COLOR_BGR2GRAY)
    #imagem com filtro top hat
    img_car_tophat = cv2.morphologyEx(imgOriginal, cv2.MORPH_TOPHAT, np.ones((38,38),np.uint8))
    img_car_tophat = cv2.cvtColor(img_car_tophat, cv2.COLOR_BGR2GRAY)

    #filtra antes
    filtrado1 = cv2.bilateralFilter(img, 11, 75, 75)
    suave1 = cv2.GaussianBlur(filtrado1, (15, 15), 0)
    #reducao de ruido
    #blur antes
    suave2 = cv2.GaussianBlur(img, (9, 9), 0)
    filtrado2 = cv2.bilateralFilter(suave2, 11, 17, 17)
    #open antes
    suave3 = cv2.GaussianBlur(img_car_opening, (9, 9), 0)
    filtrado3 = cv2.bilateralFilter(suave3, 11, 17, 17)
    #tophaat antes
    suave4 = cv2.GaussianBlur(img_car_tophat, (9, 9), 0)
    filtrado4 = cv2.bilateralFilter(suave4, 11, 17, 17)

    # stack = StackImgs([filtrado1, suave1, filtrado2])
    stack = StackImgs([suave4, filtrado4])
    MostraImagem('blur', stack, 1600)
    #region teste
    
    threshold1 = cv2.adaptiveThreshold(suave1, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    threshold2 = cv2.adaptiveThreshold(filtrado2, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    threshold3 = cv2.adaptiveThreshold(filtrado3, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    threshold4 = cv2.adaptiveThreshold(filtrado4, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 7, 5)
    
    
    contornos1, _ = cv2.findContours(threshold1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos2, _ = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos35 = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos3, _ = cv2.findContours(threshold3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos4, _ = cv2.findContours(threshold4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    kernel = np.ones((5, 5), np.uint8)
    # img_moedas_erode1 = cv2.morpho(threshold2, kernel, iterations=1)
    img_moedas_erode1 = cv2.morphologyEx(threshold2, cv2.MORPH_CLOSE, kernel)

    a= StackImgs([threshold2 ,img_moedas_erode1])
    # MostraImagem('a',a,1800)
    b = StackImgs([img_car_opening,img_car_tophat])
    # MostraImagem('a',b,1800)

    contornos35, _ = cv2.findContours(img_moedas_erode1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result1 = imgOriginal.copy()
    result2 = imgOriginal.copy()
    result25 = imgOriginal.copy()
    result3 = imgOriginal.copy()
    result4 = imgOriginal.copy()

    cv2.drawContours(result1, contornos1, -1, (255, 0, 0), 2)
    cv2.drawContours(result2, contornos2, -1, (255, 0, 0), 2)
    cv2.drawContours(result25, contornos35, -1, (255, 0, 0), 2)
    cv2.drawContours(result3, contornos3, -1, (255, 0, 0), 2)
    cv2.drawContours(result4, contornos4, -1, (255, 0, 0), 2)

    stack = StackImgs([result1, result2,result25,result3,result4])
    stack2 = StackImgs([result25,result4])
    # MostraImagem('contornos', stack, 1900)
    # MostraImagem('contornos', stack2, 1600)


    # MostraImagem('contornos', imgOriginal)
    
    # Ordenar contornos por área (maiores primeiro)
    contours = sorted(contornos4, key=cv2.contourArea, reverse=True)[:30]
    
    achou = []
    
    allCimg = imgOriginal.copy()
    # Procurar contorno retangular (placa)
    for contour in contours:
        # Aproximar o contorno
        perimetro = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.018 * perimetro, True)
        
        cv2.drawContours(allCimg, [approx], -1, (0, 255, 0), 2)

        # Verificar se é um retângulo (4 lados)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            area = cv2.contourArea(contour)
            
            # Validar proporções típicas de placas brasileiras
            # Carros: ~3.5:1, Motos: ~2.5:1 a 3:1
            if 2.0 <= aspect_ratio <= 5.0 and area > 1000:
                achou.append(approx)

    print(f'achou {len(achou)} placas')
    for i in achou:
        cv2.drawContours(imgOriginal, [i], -1, (0, 255, 0), 2)

    a = StackImgs([allCimg,imgOriginal])
    MostraImagem('contornos', a,1800)
    #endregion


    # #region veio
    # try:
    #     contours = imutils.grab_contours(contornos35)
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    #     location = None
    #     for contour in contours:
    #         approx = cv2.approxPolyDP(contour, 10, True)
    #         if len(approx) == 4:
    #             location = approx
    #             break
    #     mask = np.zeros(img.shape, np.uint8)
    #     new_image = cv2.drawContours(mask, [location], 0,255, -1)
    #     new_image = cv2.bitwise_and(img, img, mask=mask)
    #     MostraImagem('novo',cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    # except Exception as e:
    #     print(e)
    # # endregion
    # novo = False
    # if novo:
    #     #threshhold
    #     # bin2 = cv2.adaptiveThreshold(suave, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5)
    #     threshold = cv2.adaptiveThreshold(filtrado, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)
        
    #     thresholdSemFiltro = cv2.adaptiveThreshold(suave, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 5)

    #     edged = cv2.Canny(filtrado, 30, 200) #Edge detection

    #     keypoints = cv2.findContours(threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     contours = imutils.grab_contours(keypoints)
    #     contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    #     locations = []
    #     for contour in contours:
    #         approx = cv2.approxPolyDP(contour, 10, True)
    #         if len(approx) == 4:
    #             locations.append(approx)

    #     # print(locations)
        
    #     try:
    #         MostraImagem('img', imgOriginal)        
    #         print(f'achei {len(locations)} coisos')
            
    #         for l in locations:
    #             mask = np.zeros(img.shape, np.uint8)
    #             new_image = cv2.drawContours(mask, [l], 0,255,-1)
    #             new_image = cv2.bitwise_and(img, img, mask=mask)

    #             placa = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)

    #             LePlaca(new_image)
    #             MostraImagem('img', new_image)
            
    #         #outro
    #         kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #         morph_opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    #         contornos, _ = cv2.findContours(morph_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         cv2.drawContours(imgOriginal, contornos, -1, (255, 0, 0), 2)

    #         (x,y) = np.where(mask==255)
    #         (x1, y1) = (np.min(x), np.min(y))
    #         (x2, y2) = (np.max(x), np.max(y))
    #         cropped_image = suave[x1:x2+1, y1:y2+1]

    #         resultado = StackImgs([img, thresholdSemFiltro, threshold])
    #         resultado2 = StackImgs([imgOriginal, cropped_image])







    #         # cv2.drawContours(imgOriginal, contornos, -1, (0, 255, 0), 2)
            
    #         # resultado = np.vstack([np.hstack([img, bin1,imgTophat])])
    #         # resultado2 = np.vstack([np.hstack([bin1, bin2])])
    #         # cv2.imshow("Binarização da imagem", resultado1)
    #         # MoveWindow()
    #         # cv2.waitKey(0)
    #         LePlaca(threshold)
    #         MostraImagem("img", cropped_image)
    #         # MostraImagem("original", imgOriginal)
        
    #     except Exception as e:
    #         #print(e)
    #         pass
    #     ## LePlaca(binI)

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
        i = ResizeImg(i, 1000)
        LimpaImagem(i)


Main()