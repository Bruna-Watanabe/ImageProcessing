from matplotlib import pyplot as plt
import cv2

def Histograma(img, show = False):
    #Função para calcular o hisograma da imagem
    h = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.figure()
    plt.title("Histograma P&B")
    plt.xlabel("Intensidade")
    plt.ylabel("Qtde de Pixels")
    plt.plot(h)
    plt.xlim([0, 256])

    if show:
        plt.show()