import pygetwindow as gw

def MoveWindow():
    #para a imagem nao aparecer fora dos monitores
    try:
        w = gw.getWindowsWithTitle('binariza')
        # w[0].moveTo(200,162)
        w[0].moveTo(110,0)
        w[0].activate()
    except Exception as e:
        print(e)