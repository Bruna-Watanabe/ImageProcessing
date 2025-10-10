import pygetwindow as gw

def MoveWindow():
    #para a imagem nao aparecer fora dos monitores
    try:
        w = gw.getWindowsWithTitle('janela ')
        # w[0].moveTo(200,162)
        w[0].moveTo(0,0)
        w[0].activate()
    except Exception as e:
        print(f'janela {e}')