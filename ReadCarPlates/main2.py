import cv2
import pytesseract
import os
import argparse
import easyocr

# Configurar caminho do Tesseract se necessário (no Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

reader = easyocr.Reader(['pt'])

# Carregar o classificador de placas
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

def CheckPlate(text, image_path, plate_img, i, model_name):
    #veryfica se eh placa
    for t in text:
        if text != '' and t.isdigit() and t.isalpha:
            #print('é placa')
            print(model_name)
            print(f"{os.path.basename(image_path)} [{i+1}] : {text}")

            # Mostrar a imagem da placa com zoom
            cv2.imshow(f"Placa - {os.path.basename(image_path)} [{i+1}]", plate_img)

def process_image(image_path, show_text=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detectar placas
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 20))
    
    if len(plates) == 0:
        print(f"Nenhuma placa detectada em {os.path.basename(image_path)}")
        return
    
    for i, (x, y, w, h) in enumerate(plates):
        plate_img = img[y:y + h, x:x + w]

        if show_text:
            # Pré-processamento simples para OCR
            plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)
            _, plate_thresh = cv2.threshold(plate_gray, 150, 255, cv2.THRESH_BINARY)
            
            text = pytesseract.image_to_string(plate_thresh, config="--psm 8")
            CheckPlate(text, image_path, plate_img, i, 'pytesseract')
            
            easyocr_result = reader.readtext(plate_thresh)

            txt = ''
            text = ''
            for (bbox, txt, prob) in easyocr_result:
                # print(f'Text: {txt}, Probability: {prob}')
                text += f' txt'
            
            CheckPlate(text, image_path, plate_img, i, 'easyOcr')

            
    cv2.waitKey(0)

def process_folder():
    folder_path = 'imgs'
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(folder_path, file)
            process_image(image_path, True)

    print("\n--- Fim do processamento ---")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_folder()
