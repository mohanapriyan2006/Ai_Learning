'''
    first install this : https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20210506.exe
    and then RUN it
'''

try:
    from PIL import Image
except:
    import Image

import pytesseract


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def readText(file):
    text = pytesseract.image_to_string(Image.open(file))
    return text

info = readText("test/TEST3.jpg")
print("\n TEXT from given IMAGE :\n")
print(info)

with open("output.txt",'a') as file:
    file.write(info)
    file.close()
print("Text Saved in 'output.txt' file")