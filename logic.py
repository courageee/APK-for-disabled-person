import cv2
import pytesseract
from pytesseract import Output
from scipy import ndimage

debug = False

tes_v5 = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
tes_v3 = r'C:\Program Files (x86)\Tesseract-OCR-3\tesseract.exe'

# выбор версии tesseract'а
def tesseract_init(tes_version=tes_v5):
    pytesseract.pytesseract.tesseract_cmd = tes_version

# делаем преобразования картинки (для  1ого случая)
def preprocess_image(img):
    processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.medianBlur(processed_img, 3)
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15) #*
    return processed_img

# делаем преобразования картинки (для  2ого случая)
def last_chance(img):
    processed_img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    processed_img = cv2.bitwise_not(processed_img)
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.medianBlur(processed_img, 1)
    processed_img = cv2.adaptiveThreshold(processed_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)
    return processed_img

# в первом случае хорошо работает с темным текстом на светлом фоне
# во втором случае работаем со светлым текстом на темном фоне


def rectify_image_rotation(img): # функция принимает в себя картинку
    tesseract_init(tes_version=tes_v3)
    success = False

    try: # блок, где мы глушим ошибки
        rectified_image = preprocess_image(img)
        osd = pytesseract.image_to_osd(rectified_image, output_type=Output.DICT)
        success = True

    except pytesseract.pytesseract.TesseractError: # выполняем блок except, если в try ошибка

        try:
            rectified_image = last_chance(img) # если не получилось с первого раза, пробуем со второго раза функцию last_chance
            osd = pytesseract.image_to_osd(rectified_image, output_type=Output.DICT)
            success = True

        except pytesseract.pytesseract.TesseractError:
            return img, success

    angle = -osd['rotate']

    if debug:
        print('Повернуто на', angle, 'градусов')

    rectified_img = ndimage.rotate(rectified_image, angle)
    return rectified_img, success 
     #

def read_image(img):
    img, success = rectify_image_rotation(img)
    if success:
        tesseract_init(tes_version=tes_v5)
        text = pytesseract.image_to_string(img, lang='rus+eng')
        return text
    else:
        return 'не удалось прочесть текст'

