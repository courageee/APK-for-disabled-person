import cv2

import logic
import pyttsx3





logic.debug = False
engine = pyttsx3.init()
engine.setProperty('rate', 250)

img = cv2.imread(r'C:\Users\admin\PycharmProjects\pythonProject6\fasadnaya-vyveska-dlya-biblioteki.jpg')
img1 = cv2.imread(r'C:\Users\admin\PycharmProjects\pythonProject6\lib.jpg')

print('*************************************************')
text = logic.read_image(img)
print(text)
engine.say(text)
engine.runAndWait()
print('*************************************************')
text = logic.read_image(img1)
print(text)
engine.say(text)
engine.runAndWait()
print('*************************************************')
