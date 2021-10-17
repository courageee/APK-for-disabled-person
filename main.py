import cv2
import logic

logic.debug = False

img = cv2.imread(r'C:\Users\admin\PycharmProjects\pythonProject6\fasadnaya-vyveska-dlya-biblioteki.jpg')
img1 = cv2.imread(r'C:\Users\admin\PycharmProjects\pythonProject6\lib.jpg')
img2 = cv2.imread(r'C:\Users\admin\PycharmProjects\pythonProject6\images.jpg')

print('*************************************************')
text = logic.read_image(img)
print(text)
print('*************************************************')
text = logic.read_image(img1)
print(text)
print('*************************************************')
text = logic.read_image(img2)
print(text)

