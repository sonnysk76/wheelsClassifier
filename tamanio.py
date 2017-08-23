import glob
import pandas as pd
from  PIL import Image
import os
listado = pd.read_csv('/home/salvador/Documents/170520w/listaderF.csv')
label = listado['VIN']
i=0

#text = glob.glob("/home/salvador/Documents/170520w/LeftFrontWheel/*.bmp")
os.chdir("/home/salvador/Documents/170520w/comder/")

for foto in label:
    im = Image.open("/home/salvador/Documents/170520w/LeftFrontWheel/"+str(foto)+".bmp")
    img = im.resize((200, 200), Image.ANTIALIAS)    #Resize 200x200
    #gray = img.convert('L')                          #Grayscale
    nombre = str(label[i]) +".bmp"
    img.save(nombre)
    i = i + 1