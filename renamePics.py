import glob
import os

#cambiar al directorio actual
os.chdir('/home/salvador/Documents/170520w/augmented/WF3/')

#Listar todos los archivos de imagen.
lista = glob.glob('*.bmp')
for fname in lista:
    #print(fname, fname[32:40])

    #Renombrar los archivos eliminando del inicio a la posicion 31.
    nombre = fname[32:40]+".bmp"
    os.rename(fname, nombre)
    #nombre = "A"+fname[1:]
    #os.rename(fname, nombre)
