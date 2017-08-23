"""
This code reads all the names of the files in a directory
@ directorio_a_leer: ruta completa y tipo de archivo
@ archivo_a_crear: nombre del archivo csv
"""

import glob
import pandas as pd

directorio_a_leer = glob.glob("/home/salvador/Documents/170520w/augmented/WFZ/*.bmp")
archivo_a_crear = "/home/salvador/Documents/170520w/listav1.csv"

listado = []

for archivo in directorio_a_leer:
    listado.append(archivo)


listas = pd.DataFrame(listado)
listas.to_csv(archivo_a_crear)
