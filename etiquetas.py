import glob
import os
import pandas as pd
import numpy as np

#cambiar al directorio actual
listado = pd.read_csv('/home/salvador/Documents/170520w/lista.csv')

#print(listado)

label = listado['LABEL']
#print(label)

etiquetas = np.array(label)

#print(etiquetas[1])