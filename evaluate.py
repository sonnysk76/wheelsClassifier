from PIL import Image
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt



model = load_model('ClasAugm.h5')

sales_codes = {0:'WBA', 1:'WBE', 2:'WBH', 3:'WBL', 4:'WBN', 5:'WD4', 6:'WDA', 7:'WF3', 8:'WF7',
               9:'WF9', 10:'WFE', 11:'WFP', 12:'WFU', 13:'WFZ', 14:'WH3', 15:'WHE',16:'WHK', 17:'WHN',
               18:'WP3', 19:'WP4', 20:'WR2', 21:'WRA', 22:'WRF', 23:'WRG', 24:'WRJ', 25:'WRL'}

lista = ['HG724912', 'HG663007', 'HG703470', 'HG723753', 'HG722324', 'HG727133']

for vin in lista:
    img = Image.open("/home/salvador/Documents/170520w/comder/"+vin+".bmp")
    #img2 = Image.open("/home/salvador/Documents/170520w/comder/6.bmp")
    #img2 = np.array(img2)
    img = np.array(img)
    imgr = img.reshape(1, 200, 200, 3)
    #img3 = np.concatenate([[img], [img2]])


    prediction = model.predict(imgr)
    pred = np.argmax(prediction)

    #print(prediction)
    wheelPredSC = sales_codes[pred]
    print(wheelPredSC)
    image_SC = Image.open('/home/salvador/Documents/170520w/classes/'+str(wheelPredSC))

    fig = plt.figure()
    plt.subplot(121)
    plt.title('Image Feed')
    plt.imshow(img)
    plt.subplot(122)
    plt.title('Prediction '+str(wheelPredSC))
    plt.imshow(image_SC)
    plt.show()
