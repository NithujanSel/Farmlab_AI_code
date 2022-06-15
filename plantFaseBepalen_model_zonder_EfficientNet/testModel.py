from keras.models import load_model #Om de ML model te laden
import numpy as np #om beter met array te werken
import cv2 #Om fotos te lezen en manuplueren


model = load_model("./plantModel4fase.h5") #model in laden

foto = "./VIS_SV_180_z3500_321320.png" #De foto inladen om de model te testen

#Gaat de foto om zetten naar 224x244 grotte.
def fotoResize(PATHF,IMGSIZE=224):
    img_Array = cv2.imread(PATHF)
    img_Array = cv2.resize(img_Array,(IMGSIZE,IMGSIZE))
    img_Array = np.expand_dims(img_Array, axis=0)

    return img_Array



foftX = fotoResize(foto) #Uitvoering van fotoResize functie

foftX = np.array(foftX) #Omzetten naar np array


print(foftX.shape) #Kijken of de foto goed is


# perdict = np.argmax(model.predict(foftX), axis=-1)
x = model.predict([foftX])[0] #de foto perdicten welke fase het is
print(x)
perdict = np.argmax(x,axis=-1) #Heeft een array aleen de juiste fase pakken


#perdicte fase tonen
if perdict == 0:
    print("Fase 1")
elif perdict == 1:
    print("Fase 2")
elif perdict == 2:
    print("Fase 3")
elif perdict == 3:
    print("Fase 4")
else:
    print("ERROR")
    
    