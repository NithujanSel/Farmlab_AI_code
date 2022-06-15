from PIL import Image
im = Image.open('foto.png')#foto inladen

width, height = im.size #hoogte ne de breete halen van de foto
groen = 0
rest = 0

for pixel in im.getdata():
    if pixel >= (83, 126, 44): #Als de pixel boven deze RGB waarde is dat dan groen anders niet.
        groen += 1
    else:
        rest += 1

#Print alles om te kijken.

print("Hoogte:",height,"\nBreete:",width,'\nTotaal pixel:',width*height)
print("==========================================")
print('Groen=' + str(groen)+', rest='+str(rest))
print("==========================================")
