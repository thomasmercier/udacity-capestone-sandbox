from PIL import Image, ImageDraw
import numpy as np

outDir = 'out/'
N = 10 # number of images
w = 64 # image width
h = 64 # image height
D = 8 # dot diameter

for i in range(N):
    x = np.random.randint(w-D)
    y = np.random.randint(h-D)
    img = Image.new('L', (w,h) )
    draw = ImageDraw.Draw(img)
    draw.ellipse((x, y, x+D, y+D), fill=255)
    del draw
    imgFile = outDir + 'images/' + str(i) + '.png'
    img.save(imgFile, 'PNG')
    labelFile = outDir + 'labels/' + str(i) + '.txt'
    np.savetxt(labelFile, (x,y), fmt='%i' )
