from PIL import Image, ImageDraw
import numpy as np

outDir = 'out/'
N = 360 # number of images
w = 16 # image width
h = 16 # image height
D = max(h-1,w-1)

for i in range(N):
    print(i)
    theta = 2*np.pi*float(i)/float(N)
    n = ( -np.sin(theta), np.cos(theta) )
    img = Image.new('L', (w,h) )
    img_pixel_map = img.load()
    for k in range(w):
        for l in range(h):
            coord_center = ( k-float(w)/2.0+1, float(h)/2.0-l )
            color = ( np.dot(coord_center, n)*np.sqrt(2) / D + 1 ) / 2 * 255
            img_pixel_map[k,l] = int( color )
    #img.show()
    imgFile = outDir + 'images/' + str(i) + '.png'
    img.save(imgFile, 'PNG')
    labelFile = outDir + 'labels/' + str(i) + '.txt'
    np.savetxt(labelFile, [theta])
