import matplotlib.pyplot as plt
import numpy as np

height = 863
xOffset = 163.532
yOffset = height - 642.999

def colourToPix(colour):
    conversion = 1/89.895
    x = (colour/conversion)+xOffset
    return x

def magToPix(mag,d):
    M = mag + 5 - 5*np.log10(d)
    conversion = 5/166.537
    y = (M/conversion)+yOffset
    return y

path = 'Other Figures/'

stars = [{'Name':'4987', 'Colour':0.446, 'Mag':11.3431, 'Distance':1790.36},
         {'Name':'14535', 'Colour':0.781, 'Mag':12.5173, 'Distance':714.01},
         {'Name':'15284', 'Colour':1.065, 'Mag':13.5912, 'Distance':5549.87},
         {'Name':'17749', 'Colour':0.913, 'Mag':12.7905, 'Distance':1583.53}]
plotColours = ['r','b','g','k']

img = plt.imread(path+"HRDiagramOriginal.png")
fig, ax = plt.subplots()
ax.imshow(img)

for i, star in enumerate(stars):
    ax.plot(colourToPix(star['Colour']), magToPix(star['Mag'], star['Distance']), plotColours[i]+'.',
            markersize=3, label=star['Name'])

plt.axis('off')
plt.legend(fontsize='x-small')

plt.savefig(path+'HRDiagram.png', bbox_inches='tight', pad_inches=0, dpi=300)
plt.savefig(path+'HRDiagram.pdf', bbox_inches='tight', pad_inches=0, dpi=300)