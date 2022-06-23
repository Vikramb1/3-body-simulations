import imageio
images = []
filenames = ['frame_proj_'+str(i)+'.png' for i in range(0,100000,100)]
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('0-100000_proj_evolution.gif', images, fps = 30)