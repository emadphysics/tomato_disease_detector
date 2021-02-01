from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from preprocessing import extracting

path='PlantVillage\\diseases'
a=extracting.processing(path,(80,80),10)

trainX, testX, trainY, testY = train_test_split(a.split()[0], a.split()[1] , test_size=0.2, random_state = 42)
batch=32
datagen = ImageDataGenerator(rescale = 1./255,
                                   rotation_range=25,
                                   height_shift_range = 0.3,
                                   width_shift_range = 0.3,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = "nearest")
test_datagen = ImageDataGenerator(rescale = 1./255)
it_train = datagen.flow(trainX,trainY, batch_size=batch)
it_test = datagen.flow(testX, testY, batch_size=batch)
steps = int(trainX[0].shape[0] /batch)
ssteps=int(testX[1].shape[0] / batch)

