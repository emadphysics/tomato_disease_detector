
from keras.layers import Dense,BatchNormalization,GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.applications import mobilenet
from preprocessing import data_augmentation as da
from matplotlib import pyplot as plt
Mobmodel = mobilenet.MobileNet(weights='imagenet',
                               input_shape=(80, 80, 3),
                               include_top=False)
for layer in Mobmodel.layers:
    layer.trainable = False


def top_model(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = BatchNormalization()(top_model)
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation="relu")(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(512, activation="relu")(top_model)
    top_model = BatchNormalization()(top_model)
    top_model = Dense(num_classes, activation="softmax")(top_model)
    return top_model


FC_Head = top_model(Mobmodel, 10)
Mob_model = Model(inputs=Mobmodel.input, outputs=FC_Head)
Mob_model.summary()

Mob_model.compile(loss="categorical_crossentropy",
             optimizer="adam",
             metrics=['acc'])
history_Mob = Mob_model.fit(da.it_train,
                   epochs=40,
                   steps_per_epoch = da.steps,
                   validation_data = da.it_test,
                   validation_steps = da.ssteps)


def visualize_results(history):
    # Plot the accuracy and loss curves
    acc = history_Mob.history['acc']
    val_acc = history_Mob.history['val_acc']
    loss = history_Mob.history['loss']
    val_loss = history_Mob.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


visualize_results(history_Mob)