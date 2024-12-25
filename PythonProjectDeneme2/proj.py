import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt


def load_data(image_dir, target_size=(128, 128)):
    images, masks = [], []
    image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') and '_mask' not in f])
    mask_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('_mask.jpg')])

    for img_name, mask_name in zip(image_filenames, mask_filenames):
        img = load_img(os.path.join(image_dir, img_name), target_size=target_size)
        img = img_to_array(img) / 255.0
        images.append(img)

        mask = load_img(os.path.join(image_dir, mask_name), target_size=target_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0
        masks.append(mask)

    return np.array(images), np.array(masks)


train_dir = 'train/'
test_dir = 'test/'


X_train, y_train = load_data(train_dir)
X_test, y_test = load_data(test_dir)


print(f"Train images shape: {X_train.shape}, Train masks shape: {y_train.shape}")
print(f"Test images shape: {X_test.shape}, Test masks shape: {y_test.shape}")


def build_unet(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u4 = layers.UpSampling2D((2, 2))(c5)
    u4 = layers.Concatenate()([u4, c4])
    d4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u4)
    d4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(d4)

    u3 = layers.UpSampling2D((2, 2))(d4)
    u3 = layers.Concatenate()([u3, c3])
    d3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u3)
    d3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(d3)

    u2 = layers.UpSampling2D((2, 2))(d3)
    u2 = layers.Concatenate()([u2, c2])
    d2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    d2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(d2)

    u1 = layers.UpSampling2D((2, 2))(d2)
    u1 = layers.Concatenate()([u1, c1])
    d1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u1)
    d1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(d1)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d1)

    return models.Model(inputs, outputs)

input_shape = (128, 128, 3)
model = build_unet(input_shape)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


epochs = 250
batch_size = 4
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=epochs, batch_size=batch_size)


eval_results = model.evaluate(X_test, y_test, batch_size=batch_size)
print(f"Test Loss: {eval_results[0]}, Accuracy: {eval_results[1]}")

# Visualize predictions
def plot_sample(X, y, preds, ix=None):
    if ix is None:
        ix = np.random.randint(0, len(X))

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(X[ix])
    ax[0].set_title('Input Image')

    ax[1].imshow(y[ix].squeeze(), cmap='gray')
    ax[1].set_title('Ground Truth')

    ax[2].imshow(preds[ix].squeeze(), cmap='gray')
    ax[2].set_title('Prediction')
    plt.show()



# Generate predictions
preds = model.predict(X_test)
preds_binary = (preds > 0.5).astype(np.uint8)

for i in range(len(preds_binary)):
    plot_sample(X_test, y_test, preds_binary, ix=i)
