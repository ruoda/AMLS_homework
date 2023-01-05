import cv2
from sklearn.model_selection import train_test_split
import numpy as np
import os
import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation
from keras.models import Sequential

def main():
    # Check if GPU is available
    if tf.test.is_gpu_available():
        # Use GPU
        with tf.device('/device:GPU:0'):
            # Read labels
            images = []
            labels = []
            with open('../Datasets/celeba/labels.csv') as f:
                f.readline()  # Skip header
                for line in f:
                    idx, name, label1, label2 = line.strip().split('\t')
                    # Read image
                    image = cv2.imread('../Datasets/celeba/img/'+name)
                    # Resize image
                    #image = cv2.resize(image, (150, 150))
                    # Convert channels to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    images.append(image)
                    if int(label2) == -1:
                        labels.append(0)
                    else:
                        labels.append(1)
                    #print(name,label1)

            # Split data into train, val, and test sets
            x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
            x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
            # Convert list to array
            x_train = np.array(x_train)
            x_val = np.array(x_val)
            x_test = np.array(x_test)
            y_train = np.array(y_train)
            y_val = np.array(y_val)
            y_test = np.array(y_test)

            # Preprocessing for training data
            train_datagen = ImageDataGenerator(rescale=1./255)

            # Preprocessing for validation and test data
            val_datagen = ImageDataGenerator(rescale=1./255)

            # Load training data
            train_generator = train_datagen.flow(x_train, y_train, batch_size=20)

            # Load validation data
            val_generator = val_datagen.flow(x_val, y_val, batch_size=20)

            # Load test data
            test_generator = val_datagen.flow(x_test, y_test, batch_size=20)

            # # Build model
            model = Sequential()
            # 添加卷积层
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(218, 178, 3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))

            # 添加全连接层
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))

            
            model.summary()  #打印出模型概况，它实际调用的是keras.utils.print_summary

            model.compile(optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07, decay=0.0),
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit_generator(
                train_generator,
                steps_per_epoch=100,
                epochs=30,
                validation_data=val_generator,
                validation_steps=50)
            if os.path.exists('./save_model/model.h5'):
                pass
            else:
                os.makedirs('./save_model')
                model.save('./save_model/model.h5')
            # Evaluate model
            score = model.evaluate_generator(test_generator, steps=50)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

if __name__=='__main__':
    main()