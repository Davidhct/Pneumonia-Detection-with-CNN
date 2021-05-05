
# Introduction:

### This project is done under Deep Learning Course 2021 at Ariel University. 

Pneumonia is a life threatening disease, which occurs in the lungs caused by either bacterial or viral infection. It can be life endangering if not acted upon in the right time and thus early diagnosis of pneumonia is vital. The aim of this paper is to automatically detect bacterial and viral pneumonia using the digital x-ray images. <br>

Because we are limited in our lost power and the amount of dataset we have we had to overcome it by using the principles of the different architectures of different networks, and allow with a small dataset a wide enough range to achieve good results with these limitations (Data Augmentation).

# Data:

Using the data from the site: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia <br>
The data set is organized into three folders (train, test, selection) and contains subfolders for each image category (pneumonia / normal). There are 5,856 X-ray (JPEG) images and 2 categories (Pneumonia / Normal).

After we downloaded the data set we changed the amount of images in each folder so that we got a new structure. <br>


<kbd><img src="/images/date_structure.png" height="350"></kbd>

#### Since we have relatively few images and in order for our model to learn as much as possible from the training data,<br> we used data augmentation as shown in the code below:<br>

```
 # Data generation objects
    train_datagen = ImageDataGenerator(rescale=1. / 255, zoom_range=0.3, shear_range=0.2, vertical_flip=True)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_gen = train_datagen.flow_from_directory(
        directory=input_path + 'train',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    val_gen = val_datagen.flow_from_directory(
        directory=input_path + 'val',
        target_size=(img_dims, img_dims),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)
```

#### And for the prediction we created the test manually without data augmentation:<br>

```
 test_data = []
    test_labels = []
    label = 0
    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for img in tqdm(os.listdir(input_path + 'test' + cond)):
            img = plt.imread(input_path + 'test' + cond + img)
            img = cv2.resize(img, (img_dims, img_dims))
            if len(img.shape) == 1:
                img = np.dstack([img, img, img])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255
            if cond == '/NORMAL/':
                label = 0
            elif cond == '/PNEUMONIA/':
                label = 1
            test_data.append(img)
            test_labels.append(label)

    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

```

