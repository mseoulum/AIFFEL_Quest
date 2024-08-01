#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

print(tf.__version__)
print(np.__version__)


# # Data Load & Resize

# In[2]:


from PIL import Image 
import glob
import os

print("PIL 라이브러리 import 완료!")


# In[3]:


def resize_images(img_path):
	images=glob.glob(img_path + "/*.jpg")  
    
	print(len(images), " images to be resized.")

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
	target_size=(28,28)
	for img in images:
		old_img=Image.open(img)
		new_img=old_img.resize(target_size,Image.ANTIALIAS)
		new_img.save(img, "JPEG")
    
	print(len(images), " images resized.")
	
# 가위 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서
image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/scissor"
resize_images(image_dir_path)

print("가위 이미지 resize 완료!")


# In[4]:


image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/rock"

resize_images(image_dir_path)

print("바위 이미지 resize 완료!")


# In[5]:


image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/paper"
resize_images(image_dir_path)

print("보 이미지 resize 완료!")


# # load_data

# In[6]:


import numpy as np

def load_data(img_path, number_of_data=300):  # 가위바위보 이미지 개수 총합에 주의하세요.
    # 가위 : 0, 바위 : 1, 보 : 2
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1  
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는", idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[7]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0])
print('라벨: ', y_train[0])


# # DL Network 설계

# In[12]:


import tensorflow as tf
from tensorflow import keras
import numpy as np

# 하이퍼파라미터 option 1.
n_channel_1=16
n_channel_2=32
n_dense=32
n_train_epoch=10

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()


# # DL Network 학습 시키기

# In[13]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=n_train_epoch)


# # 얼마나 잘만들었는지 확인하기(test)

# In[14]:


image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/scissor"
resize_images(image_dir_path)

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/rock"
resize_images(image_dir_path)

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/paper"
resize_images(image_dir_path)

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test"
(x_test, y_test)=load_data(image_dir_path)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화
print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[15]:


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))


# # 2nd try

# In[27]:


# 하이퍼파라미터 option 2.
n_channel_1=16
n_channel_2=32
n_dense=36
n_train_epoch=10

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()


# In[28]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

# 모델 훈련
model.fit(x_train, y_train, epochs=n_train_epoch)


# In[29]:


image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/scissor"
resize_images(image_dir_path)

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/rock"
resize_images(image_dir_path)

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test/paper"
resize_images(image_dir_path)

image_dir_path = os.getenv("HOME") + "/aiffel/rock_scissor_paper/test"
(x_test, y_test)=load_data(image_dir_path)
x_test_norm = x_test/255.0   # 입력은 0~1 사이의 값으로 정규화
print("x_test shape: {}".format(x_test.shape))
print("y_test shape: {}".format(y_test.shape))


# In[30]:


test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))


# # 3rd try: 데이터셋 추가 및 재분할
# 

# In[31]:


# 데이터 로드 및 numpy 배열로 변환
def load_images_from_folder(folder_list):
    images = []
    for folder in folder_list:
        for filename in glob.glob(folder + '/*.jpg'): 
            img = Image.open(filename).convert('RGB')
            img = img.resize((28, 28))
            img_array = np.array(img)
            images.append(img_array)
    return images

base_dir = '/aiffel/aiffel/rock_scissor_paper'
categories = {
    'paper': ['paper', 'paper_SE', 'paper_test'],
    'rock': ['rock', 'rock_SE', 'rock_test'],
    'scissor': ['scissor', 'scissor_SE', 'scissor_test']
}

all_images = {}
all_labels = []

# 각 카테고리별로 이미지 로드
for label, folders in categories.items():
    folder_paths = [os.path.join(base_dir, folder) for folder in folders]
    folder_paths.extend([os.path.join(base_dir, 'test', folder) for folder in folders])
    images = load_images_from_folder(folder_paths)
    all_images[label] = images
    all_labels.extend([label] * len(images))

# 이미지와 레이블을 numpy 배열로 변환
all_images_np = np.concatenate([np.array(all_images[label]) for label in categories])
all_labels_np = np.array(all_labels)


# In[32]:


from sklearn.model_selection import train_test_split

# 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(all_images_np, all_labels_np, test_size=0.2, random_state=42)

print("Training set size:", x_train.shape)
print("Testing set size:", x_test.shape)


# In[33]:


# 모델 생성/정의
model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    keras.layers.MaxPool2D(2, 2), 
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),  
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'), 
    keras.layers.Dense(3, activation='softmax') 
])

model.summary() 


# In[34]:


# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[37]:


from sklearn.preprocessing import LabelEncoder

# 레이블 인코더 생성
label_encoder = LabelEncoder()

# 레이블 인코딩 수행
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Before encoding:", y_train[:5])  
print("After encoding:", y_train_encoded[:5])  


# In[38]:


# 데이터 정규화
x_train_norm = x_train / 255.0
x_test_norm = x_test / 255.0


# In[39]:


# 모델 훈련
model.fit(x_train_norm, y_train_encoded, epochs=10)


# In[40]:


# 모델 평가
test_loss, test_accuracy = model.evaluate(x_test_norm, y_test_encoded, verbose=2)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


# # Visualization

# In[41]:


# 모델 훈련
history = model.fit(x_train_norm, y_train_encoded, epochs=10, validation_data=(x_test_norm, y_test_encoded))


# In[42]:


# 훈련 과정에서의 손실과 정확도 그래프 그리기
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 정확도 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# # 4th. 데이터 재추가 및 재분할

# In[43]:


# 데이터 로드 및 numpy 배열로 변환
def load_images_from_folder(folder_list):
    images = []
    for folder in folder_list:
        for filename in glob.glob(folder + '/*.jpg'):
            img = Image.open(filename).convert('RGB')
            img = img.resize((28, 28))
            img_array = np.array(img, dtype=np.float32)
            images.append(img_array)
    return images

base_dir = '/aiffel/aiffel/rock_scissor_paper'
categories = {
    'paper': ['paper', 'paper_SE', 'paper_test', 'paper_SW'],
    'rock': ['rock', 'rock_SE', 'rock_test', 'rock_SW'],
    'scissor': ['scissor', 'scissor_SE', 'scissor_test', 'scissor_SW']
}

all_images = []
all_labels = []

# 각 카테고리별로 이미지 로드
for label, folders in categories.items():
    folder_paths = [os.path.join(base_dir, folder) for folder in folders]
    folder_paths.extend([os.path.join(base_dir, 'test', folder) for folder in folders])
    images = load_images_from_folder(folder_paths)
    all_images.extend(images)
    all_labels.extend([label] * len(images))

# 이미지와 레이블을 numpy 배열로 변환
all_images_np = np.array(all_images)
all_labels_np = np.array(all_labels)

print("Loaded images shape:", all_images_np.shape)
print("Loaded labels shape:", all_labels_np.shape)


# In[44]:


# 데이터 정규화
all_images_np = all_images_np / 255.0

# 훈련 데이터와 테스트 데이터로 분할
x_train, x_test, y_train, y_test = train_test_split(all_images_np, all_labels_np, test_size=0.2, random_state=42)

print("Training set size:", x_train.shape)
print("Testing set size:", x_test.shape)


# In[45]:


# 레이블 인코더 생성
label_encoder = LabelEncoder()

# 레이블 인코딩 수행
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Before encoding:", y_train[:5]) 
print("After encoding:", y_train_encoded[:5])


# In[46]:


# 모델 생성/정의
model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(3, activation='softmax')
])

model.summary()


# In[47]:


# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[48]:


# 모델 훈련
history = model.fit(x_train, y_train_encoded, epochs=10, validation_data=(x_test, y_test_encoded))


# In[49]:


# 모델 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=2)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


# In[50]:


# 훈련 과정에서의 손실과 정확도 그래프
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# 정확도 그래프
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# # + Dropout, Batch Normalization

# In[52]:


# 드롭아웃 및 배치 정규화 적용
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(3, activation='softmax')
])

model.summary()


# In[53]:


# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[54]:


# 모델 훈련
history = model.fit(x_train, y_train_encoded, epochs=10, validation_data=(x_test, y_test_encoded))


# In[55]:


# 모델 평가
test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=2)
print(f"Test loss: {test_loss}")
print(f"Test accuracy: {test_accuracy}")


# In[56]:


# 테스트 데이터에 대한 예측 생성
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# 올바르게 분류된 샘플과 잘못 분류된 샘플의 인덱스 찾기
correct_indices = np.where(y_pred_classes == y_test_encoded)[0]
incorrect_indices = np.where(y_pred_classes != y_test_encoded)[0]

print(f"올바르게 분류된 샘플 수: {len(correct_indices)}")
print(f"잘못 분류된 샘플 수: {len(incorrect_indices)}")


# In[57]:


import matplotlib.pyplot as plt

# 올바르게 분류된 샘플 시각화
plt.figure(figsize=(10, 5))
for i, idx in enumerate(correct_indices[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx])
    plt.title(f"Pred: {y_pred_classes[idx]}, True: {y_test_encoded[idx]}")
    plt.axis('off')
plt.suptitle("Correctly classified samples")
plt.show()

# 잘못 분류된 샘플 시각화
plt.figure(figsize=(10, 5))
for i, idx in enumerate(incorrect_indices[:5]):
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx])
    plt.title(f"Pred: {y_pred_classes[idx]}, True: {y_test_encoded[idx]}")
    plt.axis('off')
plt.suptitle("Incorrectly classified samples")
plt.show()


# In[ ]:




