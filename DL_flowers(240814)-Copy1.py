#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")


# In[2]:


import tensorflow as tf
print(tf.__version__)


# # Step 1. 모델 학습을 위한 데이터셋 준비

# In[3]:


import tensorflow_datasets as tfds

tfds.__version__


# In[4]:


# tf_flowers 데이터셋 다운로드 및 데이터셋 분할
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    name='tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,  
    as_supervised=True,
)


# In[5]:


# 각 데이터셋의 확인해보기
print(raw_train)
print(raw_validation)
print(raw_test)


# - 3차원 데이터, img크기가 전부 다르기 때문에 None으로 나타남

# In[6]:


# 학습 데이터에서 1개 샘플 가져와 img, label 구체적으로 확인
for image, label in raw_train.take(1):
    print(f'Image shape: {image.shape}')
    print(f'Label: {label}')


# # Step 2. 데이터셋을 모델에 넣을 수 있는 형태로 준비하기

# In[7]:


import tensorflow as tf
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[8]:


# raw_train안에 있는 데이터 확인을 위해 tf.data.Dataset에서 제공하는 take 함수 사용해보기
plt.figure(figsize=(10, 5))

# 라벨을 정수에서 문자열로 변환
get_label_name = metadata.features['label'].int2str

# raw_train 데이터셋에서 10개의 이미지를 가져와 시각화
for idx, (image, label) in enumerate(raw_train.take(10)):
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'label {label}: {get_label_name(label)}')
    plt.axis('off')

plt.show()


# - label들이 0~4까지 섞여있음
# - 이미지 사이즈가 전부 상이해 사이즈 통일이 필요

# In[9]:


# 이미지 정규화, 리사이징
IMG_SIZE = 160

def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1  # 픽셀 값을 [0, 1] 범위로 정규화
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


# - 'image = (image/127.5) - 1'일 경우
#     - 이미지 픽셀 값을 [-1, 1] 범위로 정규화
#     - 특정 신경망 구조나 사전학습된 모델에서 사용(VGG16, ResNet 등)
# - 'image = image / 255.0'의 경우
#     - 이미지 픽셀 값을 [0, 1]범위로 정규화
#     - 일반적으로 널리 사용

# In[10]:


# 학습, 검증, 테스트 데이터셋에 format_example 함수 적용
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

print(train)
print(validation)
print(test)


# In[11]:


# 학습 데이터셋에서 일부 이미지를 시각적으로 확인
plt.figure(figsize=(10, 5))

# 학습 데이터셋에서 10개의 이미지를 가져와 확인
for idx, (image, label) in enumerate(train.take(10)):
    plt.subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title(f'Label {label}')
    plt.axis('off')

plt.show()


# - WARNING은 imshow 함수가 이미지 데이터를 시각화할 때 값이 RGB 데이터의 유효 범위 벗어났기 때문
# - 이미지 데이터 유호범위는 float형일 때 [0, 1], int형일 때 [0, 255]
# - 현재 이미지를 [-1, 1]로 정규화했기 때문에 imshow가 이 값을 벗어난 값들을 [0, 1]사이로 클리핑하고 있다는 경고
# - VGG16 모델 사용 예정이기 때문에 ignore

# # Step 3. 모델 설계하기

# In[12]:


# 모델 생성에 필요한 함수들 가져오기
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D


# In[13]:


# CNN 모델 설계
# 출력층 유닛 수 5로 설정, 입력크기는 (160, 160, 3)대신 코드 유연성을 위해 변수로 입력 크기 지정
model = Sequential([
    Conv2D(filters=16, kernel_size=3, padding='same', activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(),
    Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(units=512, activation='relu'),
    Dense(units=5, activation='softmax')
])


# In[14]:


# 모델 구조 확인
model.summary()


# # Step 4. 모델 학습시키기

# In[15]:


# 학습률 설정
learning_rate = 0.0001

# 모델 컴파일: optimizer, loss, metrics setting
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])


# -  클래스 레이블이 정수형이므로 sparse_categorical_crossentropy 사용해야 함

# In[16]:


# 데이터 셔플링 및 배치화
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000


# In[17]:


train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# In[19]:


## trian_batches에서 하나의 batch만 꺼내 데이터 확인해보기
# for image_batch, label_batch in train_batches.take(1):
#     break

# image_batch.shape, label_batch.shape


# In[20]:


# 학습 데이터셋에서 한 배치를 가져와 형태 확인
for image_batch, label_batch in train_batches.take(1):
    print("Image batch shape:", image_batch.shape)
    print("Label batch shape:", label_batch.shape)


# In[21]:


# 모델 compile 후 초기 성능 확인
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# In[22]:


# epochs 10으로 학습시켜 정확도 변화 확인
EPOCHS = 10
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# # Step 5. 모델 성능 평가하기

# In[23]:


# 학습 결과 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# In[24]:


# 모델 예측 결과 확인
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    break

predictions


# In[26]:


# prediction값들을 실제 추론한 라벨로 변환해보기
import numpy as np

predictions = np.argmax(predictions, axis=1)
predictions


# In[27]:


# 32장의 이미지, 라벨, 예측 결과 시각화
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2  # 이미지 정규화 해제
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label.numpy()} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')

plt.show()


# In[28]:


# 위 32개 이미지에 대해선 prediction accuracy 계산
count = 0   # 정답을 맞춘 개수
for image, label, prediction in zip(images, labels, predictions):
    correct = label == prediction
    if correct:
        count += 1

print(f"Prediction accuracy for the batch: {count / 32 * 100:.2f}%")


# - 여기까진 한 배치(32개 이미지)에 대해 모델이 예측한 결과를 시각화하고 해당 배치에 대한 예측 정확도 계산
# - 다음 코드는 전체 데이터셋에 대해 모델의 성능(loss, accuracy) 평가해보고, 특정 배치에 대해 예측된 결과를 시각화하고 예측된 라벨 확인

# In[29]:


# 전체 테스트 데이터셋에서 모델 성능 평가
test_loss, test_accuracy = model.evaluate(test_batches)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")


# In[30]:


# 테스트 데이터셋에서 새로운 샘플 배치를 가져와 예측 결과 확인
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    predictions = tf.argmax(predictions, axis=1)
    break


# In[31]:


# 새로운 배치의 32개 이미지, 라벨, 예측 결과 시각화
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2  # 이미지 정규화 해제
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label.numpy()} / pred :{prediction.numpy()}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')

plt.show()


# In[32]:


# 다른 배치 확인해보기: 테스트 데이터셋에서 두 번째 배치를 가져와 예측 결과 확인
for image_batch, label_batch in test_batches.skip(1).take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    predictions = tf.argmax(predictions, axis=1)
    break


# In[33]:


plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2  # 이미지 정규화 해제
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label.numpy()} / pred :{prediction.numpy()}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')

plt.show()


# # Step 6. 학습된 모델 활용하기

# In[34]:


# VGG16 모델 base_model의 변수로 불러오기
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model VGG16
base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                         include_top=False,
                                         weights='imagenet')


# In[35]:


# image_batch 원래 사이즈 재확인
image_batch.shape


# In[36]:


# 모델에 이 배치를 넣으면 shape는
feature_batch = base_model(image_batch)
feature_batch.shape


# In[37]:


# 모델 구조 살펴보기
base_model.summary()


# In[38]:


# Global Average Pooling 계층 만드는 코드
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()


# In[39]:


# Global Average Pooling 적용 및 출력 크기 확인
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)


# In[41]:


# Dense Layer와 분류 레이어 생성
dense_layer = tf.keras.layers.Dense(units=512, activation='relu')
prediction_layer = tf.keras.layers.Dense(units=5, activation='softmax')
prediction_batch = prediction_layer(dense_layer(feature_batch_average))  
print(prediction_batch.shape)


# In[42]:


# VGG16 기본 모델 가중치 고정
base_model.trainable = False


# In[43]:


# 최종 모델 생성
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  dense_layer,
  prediction_layer
])


# In[44]:


model.summary()


# ## VGG16기반으로 구성된 이미지 분류기 학습시키고 결과 비교하기

# In[45]:


# 모델 정의
# 앞에서 설정한 base_model(VGG16), Global Average Pooling Layer, Dense Layer, predict Layer를
# 하나의 sequential 모델로 구성
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    dense_layer,
    prediction_layer
])


# In[46]:


# 모델이 학습하는데 필요한 학습률 파라미터 설정
learning_rate = 0.0001

# VGG16 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[47]:


# 초기 성능 평가
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))


# In[48]:


# 최종 성능 확인을 위해 모델 학습: 데이터 셔플링 및 배치화
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)


# In[49]:


# 모델 학습
EPOCHS = 5
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# In[50]:


# 학습 과정 시각화

# history 객체에서 저장된 학습 및 검증 정확도와 손실 값 불러오기
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))  # 에포크 범위 설정

# 학습 및 검증 정확도 시각화
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# 학습 및 검증 손실 시각화
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# In[51]:


# 테스트 배치에서 예측 수행
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    pass

predictions


# In[52]:


# 예측 확률을 label로 변환
predictions = np.argmax(predictions, axis=1)
predictions


# In[53]:


# 예측 결과 시각화
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2  # 정규화 해제
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')


# In[54]:


# 32개에 대한 정확도 계산
count = 0
for image, label, prediction in zip(images, labels, predictions):
    correct = label == prediction
    if correct:
        count = count + 1

print(count / 32 * 100)


# # 정확도 올리기 위한 추가 시도

# ## 1st. epochs 조정

# In[60]:


# 에포크 증가
learning_rate = 0.00001
EPOCHS = 10

# VGG16 모델 컴파일
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 모델 학습
history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# In[61]:


# 학습 과정 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# In[62]:


# 32개 샘플에 대한 예측 수행 및 정확도 계산
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(image_batch)
    break

predictions = np.argmax(predictions, axis=1)


# In[63]:


# 예측 결과 시각화 및 정확도 계산
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2  # 이미지 정규화 해제
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label.numpy()} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')

plt.show()


# In[64]:


# 32개에 대한 정확도 계산
count = 0
for image, label, prediction in zip(images, labels, predictions):
    if label == prediction:
        count += 1

print(f"Accuracy for the batch: {count / 32 * 100:.2f}%")


# ## 2nd. Fine-tuning 추가

# In[65]:


# VGG16 모델의 마지막 4개의 레이어를 학습 가능하게 설정
for layer in base_model.layers[-4:]:
    layer.trainable = True

# 미세 조정을 위해 학습률 낮게 설정
learning_rate = 1e-5

# model compile
model.compile(
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Fine-Tuning 모델 학습
EPOCHS = 10

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# In[66]:


# 학습 과정 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# In[67]:


# # 32개 샘플에 대한 예측 수행 및 정확도 계산
# for image_batch, label_batch in test_batches.take(1):
#     images = image_batch
#     labels = label_batch
#     predictions = model.predict(image_batch)
#     break

# predictions = np.argmax(predictions, axis=1)


# In[68]:


# # 예측 결과 시각화 및 정확도 계산
# plt.figure(figsize=(20, 12))

# for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
#     plt.subplot(4, 8, idx+1)
#     image = (image + 1) / 2  # 이미지 정규화 해제
#     plt.imshow(image)
#     correct = label == prediction
#     title = f'real: {label.numpy()} / pred :{prediction}\n {correct}!'
#     if not correct:
#         plt.title(title, fontdict={'color': 'red'})
#     else:
#         plt.title(title, fontdict={'color': 'blue'})
#     plt.axis('off')

# plt.show()


# In[69]:


# # 32개에 대한 정확도 계산
# count = 0
# for image, label, prediction in zip(images, labels, predictions):
#     if label == prediction:
#         count += 1

# print(f"Accuracy for the batch: {count / 32 * 100:.2f}%")


# ### WARNING 피하기 위해 model.predict 루프 밖으로 이동하고 예측 한 번에 수행하도록 수정

# In[70]:


# 32개 샘플에 대한 예측 수행: 루프 밖에서 한 번에 수행하도록
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(images)
    predictions = np.argmax(predictions, axis=1)


# In[71]:


# 예측 결과 시각화 및 정확도 계산
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2  # 이미지 정규화 해제
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label.numpy()} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')

plt.show()


# In[72]:


# 32개에 대한 정확도 계산
count = 0
for image, label, prediction in zip(images, labels, predictions):
    if label == prediction:
        count += 1

print(f"Accuracy for the batch: {count / 32 * 100:.2f}%")


# - 동일한 87.50%. 결국 큰 차이 없는 걸로...

# # 3rd. epochs 수 추가 조정
# ## epochs 10 -> 20, early stopping 적용

# In[73]:


from tensorflow.keras.callbacks import EarlyStopping

# early stopping 적용
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Fine-Tuning 모델 학습 + epochs 수 증가
EPOCHS = 20

history = model.fit(train_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches,
                    callbacks=[early_stopping])


# In[74]:


# 학습 과정 시각화
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()


# In[75]:


# 32개 샘플에 대한 예측 수행 및 정확도 계산
for image_batch, label_batch in test_batches.take(1):
    images = image_batch
    labels = label_batch
    predictions = model.predict(images)
    predictions = np.argmax(predictions, axis=1)


# In[76]:


# 예측 결과 시각화 및 정확도 계산
plt.figure(figsize=(20, 12))

for idx, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
    plt.subplot(4, 8, idx+1)
    image = (image + 1) / 2
    plt.imshow(image)
    correct = label == prediction
    title = f'real: {label.numpy()} / pred :{prediction}\n {correct}!'
    if not correct:
        plt.title(title, fontdict={'color': 'red'})
    else:
        plt.title(title, fontdict={'color': 'blue'})
    plt.axis('off')

plt.show()


# In[77]:


# 32개에 대한 정확도 계산
count = 0
for image, label, prediction in zip(images, labels, predictions):
    if label == prediction:
        count += 1

print(f"Accuracy for the batch: {count / 32 * 100:.2f}%")


# In[ ]:




