
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.layers import GlobalAveragePooling2D,Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model
from keras.optimizers import Adagrad


#图像生成器，增强数据集
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,# ((x/255)-0.5)*2  归一化到±1之间
    rotation_range=30,#图像随机旋转30度
    width_shift_range=0.2,#图像随机水平平移，最大平移值为设定值。若值为小于1的float值，则可认为是按比例平移
    height_shift_range=0.2,#图像随机垂直平移
    shear_range=0.2,# 图像随机修剪
    zoom_range=0.2, # 图像随机变焦 
    horizontal_flip=True,#图像随机水平翻转
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)




#从硬盘读入训练图片和验证图片
train_generator = train_datagen.flow_from_directory(directory='./inceptiontrain',
                                  target_size=(299,299),#Inception V3规定大小
                                  batch_size=16)
val_generator = val_datagen.flow_from_directory(directory='./inceptionval',
                                target_size=(299,299),
                                batch_size=16)
#载入inception v3模型，并丢弃全连接层
base_model = InceptionV3(weights='imagenet',include_top=False)#表示加载imagenet预训练权重，不保留顶层的全连接层


# 定义新模型，重新配置全连接层
x = base_model.output
x = GlobalAveragePooling2D()(x) # GlobalAveragePooling2D 将 MxNxC 的张量转换成 1xC 张量，C是通道数
x = Dense(1024,activation='relu')(x)
predictions = Dense(30,activation='softmax')(x)
model = Model(inputs=base_model.input,outputs=predictions)


#只训练顶层，冻结所有inception v3的卷积层
def setup_to_transfer_learning(model,base_model):
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

def setup_to_fine_tune(model,base_model):
    GAP_LAYER = 30 # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]: 
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=Adagrad(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

#训练迁移模型
setup_to_transfer_learning(model,base_model)
history_tl = model.fit_generator(generator=train_generator,
                    steps_per_epoch=800,
                    epochs=2,
                    validation_data=val_generator,
                    validation_steps=12,
                    class_weight='auto'
                    )
model.save('./inceptiontest1.h5')
setup_to_fine_tune(model,base_model)
history_ft = model.fit_generator(generator=train_generator,
                                 steps_per_epoch=800,
                                 epochs=2,
                                 validation_data=val_generator,
                                 validation_steps=1,
                                 class_weight='auto'
                                 )
model.save('./inceptiontest1.h5')

