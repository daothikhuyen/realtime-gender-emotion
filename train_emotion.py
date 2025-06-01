# Import necessary libraries
import numpy as np
import os
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.regularizers import l2



# Define dataset directory paths
dataset_path = "./archive"
train_dir = os.path.join(dataset_path, "train")
test_dir = os.path.join(dataset_path, "test")

# Define Data Augmentation with mild transformations
train_datagen = ImageDataGenerator(
    brightness_range=[0.8, 1.2],
    rescale=1./255,             # Chuẩn hóa giá trị pixel từ [0, 255] -> [0, 1]
    rotation_range=10,          # Xoay ảnh ngẫu nhiên trong khoảng ±10 độ
    width_shift_range=0.1,      # Dịch ảnh theo chiều ngang tối đa 10% kích thước
    height_shift_range=0.1,     # Dịch ảnh theo chiều dọc tối đa 10% kích thước
    zoom_range=0.1,             # Phóng to/thu nhỏ ảnh trong khoảng ±10%
    horizontal_flip=True,        # Lật ngang ảnh ngẫu nhiên
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators without 'workers' (Handled by Keras automatically)
train_generator = train_datagen.flow_from_directory(
    train_dir,                 # Đường dẫn thư mục chứa ảnh huấn luyện
    target_size=(48, 48),      # Resize ảnh về kích thước 48x48 pixel
    color_mode="grayscale",    # Ảnh đầu vào là ảnh xám (1 kênh màu)
    batch_size=64,             # Số lượng ảnh trong mỗi batch
    class_mode="categorical",   # Nhãn dưới dạng one-hot encoding
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=64,
    class_mode="categorical"
)

# Compute class weights to handle imbalanced dataset
train_classes = np.array(train_generator.classes)
class_labels = np.unique(train_classes)
class_weights = compute_class_weight('balanced', classes=class_labels, y=train_classes)
class_weights = np.clip(class_weights, 0.5, 2.0)  # Lower max cap to 2.0 to prevent extreme bias
class_weights_dict = dict(enumerate(class_weights))

print("Final Class Weights:", class_weights_dict)

# Define an improved CNN architecture
model = Sequential([
    Input(shape=(48,48,1)),

    Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    MaxPooling2D(pool_size=(2,2)),

    Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    BatchNormalization(),
    Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(1e-4)),
    MaxPooling2D(pool_size=(2,2)),

    Flatten(),
    Dense(512, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(1e-4)),
    Dropout(0.3),
    Dense(6, activation='softmax')  # 6 Emotion Classes
])



# Huấn luyện mô hình trong 50 epochs
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=50,
    class_weight=class_weights_dict,
    callbacks=[lr_scheduler]  
)

model.save("./model/fer2013_cnn_improved.h5")

# Evaluate the model on the FER-2013 test set
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# Callback: ReduceLROnPlateau để giảm learning rate khi chững lại

# Callback: EarlyStopping để dừng sớm nếu val_loss không cải thiện
# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_loss',        # Theo dõi loss trên tập validation
#     patience=6,                # Cho phép 6 lần không cải thiện trước khi dừng
#     restore_best_weights=True, # Trả về trọng số tốt nhất sau khi dừng
#     verbose=1
# )

# lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,                # Giảm LR xuống 1/2 nếu val_loss không cải thiện
#     patience=3,                # Sau 3 lần không cải thiện mới giảm
#     min_lr=1e-6,               # Không giảm thấp hơn mức này
#     verbose=1
# )

# # Huấn luyện mô hình trong tối đa 50 epochs (sẽ dừng sớm nếu không cải thiện)
# history = model.fit(
#     train_generator,
#     validation_data=test_generator,
#     epochs=50,
#     class_weight=class_weights_dict,
#     callbacks=[early_stopping, lr_scheduler]  # Gồm cả dừng sớm và giảm LR
# )

# # Lưu mô hình sau khi huấn luyện xong
# model.save("./model/fer2013_cnn_improved.h5")

# # Đánh giá mô hình trên tập test
# test_loss, test_accuracy = model.evaluate(test_generator)
# print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
# print(f"Test Loss: {test_loss:.4f}")