from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

# 1. MNIST veri setini yükle ve eğitim/test için ayır - Load MNIST dataset and allocate it for training/testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Veriyi ön işleme - Data pre-processing
# Giriş verisini [0, 1] aralığında normalleştir - Normalize input data in the range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Çıkış verisini kategorik olarak dönüştür - Convert output data categorically
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 3. Modeli oluştur - Set Model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),  # Gizli katmandaki nöron sayısını artırarak doğruluğu yükseltme şansı
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 4. Modeli derle - Model Compaile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4.5 Erken durma - Early Stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',       # Doğrulama doğruluğunu izleyin - Monitor verification accuracy
    patience=3,                   # Hedef doğruluğa ulaşamazsa 3 epoch daha devam etsin - Continue for 3 more epochs if target accuracy is not reached
    mode='max',                   # Maksimum doğruluğa ulaşmak için "max" olarak ayarlanır - Set to “max” to achieve maximum accuracy
    verbose=1,                    # Eğitimin ne zaman durduğunu görmek için - To see when training has stopped

)

# 5. Modeli eğit - Train Model
model.fit(
    x_train,
    y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# 6. Modelin doğruluğunu test et - Test the accuracy of the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest doğruluğu:', test_acc)
