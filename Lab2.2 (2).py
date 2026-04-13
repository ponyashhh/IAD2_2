import numpy as np
import matplotlib.pyplot as plt
import time
import os
import urllib.request
import zipfile
from PIL import Image

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1.0 - np.tanh(x)**2

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def d_elu(x, alpha=1.0):
    return np.where(x > 0, 1.0, alpha * np.exp(x))

def softmax(x):
    # Додано віднімання максимуму для числової стабільності
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def get_activation(name):
    activations = {
        'relu': (relu, d_relu),
        'tanh': (tanh, d_tanh),
        'leaky_relu': (leaky_relu, d_leaky_relu),
        # PReLU в межах NumPy зазвичай реалізують як LeakyReLU з більшим alpha,
        'prelu': (lambda x: leaky_relu(x, alpha=0.25), lambda x: d_leaky_relu(x, alpha=0.25)),
        'elu': (elu, d_elu)
    }
    return activations.get(name)

class MLP(object):
    def __init__(self, layer_sizes, activation_names, lr=0.01):
        self.layer_sizes = layer_sizes
        self.lr = lr
        self.W = []
        self.b = []
        self.acts = []
        self.d_acts = []
        
        for i in range(len(layer_sizes) - 1):
            # Виправлення ініціалізації: Використання He Initialization (Xe)
            fan_in = layer_sizes[i]
            std_dev = np.sqrt(2.0 / fan_in)
            self.W.append(np.random.randn(layer_sizes[i], layer_sizes[i+1]) * std_dev)
            self.b.append(np.zeros((1, layer_sizes[i+1])))
            
        for name in activation_names:
            act, d_act = get_activation(name)
            self.acts.append(act)
            self.d_acts.append(d_act)

    def forward(self, X):
        self.A = [X]
        self.Z = []
        
        for i in range(len(self.W) - 1):
            Z_curr = np.dot(self.A[-1], self.W[i]) + self.b[i]
            self.Z.append(Z_curr)
            A_curr = self.acts[i](Z_curr)
            self.A.append(A_curr)
            
        Z_out = np.dot(self.A[-1], self.W[-1]) + self.b[-1]
        self.Z.append(Z_out)
        A_out = softmax(Z_out)
        self.A.append(A_out)
        
        return A_out

    def compute_loss(self, Y_pred, Y_true):
        m = Y_true.shape[0]
        log_preds = np.log(Y_pred + 1e-8)
        return -np.sum(Y_true * log_preds) / m

    def backward(self, Y_true):
        m = Y_true.shape[0]
        dW = [np.zeros_like(w) for w in self.W]
        db = [np.zeros_like(b) for b in self.b]
        
        dZ = self.A[-1] - Y_true
        dW[-1] = np.dot(self.A[-2].T, dZ) / m
        db[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        
        for i in reversed(range(len(self.W) - 1)):
            dA = np.dot(dZ, self.W[i+1].T)
            dZ = dA * self.d_acts[i](self.Z[i])
            dW[i] = np.dot(self.A[i].T, dZ) / m
            db[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
        for i in range(len(self.W)):
            self.W[i] -= self.lr * dW[i]
            self.b[i] -= self.lr * db[i]

    def train(self, X_train, Y_train, X_val, Y_val, epochs=50, batch_size=32):
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
        m = X_train.shape[0]
        
        for epoch in range(epochs):
            # Виправлення навчання: Реалізація градієнтного спуску за міні-батчами
            indices = np.arange(m)
            np.random.shuffle(indices) # Перемішування даних перед кожною епохою
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                Y_batch = Y_shuffled[i:i+batch_size]
                
                # Прямий і зворотний прохід тільки для міні-батчу
                self.forward(X_batch)
                self.backward(Y_batch)
                
            # Оцінка метрик в кінці епохи на всій вибірці
            Y_pred_train = self.forward(X_train)
            loss = self.compute_loss(Y_pred_train, Y_train)
            acc = np.mean(np.argmax(Y_pred_train, axis=1) == np.argmax(Y_train, axis=1))
            
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(val_pred, Y_val)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(Y_val, axis=1))
            
            history['loss'].append(loss)
            history['val_loss'].append(val_loss)
            history['acc'].append(acc)
            history['val_acc'].append(val_acc)
            
        return history

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

def download_omniglot(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    dataset_dir = os.path.join(base_path, "images_background")
    if os.path.exists(dataset_dir) and len(os.listdir(dataset_dir)) > 0:
        return dataset_dir
        
    url = "https://raw.githubusercontent.com/brendenlake/omniglot/master/python/images_background.zip"
    zip_path = os.path.join(base_path, "images_background.zip")
    
    print(f"Завантаження Omniglot з {url}...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Розпакування архіву...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(base_path)
        
    os.remove(zip_path)
    return dataset_dir

def load_omniglot_data(base_path="omniglot_data", img_size=28, max_classes=30, max_samples_per_class=20):
    dataset_dir = download_omniglot(base_path)
    
    X = []
    Y_labels = []
    class_idx = 0

    for alphabet in sorted(os.listdir(dataset_dir)): # Сортування для стабільності
        alphabet_path = os.path.join(dataset_dir, alphabet)
        if not os.path.isdir(alphabet_path): continue
        
        for character in sorted(os.listdir(alphabet_path)):
            char_path = os.path.join(alphabet_path, character)
            if not os.path.isdir(char_path): continue
            
            samples_loaded = 0
            for img_name in os.listdir(char_path):
                if samples_loaded >= max_samples_per_class: break
                img_path = os.path.join(char_path, img_name)
                try:
                    img = Image.open(img_path).convert('L').resize((img_size, img_size))
                    img_array = 1.0 - (np.array(img).flatten() / 255.0)
                    X.append(img_array)
                    Y_labels.append(class_idx)
                    samples_loaded += 1
                except:
                    pass
            class_idx += 1
            if class_idx >= max_classes:
                break
        if class_idx >= max_classes:
            break
            
    X = np.array(X)
    Y_labels = np.array(Y_labels)
    Y = np.eye(class_idx)[Y_labels]
    
    return X, Y, Y_labels, class_idx

def split_data(X, Y, Y_labels, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    
    t_idx, v_idx = indices[:split_idx], indices[split_idx:]
    return X[t_idx], Y[t_idx], Y_labels[t_idx], X[v_idx], Y[v_idx], Y_labels[v_idx]

def plot_metrics(histories, titles):
    fig, axes = plt.subplots(2, len(histories), figsize=(5 * len(histories), 8))
    if len(histories) == 1:
        axes = np.expand_dims(axes, axis=1)
        
    for i, (history, title) in enumerate(zip(histories, titles)):
        axes[0, i].plot(history['loss'], label='Train Loss')
        axes[0, i].plot(history['val_loss'], label='Val Loss')
        axes[0, i].set_title(f'{title}\nLoss')
        axes[0, i].legend()
        axes[0, i].grid(True, linestyle='--', alpha=0.6)
        
        axes[1, i].plot(history['acc'], label='Train Acc')
        axes[1, i].plot(history['val_acc'], label='Val Acc')
        axes[1, i].set_title(f'{title}\nAccuracy')
        axes[1, i].legend()
        axes[1, i].grid(True, linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.show()

def show_misclassified(X_val, Y_val_labels, predictions, img_size=28, num_images=5):
    incorrect_indices = np.where(predictions != Y_val_labels)[0]
    
    if len(incorrect_indices) == 0:
        print("Усі зображення класифіковано правильно!")
        return
        
    num_images = min(num_images, len(incorrect_indices))
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    
    if num_images == 1:
        axes = [axes]
        
    for i in range(num_images):
        idx = incorrect_indices[i]
        img = X_val[idx].reshape(img_size, img_size)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {Y_val_labels[idx]}\nPred: {predictions[idx]}")
        axes[i].axis('off')
        
    plt.show()

if __name__ == "__main__":
    # Збільшив max_classes до 30, щоб датасет був змістовнішим (600 зображень)
    X, Y, Y_labels, num_classes = load_omniglot_data("omniglot_data", img_size=28, max_classes=30, max_samples_per_class=20)
    print(f"Завантажено {X.shape[0]} зображень, {num_classes} класів.")
    
    X_train, Y_train, Y_train_labels, X_val, Y_val, Y_val_labels = split_data(X, Y, Y_labels, test_size=0.2)
    print(f"Тренувальна вибірка: {X_train.shape[0]}, Перевірочна: {X_val.shape[0]}")
    
    input_size = X.shape[1]
    output_size = Y.shape[1]
    epochs_count = 50 
    batch_size = 32 # Додано розмір міні-батчу
    
    # Злегка зменшив lr, оскільки при міні-батчах оновлення ваг відбувається частіше
    models_configs = [
        {"name": "Base (1 Hid, ReLU)", "layers": [input_size, 128, output_size], "acts": ['relu'], "lr": 0.05},
        {"name": "Deep (2 Hid, Tanh)", "layers": [input_size, 128, 64, output_size], "acts": ['tanh', 'tanh'], "lr": 0.01},
        {"name": "Deep (2 Hid, LeakyReLU)", "layers": [input_size, 128, 64, output_size], "acts": ['leaky_relu', 'leaky_relu'], "lr": 0.01},
        {"name": "Deep (2 Hid, PReLU)", "layers": [input_size, 128, 64, output_size], "acts": ['prelu', 'prelu'], "lr": 0.01},
        {"name": "Deep (2 Hid, ELU)", "layers": [input_size, 128, 64, output_size], "acts": ['elu', 'elu'], "lr": 0.01}
    ]
    
    histories = []
    titles = []
    trained_models = []
    results_summary = []
    
    print("="*60)
    print("ПОЧАТОК НАВЧАННЯ ТА ТЕСТУВАННЯ МОДЕЛЕЙ")
    print("="*60)
    
    for config in models_configs:
        model = MLP(layer_sizes=config["layers"], activation_names=config["acts"], lr=config["lr"])
        
        start_train = time.time()
        # Передаємо batch_size
        history = model.train(X_train, Y_train, X_val, Y_val, epochs=epochs_count, batch_size=batch_size)
        train_time = time.time() - start_train
        
        start_pred = time.time()
        val_predictions = model.predict(X_val)
        pred_time = time.time() - start_pred
        
        histories.append(history)
        titles.append(config["name"])
        trained_models.append(model)
        
        results_summary.append({
            "name": config["name"],
            "train_time": train_time,
            "pred_time": pred_time,
            "val_acc": history['val_acc'][-1]
        })
        
        print(f"Модель [{config['name']}] успішно навчена (Точність: {history['val_acc'][-1]:.4f}).")
        
    print("\n" + "="*60)
    print("ПОРІВНЯЛЬНИЙ АНАЛІЗ РЕЗУЛЬТАТІВ")
    print("="*60)
    
    for res in results_summary:
        print(f"Модель: {res['name']}")
        print(f"  - Час навчання (50 епох): {res['train_time']:.4f} сек")
        print(f"  - Час надання прогнозу:   {res['pred_time']:.4f} сек")
        print(f"  - Фінальна точність:      {res['val_acc']:.4f}")
        print("-" * 60)
        
    plot_metrics(histories, titles)
    
    best_model = trained_models[0]
    best_predictions = best_model.predict(X_val)
    show_misclassified(X_val, Y_val_labels, best_predictions)