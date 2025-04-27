import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os
import json
from tqdm import tqdm

# Установить seed для воспроизводимости
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Проверить доступность CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Параметры с уменьшенными значениями для экономии памяти
batch_size = 16  # Сильно уменьшим размер батча для экономии памяти
num_epochs = 2  # Уменьшим до 2 эпох для экономии ресурсов
num_classes = 47  # EMNIST (буквы + цифры)
img_size = 112  # Уменьшенный размер изображения

# EMNIST имеет одноканальные изображения, но наши модели ожидают 3 канала
# Добавим функцию для преобразования 1 канала в 3
class GrayscaleToRGB:
    def __call__(self, x):
        return x.repeat(3, 1, 1)

# ВАЖНО: Сначала преобразуем в тензор, нормализуем одноканальное изображение,
# и ПОТОМ преобразуем в RGB
data_transforms = transforms.Compose([
    transforms.Resize(img_size),  # Уменьшенный размер
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),  # Нормализация для одноканального изображения
    GrayscaleToRGB()  # Только после нормализации преобразуем в RGB
])

# Загрузка датасета EMNIST
print("Загрузка датасета EMNIST...")
train_dataset = datasets.EMNIST(root='./data', split='balanced', train=True, 
                               download=True, transform=data_transforms)
test_dataset = datasets.EMNIST(root='./data', split='balanced', train=False, 
                              download=True, transform=data_transforms)

# Ограничим размер набора данных при нехватке памяти (опционально)
# Для тестирования можно использовать меньший набор
train_size = min(20000, len(train_dataset))
test_size = min(4000, len(test_dataset))

train_dataset = torch.utils.data.Subset(train_dataset, range(train_size))
test_dataset = torch.utils.data.Subset(test_dataset, range(test_size))

# Создание загрузчиков данных
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)  # Уменьшаем число workers
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)   # Уменьшаем число workers

print(f"Размер обучающего набора: {len(train_dataset)}")
print(f"Размер тестового набора: {len(test_dataset)}")

# Функция для обучения модели
def train_model(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Для хранения истории обучения
    history = {'train_loss': [], 'test_loss': [], 
               'train_acc': [], 'test_acc': []}
    
    for epoch in range(num_epochs):
        print(f'Эпоха {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Каждая эпоха имеет фазу обучения и валидации
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Установить модель в режим обучения
                dataloader = train_loader
            else:
                model.eval()   # Установить модель в режим оценки
                dataloader = test_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Проход по данным
            for inputs, labels in tqdm(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Обнуление градиентов параметров
                optimizer.zero_grad()
                
                # Очистим кэш CUDA перед прямым проходом
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Прямой проход
                # Включаем вычисление градиентов только в фазе обучения
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)
                    
                    # Обратное распространение + оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            # Сохраняем историю
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().item())
            else:
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc.cpu().item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Глубокая копия модели
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Обучение завершено за {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Лучшая точность на валидации: {best_acc:4f}')
    
    # Загружаем лучшие веса модели
    model.load_state_dict(best_model_wts)
    return model, history

# Инициализация облегченной модели DenseNet
def initialize_densenet():
    # Используем DenseNet121 вместо DenseNet161 для экономии памяти
    model = models.densenet121(weights=None)
    
    # Заменим последний слой на наш для классификации EMNIST
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    
    return model

print(f"\nОбучение модели: densenet")

# Инициализация модели DenseNet
model = initialize_densenet()
model = model.to(device)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение и оценка
model, history = train_model(model, criterion, optimizer, num_epochs=num_epochs)

# Сохранение результатов
model_name = "densenet"
# Сохранение модели
torch.save(model.state_dict(), f"{model_name}_emnist.pth")

# Сохраним историю обучения в JSON для дальнейшего анализа
# Преобразуем значения numpy в обычные Python типы для JSON
history_json = {
    'train_loss': [float(loss) for loss in history['train_loss']],
    'test_loss': [float(loss) for loss in history['test_loss']],
    'train_acc': [float(acc) for acc in history['train_acc']],
    'test_acc': [float(acc) for acc in history['test_acc']]
}

with open(f'{model_name}_history.json', 'w') as f:
    json.dump(history_json, f)

# Визуализация результатов обучения
plt.figure(figsize=(12, 8))

# График функции потерь
plt.subplot(2, 1, 1)
plt.plot(history['train_loss'], label='Обучение')
plt.plot(history['test_loss'], label='Валидация')
plt.title(f'Функция потерь модели {model_name}')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.legend()

# График точности
plt.subplot(2, 1, 2)
plt.plot(history['train_acc'], label='Обучение')
plt.plot(history['test_acc'], label='Валидация')
plt.title(f'Точность модели {model_name}')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

plt.tight_layout()
plt.savefig(f'{model_name}_results.png')
plt.show()

print(f"\nРезультаты для модели {model_name}:")
print(f"Финальная потеря (трейн): {history['train_loss'][-1]:.4f}")
print(f"Финальная потеря (тест): {history['test_loss'][-1]:.4f}")
print(f"Финальная точность (трейн): {history['train_acc'][-1]:.4f}")
print(f"Финальная точность (тест): {history['test_acc'][-1]:.4f}")

print(f"\nОбучение модели {model_name} завершено. Результаты сохранены в {model_name}_results.png и {model_name}_history.json")
