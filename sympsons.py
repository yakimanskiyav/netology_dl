import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# Устанавливаем seed для воспроизводимости
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Проверяем доступность CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Создаем класс для датасета Симпсонов
class SimpsonsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.image_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Функция для обучения модели с разными scheduler'ами
def train_model(model, criterion, optimizer, scheduler, dataloaders, device, num_epochs=10, scheduler_name=""):
    start_time = time.time()
    
    # Для отслеживания лучшей точности модели
    best_acc = 0.0
    
    # История обучения
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    for epoch in range(num_epochs):
        print(f'Эпоха {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Каждая эпоха имеет фазу обучения и валидации
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Установить модель в режим обучения
            else:
                model.eval()   # Установить модель в режим оценки
            
            running_loss = 0.0
            running_corrects = 0
            
            # Прогресс-бар для отслеживания прогресса
            bar = tqdm(dataloaders[phase])
            for inputs, labels in bar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Обнуляем градиенты параметров
                optimizer.zero_grad()
                
                # Прямой проход
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Обратный проход + оптимизация только в фазе обучения
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Статистика
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            print(f'{phase} Потери: {epoch_loss:.4f} Точность: {epoch_acc:.4f}')
            
            # Сохраняем историю
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                current_lr = optimizer.param_groups[0]['lr']
                history['lr'].append(current_lr)
                print(f'Learning Rate: {current_lr:.6f}')
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Если это лучшая модель, сохраняем веса
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
            
        # Шаг планировщика после каждой эпохи
        if scheduler_name == "ReduceLROnPlateau":
            scheduler.step(history['val_loss'][-1])
        else:
            scheduler.step()
            
        print()
    
    time_elapsed = time.time() - start_time
    print(f'Обучение завершено за {time_elapsed // 60:.0f}м {time_elapsed % 60:.0f}с')
    print(f'Лучшая валидационная точность: {best_acc:.4f}')
    
    return model, history

# Функция для визуализации результатов обучения
def plot_training_results(histories, title="Сравнение разных LR Schedulers"):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    for name, history in histories.items():
        # График точности
        axs[0, 0].plot(history['train_acc'], label=f'{name} (train)')
        axs[0, 0].plot(history['val_acc'], label=f'{name} (val)')
    axs[0, 0].set_title('Точность модели')
    axs[0, 0].set_ylabel('Точность')
    axs[0, 0].set_xlabel('Эпоха')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    for name, history in histories.items():
        # График потерь
        axs[0, 1].plot(history['train_loss'], label=f'{name} (train)')
        axs[0, 1].plot(history['val_loss'], label=f'{name} (val)')
    axs[0, 1].set_title('Потери модели')
    axs[0, 1].set_ylabel('Потери')
    axs[0, 1].set_xlabel('Эпоха')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    for name, history in histories.items():
        # График скорости обучения
        axs[1, 0].plot(history['lr'], label=name)
    axs[1, 0].set_title('Learning Rate')
    axs[1, 0].set_ylabel('LR')
    axs[1, 0].set_xlabel('Эпоха')
    axs[1, 0].legend()
    axs[1, 0].set_yscale('log')
    axs[1, 0].grid(True)
    
    # Сравнение валидационной точности
    for name, history in histories.items():
        axs[1, 1].plot(history['val_acc'], label=name)
    axs[1, 1].set_title('Сравнение валидационной точности')
    axs[1, 1].set_ylabel('Точность')
    axs[1, 1].set_xlabel('Эпоха')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.92)
    plt.savefig(f'{title.replace(" ", "_")}.png')
    plt.show()

# Основная функция
def main():
    # Пути к данным
    train_dir = 'data/the-simpsons-characters-dataset/simpsons_dataset'
    
    # Базовые трансформации для валидации
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 1. Нормальные аугментации
    normal_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Сильные аугментации, которые могут ухудшить качество
    strong_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.5),  # Сильная аугментация - вертикальный переворот
        transforms.RandomRotation(30),  # Увеличенный угол поворота
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),  # Сильное изменение цвета
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.7, 1.3)),  # Сильное искажение
        transforms.RandomPerspective(distortion_scale=0.6, p=0.5),  # Перспективные искажения
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33))  # Случайное стирание частей изображения
    ])
    
    # 3. Радикальные аугментации для сильного ухудшения качества
    radical_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),  # Всегда переворачиваем по горизонтали
        transforms.RandomVerticalFlip(p=0.8),  # Высокая вероятность вертикального переворота
        transforms.RandomRotation(180),  # Поворот на случайный угол до 180 градусов
        transforms.ColorJitter(brightness=0.9, contrast=0.9, saturation=0.9, hue=0.5),  # Экстремальное изменение цвета
        transforms.RandomAffine(
            degrees=180,  # Поворот на случайный угол до 180 градусов
            translate=(0.5, 0.5),  # Сильное смещение
            scale=(0.3, 2.0),  # Очень сильное масштабирование
            shear=45  # Значительный сдвиг
        ),
        transforms.RandomPerspective(distortion_scale=0.9, p=0.8),  # Сильные перспективные искажения
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 5.0)),  # Сильное размытие
        transforms.RandomInvert(p=0.3),  # Инвертирование цветов
        transforms.RandomSolarize(threshold=128, p=0.3),  # Соляризация
        transforms.RandomPosterize(bits=2, p=0.3),  # Постеризация (уменьшение количества бит)
        transforms.RandomEqualize(p=0.2),  # Выравнивание гистограммы
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.8, scale=(0.15, 0.6), ratio=(0.3, 3.0)),  # Значительное стирание частей изображения
    ])
    
    # Создание датасетов и даталоадеров
    print("Загрузка данных...")
    
    # 1. Эксперимент с нормальными аугментациями
    print("\nЭксперимент 1: Нормальные аугментации")
    train_dataset = SimpsonsDataset(train_dir, transform=normal_transform)
    
    # Разделение на обучающую и валидационную выборки (80/20)
    total_size = len(train_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # Применяем правильные трансформации для валидационной выборки
    val_dataset = SimpsonsDataset(train_dir, transform=val_transform)
    indices = val_subset.indices
    val_subset = torch.utils.data.Subset(val_dataset, indices)
    
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=4)
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    num_classes = len(train_dataset.classes)
    print(f"Количество классов: {num_classes}")
    
    # Создание моделей и проведение экспериментов с разными LR schedulers
    histories = {}
    
    # Базовая модель MobileNetV2 с предобученными весами
    def get_mobilenet_model():
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model.to(device)
    
    # Эксперимент 1: StepLR
    print("\nЭксперимент с StepLR scheduler")
    model_step = get_mobilenet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_step.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    
    model_step, history_step = train_model(
        model_step, criterion, optimizer, scheduler, 
        dataloaders, device, num_epochs=10, scheduler_name="StepLR"
    )
    
    histories['StepLR'] = history_step
    
    # Эксперимент 2: CosineAnnealingLR
    print("\nЭксперимент с CosineAnnealingLR scheduler")
    model_cos = get_mobilenet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cos.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    model_cos, history_cos = train_model(
        model_cos, criterion, optimizer, scheduler, 
        dataloaders, device, num_epochs=10, scheduler_name="CosineAnnealingLR"
    )
    
    histories['CosineAnnealingLR'] = history_cos
    
    # Эксперимент 3: ReduceLROnPlateau
    print("\nЭксперимент с ReduceLROnPlateau scheduler")
    model_reduce = get_mobilenet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_reduce.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    
    model_reduce, history_reduce = train_model(
        model_reduce, criterion, optimizer, scheduler, 
        dataloaders, device, num_epochs=10, scheduler_name="ReduceLROnPlateau"
    )
    
    histories['ReduceLROnPlateau'] = history_reduce
    
    # Визуализация результатов экспериментов с разными schedulers
    plot_training_results(histories, "Сравнение разных LR Schedulers")
    
    # Эксперимент с сильными аугментациями
    print("\nЭксперимент 2: Сильные аугментации")
    train_dataset_strong = SimpsonsDataset(train_dir, transform=strong_transform)
    
    train_subset_strong, _ = torch.utils.data.random_split(
        train_dataset_strong, [train_size, val_size]
    )
    
    train_loader_strong = DataLoader(train_subset_strong, batch_size=32, shuffle=True, num_workers=4)
    
    dataloaders_strong = {
        'train': train_loader_strong,
        'val': val_loader
    }
    
    # Используем лучший LR scheduler из предыдущих экспериментов
    model_strong = get_mobilenet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_strong.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    model_strong, history_strong = train_model(
        model_strong, criterion, optimizer, scheduler, 
        dataloaders_strong, device, num_epochs=10, scheduler_name="CosineAnnealingLR"
    )
    
    # Эксперимент с радикальными аугментациями
    print("\nЭксперимент 3: Радикальные аугментации")
    train_dataset_radical = SimpsonsDataset(train_dir, transform=radical_transform)
    
    train_subset_radical, _ = torch.utils.data.random_split(
        train_dataset_radical, [train_size, val_size]
    )
    
    train_loader_radical = DataLoader(train_subset_radical, batch_size=32, shuffle=True, num_workers=4)
    
    dataloaders_radical = {
        'train': train_loader_radical,
        'val': val_loader
    }
    
    # Используем лучший LR scheduler из предыдущих экспериментов
    model_radical = get_mobilenet_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_radical.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    
    model_radical, history_radical = train_model(
        model_radical, criterion, optimizer, scheduler, 
        dataloaders_radical, device, num_epochs=10, scheduler_name="CosineAnnealingLR"
    )
    
    # Сравнение результатов всех аугментаций
    augmentation_histories = {
        'Нормальные аугментации': history_cos,  # Лучший результат из предыдущих экспериментов
        'Сильные аугментации': history_strong,
        'Радикальные аугментации': history_radical
    }
    
    plot_training_results(augmentation_histories, "Сравнение разных аугментаций")
    
    # Выводы по результатам экспериментов
    print("\n--- ВЫВОДЫ ПО РЕЗУЛЬТАТАМ ЭКСПЕРИМЕНТОВ ---")
    
    # 1. Сравнение LR Schedulers
    best_scheduler = max(histories.items(), key=lambda x: max(x[1]['val_acc']))
    print(f"1. Лучший LR Scheduler: {best_scheduler[0]} с максимальной валидационной точностью {max(best_scheduler[1]['val_acc']):.4f}")
    
    for name, history in histories.items():
        print(f"   - {name}: максимальная валидационная точность {max(history['val_acc']):.4f}")
    
    # 2. Сравнение аугментаций
    normal_max_acc = max(history_cos['val_acc'])
    strong_max_acc = max(history_strong['val_acc'])
    radical_max_acc = max(history_radical['val_acc'])
    
    acc_diff_strong = normal_max_acc - strong_max_acc
    acc_diff_radical = normal_max_acc - radical_max_acc
    
    print(f"\n2. Влияние аугментаций:")
    print(f"   - Нормальные аугментации: максимальная валидационная точность {normal_max_acc:.4f}")
    print(f"   - Сильные аугментации: максимальная валидационная точность {strong_max_acc:.4f}")
    print(f"   - Радикальные аугментации: максимальная валидационная точность {radical_max_acc:.4f}")
    print(f"   - Разница (сильные): {acc_diff_strong:.4f} ({acc_diff_strong*100:.2f}%)")
    print(f"   - Разница (радикальные): {acc_diff_radical:.4f} ({acc_diff_radical*100:.2f}%)")
    
    if acc_diff_strong > 0.1:
        print("   - Сильные аугментации значительно ухудшили качество модели (более 10%)")
    
    if acc_diff_radical > 0.1:
        print("   - Радикальные аугментации значительно ухудшили качество модели (более 10%)")
    
    print("\n3. Анализ MobileNet:")
    print("   - MobileNet является легковесной моделью, оптимизированной для мобильных устройств")
    print("   - Количество параметров значительно меньше по сравнению с VGG или другими крупными архитектурами")
    print("   - Несмотря на меньшее количество параметров, модель показывает хорошую точность на данном датасете")
    
    print("\n4. Общие наблюдения:")
    print("   - Выбор правильного LR scheduler важен для достижения оптимальных результатов")
    print("   - Сильные аугментации могут ухудшить производительность, если они слишком искажают данные")
    print("   - Transfer learning с предобученной MobileNet дает хорошие результаты даже при ограниченном количестве эпох обучения")

if __name__ == "__main__":
    main()
