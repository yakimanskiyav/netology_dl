# Сравнение моделей глубокого обучения на датасете EMNIST

Этот проект демонстрирует использование предварительно обученных моделей из библиотеки torchvision для классификации изображений рукописных букв и цифр из датасета EMNIST.

## Описание

Каждая модель обучается на датасете EMNIST , который содержит рукописные буквы (верхний и нижний регистр) и цифры, всего 47 классов.

## Структура проекта

- `emnist_resnet18.py` - обучение модели ResNet18
- `emnist_vgg.py` - обучение облегченной версии модели VGG (VGG11)
- `emnist_inception.py` - обучение модели MobileNet V2 (замена Inception v3)
- `emnist_densenet.py` - обучение облегченной версии модели DenseNet (DenseNet121)
- `compare_models.py` - скрипт для сравнения результатов всех моделей
- `run_all_models.py` - скрипт для последовательного запуска всех моделей и сравнения

## Требования

- Python 3.6+
- PyTorch 1.7+
- torchvision
- matplotlib
- tqdm
- numpy

Установка зависимостей:

```bash
pip install torch torchvision matplotlib tqdm numpy
```

## Запуск

### Оптимизации для работы на локальной машине

Все скрипты оптимизированы для работы на компьютере с ограниченной памятью GPU:
- Используются более легкие версии моделей
- Уменьшены размеры изображений (112x112 или 150x150)
- Уменьшены размеры батча (16-32 вместо 64)
- Ограничен размер датасета (20000 изображений для обучения, 4000 для тестирования)
- Уменьшено количество рабочих потоков (workers)
- Добавлена очистка кэша CUDA между проходами

### Обучение отдельной модели

Для обучения конкретной модели запустите соответствующий файл:

```bash
python emnist_resnet18.py  # для ResNet18
python emnist_vgg.py       # для VGG11
python emnist_inception.py # для MobileNet V2 (замена для Inception v3)
python emnist_densenet.py  # для DenseNet121
```

### Обучение всех моделей и сравнение

Для запуска последовательного обучения всех моделей и автоматического сравнения результатов:

```bash
python run_all_models.py
```

Этот скрипт также очищает память GPU между запусками и отображает информацию о свободной памяти.

## Настройка ресурсов

В каждом файле модели можно дополнительно настроить следующие параметры:

- `batch_size` - размер батча (уменьшите еще больше при нехватке памяти GPU)
- `num_epochs` - количество эпох обучения
- `img_size` - размер изображения (уменьшите для экономии памяти)
- `train_size`, `test_size` - количество изображений для обучения и тестирования

## Результаты

После обучения каждой модели результаты сохраняются в:
- `{model_name}_emnist.pth` - веса модели
- `{model_name}_history.json` - история обучения (потери и точность)
- `{model_name}_results.png` - графики обучения отдельной модели

Сравнение всех моделей сохраняется в:
- `model_comparison_results.png` - графики сравнения всех моделей

## Особенности архитектур и оптимизации

### ResNet18
- Легкая и быстрая модель
- Размер батча: 32
- Размер изображения: 112x112

### VGG11 (вместо VGG16)
- Используется облегченная версия VGG с меньшим количеством слоев
- Размер батча: 16
- Размер изображения: 112x112

### MobileNet V2 (вместо Inception v3)
- Эффективная модель, специально разработанная для устройств с ограниченными ресурсами
- Размер батча: 16
- Размер изображения: 150x150

### DenseNet121 (вместо DenseNet161)
- Используется меньшая версия DenseNet
- Размер батча: 16
- Размер изображения: 112x112

