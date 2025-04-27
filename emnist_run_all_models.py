import subprocess
import os
import time
import torch
import gc

def clean_gpu_memory():
    """Очистка памяти GPU между запусками различных моделей"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("Память GPU очищена")

def get_gpu_memory_info():
    """Получить информацию о свободной и занятой памяти GPU"""
    if not torch.cuda.is_available():
        return "GPU недоступен"
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # В ГБ
    reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
    allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
    free_memory = total_memory - reserved_memory
    
    return f"GPU память: всего {total_memory:.2f} ГБ, свободно {free_memory:.2f} ГБ, занято {allocated_memory:.2f} ГБ"

def run_python_script(script_name):
    """Запуск Python скрипта и возврат статуса выполнения"""
    print(f"\n{'='*50}")
    print(f"Запуск {script_name}...")
    print(get_gpu_memory_info())  # Выводим информацию о памяти перед запуском
    print(f"{'='*50}\n")
    
    result = subprocess.run(['python', script_name], capture_output=False, text=True)
    
    clean_gpu_memory()  # Очищаем память после запуска
    
    if result.returncode == 0:
        print(f"\n{script_name} выполнен успешно.")
        return True
    else:
        print(f"\nОшибка при выполнении {script_name}!")
        return False

# Список моделей для обучения, начиная с наименее требовательных к памяти
model_scripts = [
    "emnist_resnet18.py",  # ResNet18 - наименее требовательная
    "emnist_vgg.py",       # VGG требует больше памяти
    "emnist_inception.py", # MobileNet (замена Inception)
    "emnist_densenet.py"   # DenseNet - может требовать много памяти
]

print("\nПроверка доступности GPU...")
print(get_gpu_memory_info())

print("\nНастройки для запуска на локальном компьютере:")
print("- Размер датасета ограничен 20к/4к изображений")
print("- Размер изображений уменьшен")
print("- Размер батча уменьшен")
print("- Используются легкие версии моделей")
print("- Память GPU очищается между запусками")

# Запуск обучения всех моделей последовательно
successful_runs = []
failed_runs = []

start_time = time.time()

# Очистим память перед началом
clean_gpu_memory()

for script in model_scripts:
    if os.path.exists(script):
        success = run_python_script(script)
        if success:
            successful_runs.append(script)
        else:
            failed_runs.append(script)
    else:
        print(f"Файл {script} не найден!")
        failed_runs.append(script)

# Запуск скрипта сравнения моделей
if successful_runs:
    print("\nЗапуск сравнения моделей...")
    run_python_script("compare_models.py")

total_time = time.time() - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print("\n" + "="*50)
print("Статус выполнения экспериментов:")
print(f"Всего времени затрачено: {int(hours)}ч {int(minutes)}м {int(seconds)}с")

if successful_runs:
    print("\nУспешно выполнены:")
    for script in successful_runs:
        print(f"  - {script}")

if failed_runs:
    print("\nНе удалось выполнить:")
    for script in failed_runs:
        print(f"  - {script}")

print("="*50)
print("\nРабота завершена! Если все модели были успешно обучены, то результаты сравнения находятся в файле model_comparison_results.png") 