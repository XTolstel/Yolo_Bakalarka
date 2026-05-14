
import psutil
import time
import os
import subprocess
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates, nvmlDeviceGetPowerUsage, NVMLError
import pynvml
import numpy as np
from Yolo_test import object_detection  # Импортируем класс Object из файла objects.py
import matplotlib.pyplot as plt
import traceback

def read_energy(path="/sys/class/powercap/intel-rapl:0/energy_uj"):
    try:
        result = subprocess.run(["sudo", "cat", path], capture_output=True, text=True, check=True)
        return int(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print("Файл не найден")
        return None

def calculate_power(start_energy, end_energy, time):
    energy_diff = end_energy - start_energy
    #print(start_energy)
   # print(end_energy)
   # print(energy_diff)
    #print(time)
    power=energy_diff / (time* 1e6)
    #print(power)
    return power

try:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    print("GPU доступен")
    gpu_available = True
except pynvml.NVMLError as e:
    print(f"Ошибка NVML: {e}")
    print("Переход на использование CPU")
    gpu_available = False
#nvmlInit()
#handle = nvmlDeviceGetHandleByIndex(0)

# Мониторинг ресурсов
cpu_usage = []
gpu_usage = []
gpu_power = []
cpu_power = []
start_time = time.time()

process = psutil.Process()
result = None
detection_error = None
start_energy = None
#global start_energy
try:
    # Отдельный поток для запуска detection()
    from threading import Thread
    from Yolo_test import object_detection 

    def run_detection():
        global result, start_energy, detection_error
        try:
            start_energy = read_energy()
            result = object_detection.detection()
        except Exception as e:
            detection_error = e
            print("Detection failed:")
            traceback.print_exc()
    detection_thread = Thread(target=run_detection)
    detection_thread.start()
    # Мониторинг ресурсов
    time_graph = []
    tim = 0
    while detection_thread.is_alive():
        # CPU загрузка
        cpu_usage.append(psutil.cpu_percent(interval=0.1))

        cpu_power.append(process.cpu_percent() / 100)
        time.sleep(0.1) 
        tim=(time.time()-start_time)
        
        #print(f"Program is working: {tim:.1f}")
        time_graph.append(time.time()-start_time)
        
        if gpu_available:
            gpu_util = nvmlDeviceGetUtilizationRates(handle)
            gpu_usage.append(gpu_util.gpu)
            gpu_power.append(nvmlDeviceGetPowerUsage(handle) / 1000)
        else:
            gpu_usage.append(0)
            gpu_power.append(0)

except Exception as e:
    print(f"Ошибка: {e}")

if detection_error is not None:
    raise RuntimeError(f"Detection did not complete: {detection_error}")

end_energy = read_energy()
end_time = time.time()
total_time = end_time - start_time
power_cpu = 0
if start_energy is not None and end_energy is not None and total_time > 0:
    power_cpu = calculate_power(start_energy, end_energy, total_time)

    #print(cpu_power[i])
def save_plot(x, y, filename, xlabel, ylabel, title):
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label="Дані")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

#создание графиков 
save_plot(time_graph,cpu_usage,"plot_cpu","Time","cpu_usage","CPU_usage")
save_plot(time_graph,gpu_usage,"plot_gpu","Time","gpu_usage","GPU_usage")
save_plot(time_graph,gpu_power,"plot_gpu_power","Time","gpu_power","GPU_power")

# Средние значения
avg_cpu_usage = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
avg_gpu_usage = sum(gpu_usage) / len(gpu_usage) if gpu_usage else 0
avg_gpu_power = sum(gpu_power) / len(gpu_power) if gpu_power else 0

print(f"Время компиляции: {total_time:.2f} сек")
print(f"Загруженность CPU; {avg_cpu_usage:.2f}")
print(f"Загруженность GPU: {avg_gpu_usage:.2f}")
print(f"Средняя мощность видеокарты: {avg_gpu_power:.2f}")
print(f"Средняя мощность процессора: {power_cpu:.2f}")
avg_time_per_frame = 0
latencies_ms = []
processed_frames = 0
processing_time_s = 0

if isinstance(result, dict):
    avg_time_per_frame = result.get("avg_time_per_frame_s", 0)
    latencies_ms = result.get("latencies_ms", []) or []
    processed_frames = result.get("total_frames", 0) or 0
    processing_time_s = result.get("total_processing_time_s", 0) or 0
else:
    avg_time_per_frame = result if result is not None else 0

energy_per_frame_j = 0
if processed_frames > 0:
    total_gpu_energy_j = avg_gpu_power * total_time
    total_cpu_energy_j = (end_energy - start_energy) / 1e6 if (start_energy is not None and end_energy is not None) else 0
    energy_per_frame_j = (total_cpu_energy_j + total_gpu_energy_j) / processed_frames

avg_fps = (processed_frames / processing_time_s) if processing_time_s > 0 else 0
p95_latency_ms = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0
p99_latency_ms = float(np.percentile(latencies_ms, 99)) if latencies_ms else 0

# Сохраняем полный исходный вывод + дополняем его 4 новыми метриками
print(f"Время обработки одного кадра: {avg_time_per_frame:.2f}")
print(f"avg_fps: {avg_fps:.2f}")
print(f"p95_latency_ms: {p95_latency_ms:.2f}")
print(f"p99_latency_ms: {p99_latency_ms:.2f}")
print(f"energy_per_frame_j: {energy_per_frame_j:.6f}")
if gpu_usage == 0 and gpu_power == 0:
    print("GPU is missing")
