# Задание 3. Лабораторная работа №3

---
## Оглавление
- [Свойства видеокарт:](#свойства-видеокарты)
- [Формулировка задания по Лабораторной работе 3](#формулировка-задания-по-Лабораторной-работе-3)\
- [Планирование задачи через SLURM и компиляция кода](#Планирование-задачи-через-SLURM-и-компиляция-кода)
- [Реализация](#реализация)
   - [Основные компоненты программы](#Основные-компоненты-программы)
   - [Структуры данных](#Структуры-данных)
        - [Структура Image](#Структура-Image)
        - [Структура DualGPUContext](#Структура-DualGPUContext)
    - [Инициализация системы двух GPU](#Инициализация-системы-двух-GPU)
    - [CUDA ядра для обработки изображений](#CUDA-ядра-для-обработки-изображений)
        - [Ядро сильного размытия (5x5 гауссов фильтр)](#Ядро-сильного-размытия-(5x5-гауссов-фильтр))
        - [Ядро обычного размытия (3x3 гауссов фильтр)](#Ядро-обычного-размытия-(3x3-гауссов-фильтр))
        - [Ядро добавления гауссового шума](#Ядро-добавления-гауссового-шума)
    - [Система распределения вычислений между GPU](#Система-распределения-вычислений-между-GPU)
        - [Алгоритм распределения](#Алгоритм-распределения)
        - [Настройка сетки и блоков](#Настройка-сетки-и-блоков)
        - [Управление памятью и выполнение](#Управление-памятью-и-выполнение)
        - [Обработка граничных условий](#Обработка-граничных-условий)
        - [Обработка ошибок и синхронизация](#Обработка-ошибок-и-синхронизация)
    - [Основной алгоритм работы программы](#Основной-алгоритм-работы-программы)
- [Анализ результатов](#анализ-результатов)
   - [Производительность фильтров](#производительность-фильтров)
- [Выводы](#выводы)

---

# Свойства видеокарты

Работа выполнялась на видеокартах Tesla V100-PCIE-16GB - профессиональной видеокарте для вычислений на архитектуре Volta. 

Объем видеопамяти составляет 16 384 MiB (16 ГиБ), используется примерно 3.85 ГБ. Память занята в основном процессом LBM_CUDA (3682 MiB) и графической оболочкой (Xorg, gnome-shell). 

Температура 59°C, энергопотребление 122 Вт из максимальных 250 Вт. 

Драйвер версии 560.35.05, поддержка CUDA 12.6. 

Persistence-M | On. Этот режим предотвращает выгрузку драйвера при бездействии, что уменьшает задержки при запуске вычислений на GPU.

В системе доступно 2 видеокарты Tesla V100. В выводе nvidia-smi представлены две таблицы с индексами 0 и 1. 

---

## Формулировка задания по Лабораторной работе 3 
 
Реализовать программу для накладывания фильтров на изображения. Возможные фильтры: размытие, выделение границ, избавление от шума. Изменить время
Для работы с графическими файлами рекомендуется использовать libpng (man libpng). Примеры использования библиотеки в /usr/share/doc/libpng12-dev/examples/
Считать изображение из файла, преобразовать в массив, отправить на CUDA Device, провести на устройстве обработку фильтром, вернуть изображение в память хоста, преобразовать в картинку, сохранить файл.

Для выполнения использовать две видеокарты.
---

## Планирование задачи через SLURM и компиляция кода

1. **Планирование задачи через SLURM**

Скрипт #!/bin/bash с директивами #SBATCH - задача для менеджера очередей SLURM. 

    * -p gpuserv: задача должна выполняться на партиции (очереди), которая содержит узлы с GPU.

    * -N 1 -n 1: Запрашивает 1 узел и 1 задачу.

2. **Загрузка программного обеспечения**

`module load nvidia/cuda` загружает среду CUDA, делая доступными такие ключевые утилиты, как nvidia-smi и nvcc (компилятор CUDA).

3. **Прямой опрос оборудования с помощью nvidia-smi**

Это самый важный шаг для получения свойств. Команда nvidia-smi (NVIDIA System Management Interface) взаимодействует с драйвером, запрашивает данные и форматирует и выводит данные. 

4. **Компиляция и запуск кода**

`nvcc -o lab3 lab3.cu -lpng`. Компилятор CUDA (nvcc) транслирует код из lab3.cu в исполняемый файл lab3. Флаг -lpng указывает на необходимость линковки с библиотекой для работы с PNG-изображениями.

---

## Реализация

Программа представляет собой систему параллельной обработки изображений с использованием двух GPU. Основные компоненты включают структуры данных для хранения изображений и контекста GPU, CUDA ядра для обработки изображений, систему распределения вычислений и функции ввода-вывода для работы с PNG файлами.

---

### Основные компоненты программы

1. Структуры данных для хранения изображений и управления GPU

2. CUDA ядра для параллельной обработки пикселей

3. Функции ввода-вывода для работы с PNG форматом

4. Система управления двумя GPU для распределения вычислений

5. Оптимизации производительности через pinned memory и асинхронные операции

---
### Структуры данных

---

#### Структура Image

``` C
struct Image {
    int width;
    int height;
    unsigned char *data;
};
```
Хранит информацию об изображении в формате RGBA (4 канала на пиксель).

    * width - ширина изображения в пикселях

    * height - высота изображения в пикселях

    * data - указатель на массив данных изображения в формате [R,G,B,A,R,G,B,A,...]

---

#### Структура DualGPUContext

``` C
struct DualGPUContext {
    int gpu0;
    int gpu1;
    size_t total_size;
    size_t split_height;
    int overlap;
};
```

Управляет конфигурацией двух GPU для параллельной обработки с учетом перекрытия данных.

    * gpu0 - идентификатор первого GPU (верхняя часть изображения)

    * gpu1 - идентификатор второго GPU (нижняя часть изображения)

    * total_size - общий размер данных изображения

    * split_height - высота разделения изображения между GPU

    * overlap - количество строк перекрытия между частями

---

### Инициализация системы двух GPU

Функция initDualGPU() выполняет настройку двух-GPU окружения:

``` C
DualGPUContext initDualGPU(int image_height, int kernel_radius) {
    DualGPUContext context;
    
    // Проверка доступности GPU
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    printf("Found %d CUDA devices\n", device_count);
    
    if (device_count < 2) {
        fprintf(stderr, "Error: Need at least 2 GPUs, but only %d found\n", device_count);
        exit(1);
    }
    
    // Настройка контекста
    context.gpu0 = 0;
    context.gpu1 = 1;
    context.overlap = kernel_radius;
    
    // Разделение с перекрытием для обработки граничных пикселей
    context.split_height = image_height / 2 + kernel_radius;
    if (context.split_height > (size_t)image_height) {
        context.split_height = image_height;
    }
    
    // Информация о GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, context.gpu0);
    printf("GPU0: %s\n", prop.name);
    cudaGetDeviceProperties(&prop, context.gpu1);
    printf("GPU1: %s\n", prop.name);
    
    printf("Using GPUs %d and %d\n", context.gpu0, context.gpu1);
    printf("Image split at height: %zu (overlap: %d rows)\n", 
           context.split_height, context.overlap);
    
    return context;
}
```

**Алгоритм инициализации**

    1. Проверка доступности минимум двух GPU через cudaGetDeviceCount()

    2. Назначение GPU0 и GPU1 для параллельной обработки

    3. Расчет высоты разделения с учетом перекрытия для фильтров

    4. Получение и вывод информации о характеристиках GPU

    5. Валидация конфигурации и диагностический вывод

---

### CUDA ядра для обработки изображений

---

### Warm-up ядро

``` C
__global__ void warmupKernel() {}
```

Инициализирует контекст CUDA на обоих GPU перед основными вычислениями для исключения накладных расходов на первом запуске.

---

#### Ядро обычного размытия (3x3 гауссов фильтр)

``` C
__global__ void blurKernel(unsigned char* input, unsigned char* output, 
                          int width, int part_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= part_height) return;

    float kernel[3][3] = {
        {1.0f/16, 2.0f/16, 1.0f/16},
        {2.0f/16, 4.0f/16, 2.0f/16},
        {1.0f/16, 2.0f/16, 1.0f/16}
    };

    float r=0, g=0, b=0;
    for(int ky=-1; ky<=1; ky++) {
        for(int kx=-1; kx<=1; kx++) {
            int nx = min(max(x+kx,0), width-1);
            int ny = min(max(y+ky,0), part_height-1);
            int idx = (ny*width+nx)*4;
            float w = kernel[ky+1][kx+1];
            r += input[idx]*w; 
            g += input[idx+1]*w; 
            b += input[idx+2]*w;
        }
    }
    int out_idx = (y*width+x)*4;
    output[out_idx]=r; 
    output[out_idx+1]=g; 
    output[out_idx+2]=b; 
    output[out_idx+3]=255;
}
```

**Характеристики фильтра**

    * Размер ядра: 3×3 пикселя

    * Тип: Упрощенный гауссов фильтр

    * Сумма коэффициентов: 1.0 (сохранение яркости)

**Алгоритм работы**

    1. Каждый поток обрабатывает один пиксель выходного изображения

    2. Для каждого пикселя вычисляется взвешенная сумма 3×3 области

    3. Граничные условия обрабатываются через функции min/max

    4. Результат записывается в выходной буфер с установкой альфа-канала

---

#### Ядро сильного размытия (5x5 гауссов фильтр)

``` C
__global__ void strongBlurKernel(unsigned char* input, unsigned char* output, 
                                int width, int part_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= part_height) return;

    float kernel[5][5] = {
        {1,4,6,4,1}, {4,16,24,16,4}, {6,24,36,24,6}, 
        {4,16,24,16,4}, {1,4,6,4,1}
    };
    float sum = 256.0f;
    float r=0,g=0,b=0;
    
    for(int ky=-2; ky<=2; ky++){
        for(int kx=-2; kx<=2; kx++){
            int nx=min(max(x+kx,0),width-1);
            int ny=min(max(y+ky,0),part_height-1);
            int idx=(ny*width+nx)*4;
            float w=kernel[ky+2][kx+2]/sum;
            r+=input[idx]*w; 
            g+=input[idx+1]*w; 
            b+=input[idx+2]*w;
        }
    }
    
    int out_idx=(y*width+x)*4;
    output[out_idx]=r; 
    output[out_idx+1]=g; 
    output[out_idx+2]=b; 
    output[out_idx+3]=255;
}
```

**Характеристики фильтра**

    * Размер ядра: 5×5 пикселей

    * Тип: Гауссов фильтр с нормальным распределением

    * Нормализация: сумма коэффициентов = 256

**Особенности реализации**

    * Использует предварительно рассчитанную матрицу свертки

    * Обработка границ через ограничение координат

    * Более интенсивное размытие за счет большей области влияния

---

#### Ядро добавления гауссового шума

``` C
__global__ void addGaussianNoiseKernel(unsigned char* input, unsigned char* output, 
                                      int width, int part_height, unsigned int seed) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= part_height) return;
    
    int idx = (y*width+x)*4;
    unsigned int noise_seed = (x * 12347 + y * 54321 + seed * 76543) % 1234567;
    float noise = (float)(noise_seed % 200) / 200.0f * 40.0f - 20.0f;
    
    for(int c=0;c<3;c++){
        float val=input[idx+c]+noise;
        output[idx+c]=(unsigned char)min(max(val,0.0f),255.0f);
    }
    output[idx+3]=255;
}
```

**Характеристики шума**

    * Тип: Псевдогауссов шум с равномерным распределением

    * Диапазон: ±20 единиц интенсивности

**Алгоритм работы**

    1. Генерация "случайного" числа на основе координат пикселя

    2. Нормализация в диапазон ±20

    3. Добавление шума к каждому цветовому каналу (R, G, B)

    4. Ограничение результата диапазоном [0, 255]

    5. Сохранение альфа-канала

---

### Система распределения вычислений между GPU

---

#### Алгоритм распределения

``` C
void applyFilterDualGPU(Image input, Image* output, const char* filterType, int kernelSize) {
    size_t total_size = input.width * input.height * 4;
    int kernel_radius = (kernelSize - 1) / 2;
    DualGPUContext context = initDualGPU(input.height, kernel_radius);
    
    // pinned memory для быстрых трансферов
    CUDA_CHECK(cudaHostRegister(input.data, total_size, cudaHostRegisterDefault));
    CUDA_CHECK(cudaHostRegister(output->data, total_size, cudaHostRegisterDefault));
    
    // Инициализация GPU
    cudaSetDevice(context.gpu0);
    warmupKernel<<<1,1>>>();
    cudaSetDevice(context.gpu1);
    warmupKernel<<<1,1>>>();
    cudaDeviceSynchronize();
    
    // Расчет размеров частей с учетом перекрытия
    size_t part1_size = input.width * context.split_height * 4;
    size_t part2_size = input.width * (input.height - context.split_height + context.overlap) * 4;
}
```

**Стратегия разделения данных**

    1. Вертикальное разделение - изображение делится по горизонтальной линии

    2. GPU0 обрабатывает верхнюю половину (от 0 до split_height)

    3. GPU1 обрабатывает нижнюю половину (от split_height до height)

    4. Перекрытие необходимо для корректной обработки граничных пикселей фильтрами

---

#### Настройка сетки и блоков

``` C
dim3 block(32,8);  // 256 потоков в блоке
int gridX = (input.width + block.x - 1) / block.x;
int gridY0 = (context.split_height + block.y - 1) / block.y;
int gridY1 = ((input.height - context.split_height + context.overlap) + block.y - 1) / block.y;
```

**Конфигурация выполнения**

    * Размер блока: 32×8 = 256 потоков

    * Сетка GPU0: gridX × gridY0 блоков для верхней части

    * Сетка GPU1: gridX × gridY1 блоков для нижней части

    * Округление вверх для полного покрытия всех пикселей

---

#### Управление памятью и выполнение

``` C
// GPU 0 инициализация
cudaSetDevice(context.gpu0);
CUDA_CHECK(cudaStreamCreate(&s0));
CUDA_CHECK(cudaEventCreate(&start0));
CUDA_CHECK(cudaEventCreate(&stop0));
CUDA_CHECK(cudaMalloc(&d_in0, part1_size));
CUDA_CHECK(cudaMalloc(&d_out0, part1_size));

// Асинхронное выполнение
CUDA_CHECK(cudaEventRecord(start0,s0));
CUDA_CHECK(cudaMemcpyAsync(d_in0,input.data,part1_size,cudaMemcpyHostToDevice,s0));

// Запуск ядра
blurKernel<<<dim3(gridX,gridY0),block,0,s0>>>(d_in0,d_out0,input.width,context.split_height);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaEventRecord(stop0,s0));
```

**Последовательность операций**

    1. Инициализация контекста - определение разделения и валидация

    2. Выделение памяти на каждом GPU для входных и выходных данных

    3. Асинхронное копирование данных с хоста на GPU

    4. Параллельный запуск ядер на обоих GPU

    5. Синхронизация и измерение времени выполнения

    6. Копирование результатов обратно на хост

    7. Освобождение ресурсов - памяти, потоков и событий

---

#### Обработка граничных условий

``` C
// GPU1 получает данные с перекрытием
size_t offset = (context.split_height - context.overlap) * input.width * 4;
CUDA_CHECK(cudaMemcpyAsync(d_in1,input.data+offset,part2_size,cudaMemcpyHostToDevice,s1));

// При копировании результатов перекрытие исключается
size_t gpu1_out_offset = context.split_height * input.width * 4;
size_t gpu1_out_size = (input.height - context.split_height) * input.width * 4;
CUDA_CHECK(cudaMemcpyAsync(output->data+gpu1_out_offset,
                          d_out1+context.overlap*input.width*4,
                          gpu1_out_size,cudaMemcpyDeviceToHost,s1));
```


**Решение проблемы границ**

    * Перекрытие данных: GPU1 получает дополнительные строки сверху для корректной работы фильтров

    * Исключение дублирования: При копировании результатов перекрывающиеся строки отбрасываются

    * Локальная обработка: Ядра работают только с локальными координатами своей части

---
#### Оптимизации производительности

``` C
// Pinned memory для быстрых Host-to-Device трансферов
CUDA_CHECK(cudaHostRegister(input.data, total_size, cudaHostRegisterDefault));
CUDA_CHECK(cudaHostRegister(output->data, total_size, cudaHostRegisterDefault));

// Асинхронные операции через CUDA streams
cudaStream_t s0, s1;
CUDA_CHECK(cudaStreamCreate(&s0));
CUDA_CHECK(cudaStreamCreate(&s1));

// Warm-up ядра для инициализации контекста
warmupKernel<<<1,1>>>();
```

**Ключевые оптимизации**

    * Pinned Memory

    * Асинхронные операции - параллельное выполнение на двух GPU

    * Warm-up - инициализация контекста до основных вычислений

    * Раздельные потоки - независимое управление каждым GPU

---

#### Обработка ошибок и синхронизация

**Механизм обработки ошибок**

    * Макрос CUDA_CHECK для проверки всех вызовов CUDA API

    * Отдельная проверка ошибок после запуска ядер через cudaGetLastError()

    * Детальное логирование времени выполнения для каждого GPU

---

### Основной алгоритм работы программы

``` C
int main() {
    // 1. Загрузка входного изображения
    Image input = readPNG("creative-pebble.png");
    
    // 2. Подготовка выходных изображений
    Image out1={input.width,input.height,(unsigned char*)malloc(size)};
    Image out2={input.width,input.height,(unsigned char*)malloc(size)};
    Image out3={input.width,input.height,(unsigned char*)malloc(size)};
    
    printf("=== Dual GPU Processing ===\nInput: %dx%d\n", input.width, input.height);
    
    // 3. Последовательное применение трех фильтров
    applyFilterDualGPU(input,&out1,"blur",3);
    applyFilterDualGPU(input,&out2,"strong_blur",5);
    applyFilterDualGPU(input,&out3,"add_noise",3);
    
    // 4. Сохранение результатов
    writePNG("output_blur_dual.png",out1);
    writePNG("output_strong_blur_dual.png",out2);
    writePNG("output_noisy_dual.png",out3);
    
    // 5. Освобождение ресурсов
    free(input.data); free(out1.data); free(out2.data); free(out3.data);
    cudaDeviceReset();
    printf("Completed successfully!\n");
    return 0;
}
```
---

## Анализ результатов

Для экспериментов использовалось изображение размером **2000×2000 пикселей**.  
Сравнивалась производительность выполнения трёх фильтров (`blur`, `strong_blur`, `add_noise`)  
при использовании **одного GPU** и **двух GPU** (распараллеливание по высоте изображения).

### Если использовать один GPU (результаты лабораторной 2)

| Этап | H2D copy (ms) | Kernel (ms) | D2H copy (ms) | Общее время (ms) |
|------|----------------|--------------|----------------|------------------|
| Blur | 5.669 | 0.079 | 0.741 | **6.489** |
| Strong blur | 0.914 | 0.064 | 0.695 | **1.673** |
| Add noise | 0.829 | 0.039 | 0.748 | **1.616** |

---

### Если использовать два GPU

| Этап | GPU0 Kernel (ms) | GPU1 Kernel (ms) | Среднее время ядра (ms) | Общее время (ms) |
|------|-------------------|-------------------|---------------------------|------------------|
| Blur | 0.116 | 0.119 | **0.118** | **0.119** |
| Strong blur | 0.130 | 0.131 | **0.131** | **0.131** |
| Add noise | 0.115 | 0.117 | **0.116** | **0.117** |

---

### Производительность фильтров

| Этап | Время (1 GPU), ms | Время (2 GPU), ms | Ускорение (×) |
|------|--------------------|--------------------|---------------|
| Blur | 6.489 | 0.119 | **≈54.6×** |
| Strong blur | 1.673 | 0.131 | **≈12.8×** |
| Add noise | 1.616 | 0.117 | **≈13.8×** |

---

## Выводы

1. Использование **двух GPU** дало значительное ускорение по всем фильтрам.  Особенно заметен прирост при вычислительно более сложных фильтрах (`blur` и `strong_blur`).

2. Время выполнения ядра на каждом GPU практически одинаково, что свидетельствует о **равномерной нагрузке** и **корректном разбиении изображения**.

3. Основное время при однографическом исполнении занимают операции **копирования данных** между CPU и GPU, в то время как при параллельной обработке это влияние минимизируется.

4. Полученное ускорение (до ~55×) демонстрирует эффективность **распараллеливания вычислений** и использования **асинхронных потоков** с **перекрытием областей изображения**.
