# 1. Text to vector

### Название файла: text_to_vect.py

### Скрипт преобрует тексты в их векторное представление

#### Используемые предобученные модели
- longformer-base-4096
- longformer-large-4096
- sbert paraphrase-MiniLM-L6-v2
- sbert all-MiniLM-L6-v2
- sbert paraphrase-MiniLM-L3-v2

> https://www.sbert.net/docs/pretrained_models.html#model-overview

> https://github.com/allenai/longformer#how-to-use

#### Описание
Программа на вход получает в качестве обязательных аргументов:
- путь до датасета с текстами
- путь до директории, в которой будет сохранен файл с векторами
- режим работы программы

В качестве опциональных аргументов:
- метка об использовании GPU (по умолчанию CPU)
- название тренированной finetune модели

После обработки всех текстов в датасете результаты будут сохранены в csv-файле, который будет находиться в указанной директории. Файл будет иметь два столбца: 
1. id текста
2. векторное представление

Ниже в описании режимов работы программы есть пометка "полный текст" и "начало текста". Это означает, что для формирования векторного представления берется весь текст или начало текста (4000 токенов) соответственно.

#### Требования 
- модели должны храниться в директории **../models/**
- тексты в датасете должны быть в формате **.txt**
- должны быть скачаны **longformer-base-4096**, **longformer-large-4096**, **sentence-transformers**

#### Скачать нужные модели
- pip install -U sentence-transformers (загружается в requirements)
- https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz (загружается и распаковывается в Dockerfile)
- https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-large-4096.tar.gz (загружается и распаковывается в Dockerfile)


## Для запуска 

### Аргументы:
- path_text - путь до датасета с текстами
- path_vect - путь до директории, в которой будет файл с векторами
- mode - режим работы программы
    - 1-5 base-4096
    - 6-10 large-4096
    - 11-12 sbert

### Опциональные аргументы:
- usegpu - использовать GPU или CPU (если флаг указан - GPU, если не указан - CPU) (-ug or --usegpu)
- finetune - использовать тренированную модель (-ft or --finetune). 
    - После ключа указывается название модели, которая должна находиться в папке **../models/**

### mode:

- longformer base-4096
- 1 - суммирование векторов, полный текст
- 2 - суммирование векторов, начало текста
- 3 - максимальное в столбце, полный текст
- 4 - максимальное в столбце, начало текста
- 5 - **cls** токен
- 
- longformer large-4096
- 6 - суммирование векторов, полный текст
- 7 - суммирование векторов, начало текста
- 8 - максимальное в столбце, полный текст
- 9 - максимальное в столбце, начало текста
- 10 - **cls** токен
- 
- sbert
- 11 - суммирование векторов
- 12 - максимальное в столбце


### Пример запуска 1

`python3 text_to_vect.py ../data/dataset_v1/text ../notebooks/vectors_finetune 11 -ug`

- `../data/dataset_v1/text` - путь до датасета с текстами
- `../notebooks/vectors_finetune` - путь до директории, в которой будет файл с векторами
- `11` - режим работы программы
- `-ug` - использовать GPU

### Вывод в консоль для запуска 1

```
  ______    ______   _  __  ______       __           ______   ____        __          _    __    ______   ______  ______
 /_  __/   / ____/  | |/ / /_  __/       \ \         /_  __/  / __ \       \ \        | |  / /   / ____/  / ____/ /_  __/
  / /     / __/     |   /   / /           \ \         / /    / / / /        \ \       | | / /   / __/    / /       / /   
 / /     / /___    /   |   / /            / /        / /    / /_/ /         / /       | |/ /   / /___   / /___    / /    
/_/     /_____/   /_/|_|  /_/            /_/        /_/     \____/         /_/        |___/   /_____/   \____/   /_/     
                                                                                                                         

The script is running...
__
The path of the dataset with texts: ../data/dataset_v1/text
The path to the directory for output: ../notebooks/vectors_finetune
Text processing method: 11
Using GPU: True
Using fine tune model: None
__
After successful execution of the program, the result of the program will be in the file: ../notebooks/vectors_finetune/vectors11.csv
__
The model is loading...
__
The script was launched at 22-07-2022 11:15:59
 processing... 20179 / 20179
The script is completed at 22-07-2022 11:42:12
Execution time = 0:26:12.613864
```


### Полученный файл csv для запуска 1

>doc_id,vector
>
>312038,"[-12.530302047729492, 45.77742385864258, ..., 12.77685832977295]"
>
>...


### Пример запуска 2

`python3 text_to_vect.py ../data/dataset_v1/text ../notebooks/vectors_finetune 5 -ug -ft longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500`

- `../data/dataset_v1/text` - путь до датасета с текстами
- `../notebooks/vectors_finetune` - путь до директории, в которой будет файл с векторами
- `5` - режим работы программы
- `-ug` - использовать GPU
- `-ft longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500` - использовать тренированную модель с таким названием


### Вывод в консоль для запуска 2
```
  ______    ______   _  __  ______       __           ______   ____        __          _    __    ______   ______  ______
 /_  __/   / ____/  | |/ / /_  __/       \ \         /_  __/  / __ \       \ \        | |  / /   / ____/  / ____/ /_  __/
  / /     / __/     |   /   / /           \ \         / /    / / / /        \ \       | | / /   / __/    / /       / /   
 / /     / /___    /   |   / /            / /        / /    / /_/ /         / /       | |/ /   / /___   / /___    / /    
/_/     /_____/   /_/|_|  /_/            /_/        /_/     \____/         /_/        |___/   /_____/   \____/   /_/     
                                                                                                                         

The script is running...
__
The path of the dataset with texts: ../data/dataset_v1/text
The path to the directory for output: ../notebooks/vectors_finetune
Text processing method: 5
Using GPU: True
Using fine tune model: longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500
__
After successful execution of the program, the result of the program will be in the file: ../notebooks/vectors_finetune/vectors-longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500.csv
__
The model is loading...
__
The script was launched at 22-07-2022 11:53:22
 processing... 20179 / 20179
The script is completed at 22-07-2022 13:07:41
Execution time = 1:14:19.375762
```

### Полученный файл csv для запуска 2

>doc_id,vector
>
>312038,"[0.36936789751052856, 0.022108297795057297, ..., 0.853970468044281]"
>
>...


# 2. Document semantic matching (testing vectors on test file)

### Название файла: doc_sem_matching.py

### Скрипт тестирует полученные векторные представления текстов на тестовом файле на задаче попарного сравнения докуменов (document semantic matching)

#### Описание
Программа на вход получает в качестве обязательных аргументов:
- путь до файла с векторами
- метод формирования векторов
- путь до файла с тестовыми значениями
- путь до директории, в которой будет файл .json с оценками
- начальное и конечное значения порогового значения, а также шаг увеличения порогового значения. 

В качестве опциональных аргументов:
- название датасета, по которому формировались векторы (по умолчанию 'dataset')
- метка использования finetune модели (если не указана, значит не используется)

Программа меняет пороговое значение в заданном диапазоне. И для каждого порогового значения производится вычисление предполагаемых значений (prediction label)(если значение косинусного расстояния между двумя текстами > порогового значения, то предполагаемое значение = 1, иначе  = 0) и целевых значений (true label)(целевое значение берется из тестового файла). Далее на основе результатов считаются необходимые метрики (precision, recall, f1-score).

После тестирования всех векторных представлений оценочные результаты будут сохранены в json-файле, который будет находиться в указанной директории. Файл будет включать в себя: информацию об используемой модели, дату тестирования, название датасета, характеристики задания, метрики.

#### Требования
- расширение файла с векторами **.csv**
- расширение файла с тестовыми значениями **.csv** 
- формат файла с тестовыми значениями. Минимум 5 столбцов с наименованиями: **doc_id_1, title_1, doc_id_2, title_2, label**


## Для запуска

### Аргументы:
- path_vect - путь до файла с векторами
- mode - метод формирования векторов
    - 1-5 base-4096
    - 6-10 large-4096
    - 11-12 sbert
- path_test - путь до файла с тестовыми значениями
- path_eval - путь до директории, в которой будет файл .json с оценками
- start - начальное значение порогового значения
- stop - конечное значение порогового значения
- step - шаг увеличения порогового значения

#### Опциональные аргументы:
- dataset (-d or --dataset) - название датасета, по которому формировались векторы (default='dataset')
- finetune (-ft or --finetune) - используется finetune модель (если ключ указан, то True, не указан - False)


### mode:

- base-4096
- 1 - суммирование векторов, полный текст
- 2 - суммирование векторов, начало текста
- 3 - максимальное в столбце, полный текст
- 4 - максимальное в столбце, начало текста
- 5 - **cls** токен
- 
- large-4096
- 6 - суммирование векторов, полный текст
- 7 - суммирование векторов, начало текста
- 8 - максимальное в столбце, полный текст
- 9 - максимальное в столбце, начало текста
- 10 - **cls** токен
- 
- sbert
- 11 - суммирование векторов
- 12 - максимальное в столбце


### Пример запуска 1

`python3 doc_sem_matching.py vectors-longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500.csv 5 ../data/dataset_v1/test.csv ../notebooks 0.6 1 0.1 -d dataset_v1 -ft`

- `vectors-longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500.csv` - путь до файла с векторами
- `5` - метод формирования векторов
- `../data/dataset_v1/test.csv` - путь до файла с тестовыми значениями
- `../notebooks` - путь до директории, в которой будет файл .json с оценками
- `0.6` - начальное значение порогового значения
- `1` - конечное значение порогового значения
- `0.1` - шаг увеличения порогового значения
- `-d dataset_v1` - название датасета, по которому формировались векторы
- `-ft` - для формирования используемых векторов применялась finetune модель

### Вывод в консоль для запуска 1
```
    ____                         _____                             __  ___           __            __  
   / __ \  ____   _____         / ___/  ___    ____ ___           /  |/  /  ____ _  / /_  _____   / /_ 
  / / / / / __ \ / ___/ ______  \__ \  / _ \  / __ `__ \ ______  / /|_/ /  / __ `/ / __/ / ___/  / __ \
 / /_/ / / /_/ // /__  /_____/ ___/ / /  __/ / / / / / //_____/ / /  / /  / /_/ / / /_  / /__   / / / /
/_____/  \____/ \___/         /____/  \___/ /_/ /_/ /_/        /_/  /_/   \__,_/  \__/  \___/  /_/ /_/   
                                                                                                            

The script is running...
__
The path to the file with vectors: ../notebooks/vectors11_dataset_v1_20179.csv
The vector formation method: 11
The path to the file with test values: ../data/dataset_v1/test.csv
The path to the directory for output with evaluations: ../notebooks
The threshold value from 0.6 to 1.0 with step 0.1
The name of the dataset by which the vectors were formed: dataset_v1
Using fine tune model: False
__
After successful execution of the program, the result of the program will be in the file: ../notebooks/evaluation11.json
__
The script was launched at 22-07-2022 12:30:05

 reading test file... done

 reading file with vectors... done
 processing... 100 % / 100 %
The script is completed at 22-07-2022 12:30:18
Execution time = 0:00:13.361720
```

### Полученный файл json с оценками для запуска 1
```
[
    {
        "model": "sbert",
        "model_settings": {
            "mode": "sum_vectors"
        },
        "date": "22-07-2022 12:30:05",
        "dataset": "dataset_v1",
        "task": {
            "file": "../data/dataset_v1/test.csv",
            "method": "cosine similarity",
            "threshold": 0.6
        },
        "metrics": {
            "0": {
                "precision": 0.5783718104495748,
                "recall": 0.6380697050938338,
                "f1-score": 0.6067558954748247
            },
            "1": {
                "precision": 0.5623987034035657,
                "recall": 0.5,
                "f1-score": 0.5293668954996187
            }
        }
    },
    ...
]
```
### Пример запуска 2

`python3 doc_sem_matching.py ../notebooks/vectors11_dataset_v1_20179.csv 11 ../data/dataset_v1/test.csv ../notebooks 0.6 1 0.1 -d dataset_v1`

- `../notebooks/vectors11_dataset_v1_20179.csv` - путь до файла с векторами
- `11` - метод формирования векторов
- `../data/dataset_v1/test.csv` - путь до файла с тестовыми значениями
- `../notebooks` - путь до директории, в которой будет файл .json с оценками
- `0.6` - начальное значение порогового значения
- `1` - конечное значение порогового значения
- `0.1` - шаг увеличения порогового значения
- `-d dataset_v1` - название датасета, по которому формировались векторы

### Вывод в консоль для запуска 2

```
    ____                         _____                             __  ___           __            __  
   / __ \  ____   _____         / ___/  ___    ____ ___           /  |/  /  ____ _  / /_  _____   / /_ 
  / / / / / __ \ / ___/ ______  \__ \  / _ \  / __ `__ \ ______  / /|_/ /  / __ `/ / __/ / ___/  / __ \
 / /_/ / / /_/ // /__  /_____/ ___/ / /  __/ / / / / / //_____/ / /  / /  / /_/ / / /_  / /__   / / / /
/_____/  \____/ \___/         /____/  \___/ /_/ /_/ /_/        /_/  /_/   \__,_/  \__/  \___/  /_/ /_/ 
                                                                                                       

The script is running...
__
The path to the file with vectors: vectors-longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500.csv
The vector formation method: 5
The path to the file with test values: ../data/dataset_v1/test.csv
The path to the directory for output with evaluations: ../notebooks
The threshold value from 0.6 to 1.0 with step 0.1
The name of the dataset by which the vectors were formed: dataset_v1
Using fine tune model: True
__
After successful execution of the program, the result of the program will be in the file: ../notebooks/evaluation-vectors-longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500.json
__
The script was launched at 22-07-2022 13:22:16

 reading test file... done

 reading file with vectors... done
 processing... 100 % / 100 %
The script is completed at 22-07-2022 13:22:38
Execution time = 0:00:21.968592
```

### Полученный файл json с оценками для запуска 2

```
[
    {
        "model": "vectors-longformer-base-4096-epoch2-lr3e-5-cosemblos-dataset-train-3500",
        "model_settings": {
            "mode": "cls_token",
            "size": 4096
        },
        "date": "22-07-2022 13:22:18",
        "dataset": "dataset_v1",
        "task": {
            "file": "../data/dataset_v1/test.csv",
            "method": "cosine similarity",
            "threshold": 0.6
        },
        "metrics": {
            "0": {
                "precision": 1.0,
                "recall": 0.0013404825737265416,
                "f1-score": 0.0026773761713520753
            },
            "1": {
                "precision": 0.48227936066712995,
                "recall": 1.0,
                "f1-score": 0.6507266760431317
            }
        }
    },
    ...
]
```

