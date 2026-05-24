# TrashSort

Проект для автоматической классификации бытового мусора по изображению.

Модель определяет тип отходов по фотографии и относит изображение к одному из классов: cardboard, glass, metal, paper, plastic или trash.

Проект сделан как полноценный MLOps pipeline: обучение модели, управление данными, конфигурации, логирование экспериментов, API для инференса и воспроизводимое окружение.

---

## Описание проекта
В рамках проекта обучается модель компьютерного зрения, которая по изображению определяет категорию мусора.

Задача сформулирована как **multiclass image classification**.

### Классы

- cardboard
- glass
- metal
- paper
- plastic
- trash

### Входные данные

RGB изображение в формате `.jpg` / `.png`

Размер после препроцессинга:

```txt
3 × 224 × 224
```

### Выходные данные

Предсказанный класс мусора:

```json
{
  "class_id": 2,
  "class_name": "glass"
}
```

---

## Dataset

Используется датасет **TrashNet**.

Источник:

https://github.com/garythung/trashnet

Датасет содержит изображения различных видов бытового мусора, собранных в реальные категории отходов.

Пример структуры данных:

```txt
raw/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

### Особенности датасета

- сравнительно небольшой размер
- дисбаланс классов
- вариативность освещения и ракурсов
- шум в изображениях

---

## Архитектура модели

В качестве основной модели используется **ResNet18** из torchvision с предобученными весами (`ImageNet`).

Последний классификационный слой заменён на линейный слой с количеством выходов, равным числу классов.

Причины выбора:

- простая и устойчивая архитектура
- подходит для небольшого датасета
- легко воспроизводится

---

## Метрики

Для оценки качества используются:

- **Accuracy**
- **Train Loss**
- **Validation Loss**

Целевое качество:

```txt
Validation Accuracy > 75%
```

Так как датасет небольшой, целью проекта является стабильное обучение и воспроизводимость пайплайна, а не достижение SOTA.

Итоговые метрики:
<img width="348" height="74" alt="image" src="https://github.com/user-attachments/assets/5cbe13ba-38bb-4de6-bc83-e872e88c844c" />


---

## Технологии

Проект использует:

- PyTorch
- PyTorch Lightning
- Hydra
- MLflow
- DVC
- FastAPI
- Poetry
- Ruff
- pre-commit
- pytest

---

## Структура проекта

```txt
trashsort/
├── app/
│   └── main.py
│
├── configs/
│   ├── config.yaml
│   ├── train/
│   ├── model/
│   ├── data/
│   ├── mlflow/
│   └── inference/
│
├── trashsort/
│   ├── datamodule.py
│   ├── lightning_module.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── plots.py
│   └── utils.py
│
├── tests/
│   └── test_api.py
│
├── data/
├── models/
├── plots/
│
├── .github/workflows/
├── .pre-commit-config.yaml
├── pyproject.toml
├── poetry.lock
├── Dockerfile
└── README.md
```

---

# Setup

## 1. Клонирование репозитория

```bash
git clone <repository_url>

cd trashsort
```

---

## 2. Установка Poetry

Если Poetry ещё не установлен:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Проверка:

```bash
poetry --version
```

---

## 3. Установка зависимостей

```bash
poetry install
```

---

## 4. Активация окружения

Poetry 2.x:

```bash
poetry env activate
```

или запуск через:

```bash
poetry run
```

---

## 5. Установка pre-commit hooks

```bash
poetry run pre-commit install
```

Проверка:

```bash
poetry run pre-commit run -a
```

---

## 6. Загрузка данных через DVC

```bash
dvc pull
```

После этого датасет появится в:

```txt
data/raw/
```

---

# Training

## Запуск MLflow

В отдельном терминале:

```bash
poetry run mlflow ui --host 127.0.0.1 --port 8080
```

Интерфейс будет доступен:

```txt
http://127.0.0.1:8080
```

---

## Запуск обучения

Базовый запуск:

```bash
poetry run python -m trashsort.train
```

---

## Изменение гиперпараметров через Hydra

Например:

```bash
poetry run python -m trashsort.train train.epochs=20
```

или:

```bash
poetry run python -m trashsort.train train.batch_size=64
```

---

## Результат обучения

После обучения сохраняются:

### Checkpoint модели

```txt
models/checkpoints/
```

### Графики обучения

```txt
plots/
```

### Эксперименты MLflow

```txt
mlruns/
```

---

# Построение графиков

После обучения:

```bash
poetry run python -m trashsort.plots
```

Создаются графики:

- accuracy curve
- loss curve
- metrics summary

---


# Inference (через API или UI)

Для запуска сервиса:

```bash
poetry run uvicorn app.main:app --reload
```

Swagger UI:

```txt
http://127.0.0.1:8000/docs
```

Доступен endpoint:

```txt
POST /predict
```

Куда можно загрузить изображение и получить предсказание.

---

# DVC

Данные и модели хранятся через **DVC**, а не в Git.

Основные команды:

Добавить данные:

```bash
dvc add data/raw
```

Скачать данные:

```bash
dvc pull
```

Загрузить изменения:

```bash
dvc push
```

---

# Проверка качества кода

Проверка форматирования:

```bash
poetry run pre-commit run -a
```

---

