# GigaAM ASR — оффлайн и realtime распознавание речи

Speech-to-text на базе [`onnx-asr`](https://pypi.org/project/onnx-asr/) с моделями [GigaAM v3](https://istupakov.github.io/onnx-asr/usage/). Работает без PyTorch, модели грузятся с Hugging Face или из локального каталога.

## Модели

| Модель | Назначение |
|--------|-----------|
| `gigaam-v3-e2e-ctc` | Основной вариант. Текст с пунктуацией и нормализацией. Подходит для субтитров и расшифровки созвонов. |
| `gigaam-v3-ctc` | Запасной вариант. «Сырой» вывод CTC без пунктуации и нормализации. Для сравнения. |

## Установка (Fedora 43)

```bash
# 1. Системные зависимости
sudo dnf install -y python3.13 python3-pip git ffmpeg

# 2. Окружение
mkdir -p ~/src/gigaam-test && cd ~/src/gigaam-test
python3.13 -m venv .venv
source .venv/bin/activate

# 3. Зависимости Python
python -m pip install -U pip setuptools wheel
python -m pip install "onnx-asr[cpu,hub]==0.11.0" "onnxruntime!=1.24.1" rich numpy
```

> **Важно:** `onnxruntime 1.24.1` несовместим с `onnx-asr` — версия явно исключена.

### Проверка установки

```bash
python -c "
import onnx_asr, onnxruntime, sys
print(f'Python:       {sys.version.split()[0]}')
print(f'onnx-asr:     {onnx_asr.__version__}')
print(f'onnxruntime:  {onnxruntime.__version__}')
"
```

## Быстрый старт

### CLI

На первом запуске модель скачается с Hugging Face (~зависит от модели).

```bash
# Подготовить тестовый WAV (моно 16 kHz PCM16)
ffmpeg -i input.m4a -ac 1 -ar 16000 -c:a pcm_s16le test.wav

# Распознать (E2E — с пунктуацией)
onnx-asr gigaam-v3-e2e-ctc test.wav

# Распознать (сырой CTC — без пунктуации)
onnx-asr gigaam-v3-ctc test.wav
```

### Python-скрипт

```python
# test_gigaam.py
import sys
import onnx_asr

MODEL_NAME = "gigaam-v3-e2e-ctc"
MODEL_DIR  = "./models/gigaam-v3-e2e-ctc"

model = onnx_asr.load_model(MODEL_NAME, MODEL_DIR)
print(model.recognize(sys.argv[1]), flush=True)
```

```bash
python test_gigaam.py test.wav
```

## Длинное аудио (VAD)

Для файлов длиннее 20–30 секунд используйте VAD (Silero). Модель оборачивается через `.with_vad()` и выдаёт сегменты:

```python
# test_long.py
import sys
import onnx_asr

vad   = onnx_asr.load_vad("silero")
model = onnx_asr.load_model("gigaam-v3-e2e-ctc").with_vad(vad)

for seg in model.recognize(sys.argv[1]):
    print(seg.text, flush=True)
```

```bash
python test_long.py long.wav
```

## Realtime субтитры (PipeWire)

Проект включает `pipewire_asr.py` — realtime пайплайн с захватом аудиопотока из PipeWire:

- Перехват output stream произвольного приложения (по умолчанию — Brave)
- Автоматический поиск и переподключение к target
- Встроенный VAD на основе RMS-порога
- Инкрементальный вывод текста в stdout
- Rich-терминал с уровнем сигнала, partial/final текстом, историей

```bash
source .venv/bin/activate

# Субтитры для Brave (автопоиск stream по паттерну)
python pipewire_asr.py --target Brave

# Явный target (object.serial или node.name)
python pipewire_asr.py --target 4321

# Capture sink monitor (системный звук)
python pipewire_asr.py --capture-sink

# Другая модель
python pipewire_asr.py --model gigaam-v3-ctc
```

Параметры VAD и таймингов (при необходимости):

```sh
--silence-rms 0.0035       RMS-порог тишины
--tail-silence-sec 0.80    тишине после которой utterance завершается
--min-utt-sec 0.60         минимальная длина utterance
--max-utt-sec 6.0          максимальная длина utterance
--first-emit-sec 2.0       первый partial через N сек речи
--emit-every-sec 2.5       повторять partial каждые N сек
--preroll-sec 0.40         буфер до начала речи
--block-sec 0.20           размер аудиоблока
```

## Поддерживаемые аудиоформаты

`onnx-asr` ожидает WAV в PCM-форматах: `PCM_U8`, `PCM_16`, `PCM_24`, `PCM_32`. Умеет ресемплить. Для надёжности используйте моно PCM16 16 kHz.

Конвертация из произвольного формата:

```bash
ffmpeg -i input.mp3 -ac 1 -ar 16000 -c:a pcm_s16le output.wav
```


## Просмотр активных процессов с аудио:
```yaml
pw-dump | jq -r '                                    
  .[] |                           
  select(.info.props."media.class" == "Stream/Output/Audio") |
  [
    .info.props."object.serial",
    .info.props."node.name",
    .info.props."application.name",
    .info.state,
    .info.props."media.name"
  ] | @tsv'
```

## Решение проблем

| Проблема | Решение |
|----------|---------|
| `ModuleNotFoundError` | Активируй venv: `source .venv/bin/activate` |
| Ошибки `onnxruntime` | Проверь, что не установлена версия 1.24.1: `pip show onnxruntime` |
| «Зависает» на файле | Убедись, что файл — PCM WAV. Конвертируй через `ffmpeg`. |
| Длинный файл: результат рвётся или пустой | Используй VAD (`.with_vad()`) или ограничь файл до 20–30 сек. |

## Ссылки

- [onnx-asr на PyPI](https://pypi.org/project/onnx-asr/)
- [onnx-asr документация](https://istupakov.github.io/onnx-asr/usage/)
- [onnx-asr на GitHub](https://github.com/istupakov/onnx-asr)
