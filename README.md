# AIponchik: подсказчик хода без автокликов

Проект разбирает скриншот поля Cookie Cats-подобной игры, оценивает цепочки и показывает рассчитанный лучший ход. Скрипт **не отправляет на телефон нажатия**: он только читает кадр и сохраняет JSON, debug-изображение и изображение с оверлеем рекомендуемой цепочки.

## Live GUI без перезапуска

Для постоянной работы на Windows запустите ADB-режим с графическим окном через готовый bat-скрипт:

```bat
scripts\run_adb_gui.bat
```

Он повторяет рекомендованный ADB screencap-подход из раздела ниже: ищет `adb.exe` в `AIPONCHIK_ADB`, затем в `PATH`, затем рядом с `scrcpy`; пишет результаты в `run_outputs`; открывает постоянное topmost-окно подсказки и не нажимает на телефон. Если нужно указать конкретное устройство или другую папку вывода:

```bat
set AIPONCHIK_SERIAL=emulator-5554
set AIPONCHIK_OUT_DIR=D:\aiponchik_runs
scripts\run_adb_gui.bat
```

Та же команда напрямую через Python:

```bash
python game_parser.py --adb --gui --out run_outputs/adb_capture.json --overlay run_outputs/adb_MOVE.jpg
```

Окно будет циклически показывать один из двух статусов:

* **Ожидание стабилизации экрана** — игра анимируется, меняется поле или кадры ещё отличаются сильнее порога `--stable-threshold`.
* **Стабильный экран: лучший ход рассчитан** — найдено поле, распознаны цели/оставшиеся ходы, а поверх стабильного кадра нарисована рекомендуемая цепочка.

Если стабильный кадр получен, но поле не найдено (например, открыто меню или экран перехода уровня), программа не падает: она пишет статус в GUI/консоль и ждёт следующее стабильное состояние. Закрыть live-режим можно клавишей `q` или `Esc` в окне.

Для scrcpy/V4L2 используется тот же режим:

```bash
python game_parser.py --stream /dev/video2 --gui --out run_outputs/live.json --overlay run_outputs/live_MOVE.jpg
```

Если GUI не нужен, но нужно не завершаться после одного анализа, используйте `--watch` вместо `--gui`.

Солвер учитывает распознанное число оставшихся ходов (`movesLeft`): для незавершённых целей он повышает приоритет цепочек, которые собирают достаточно целевых клеток «в темпе» оставшегося лимита ходов, и штрафует бесполезные нецелевые ходы при срочных целях. В JSON это видно в `analysis.movesLeftParsed`, а причины начисления/штрафов попадают в `bestMove.reasons`.

## Что делать, если окно scrcpy пикселит

Строки scrcpy вроде `Renderer: direct3d` и `Texture: 1080x2400` означают, что окно уже рендерится в полном размере текстуры. Пикселизация чаще всего появляется раньше — на этапе кодирования видеопотока телефоном. Напрямую «снять кадр из Direct3D renderer» текущий скрипт не умеет: это потребовало бы встраиваться в клиент scrcpy или менять сам scrcpy. Для алгоритма лучше использовать **ADB screencap**: он берёт PNG-скриншот напрямую с телефона, без сжатия видеопотока scrcpy, и не нажимает на экран.

Для более приятного ручного просмотра можно поднять битрейт scrcpy:

```bat
scripts\start_scrcpy_high_quality.bat
```

По умолчанию скрипт запускает read-only mirroring с `--no-control`, `--video-bit-rate=32M`, `--max-fps=60` и Direct3D renderer. Если всё ещё мылит, попробуйте H.265 и больший битрейт:

```bat
scripts\start_scrcpy_high_quality.bat 48M h265
```

Если конкретный телефон/декодер не любит H.265, вернитесь на H.264:

```bat
scripts\start_scrcpy_high_quality.bat 48M h264
```

## Парсер с нейронкой

После обучения можно запускать отдельный входной скрипт `neural_game_parser.py`. Он принимает те же основные режимы, что и `game_parser.py` (`--image`, `--adb`, `--stream`, `--gui`, `--watch`, `--overlay`, `--out`), но ранжирует ход обученной моделью из `models/policy_network.json`. По умолчанию нейронке передаются 32 лучших кандидата на ход:

```bash
python neural_game_parser.py --image test_images/Screenshot_2026-06-01-13-30-11-51.jpg --out run_outputs/neural_check.json --overlay run_outputs/neural_check_MOVE.jpg
```

Для ADB-подсказки используется тот же безопасный режим чтения скриншота без нажатий:

```bash
python neural_game_parser.py --adb --out run_outputs/neural_adb.json --overlay run_outputs/neural_adb_MOVE.jpg --show-overlay-window
```

Если модель лежит в другом месте или нужно изменить число кандидатов, задайте параметры явно:

```bash
python neural_game_parser.py --policy-model models/policy_network.json --policy-candidates 32 --image test_images/Screenshot_2026-06-01-13-30-11-51.jpg
```

В JSON-поле `analysis.ranker` будет `neuralPolicy`, `analysis.policyCandidateLimit` покажет лимит 32, а у ходов появятся `policyScore` и исходный `heuristicScore`.

## Быстро получить ход на Windows

1. Откройте игру на телефоне и оставьте поле неподвижным.
2. В отдельном терминале запустите:

   ```bat
   scripts\get_move_adb.bat
   ```

3. Скрипт сохранит картинку `run_outputs\adb_MOVE.jpg` и сразу покажет её в отдельном topmost-окне OpenCV. Это не настоящий слой внутри ADB: у ADB нет окна с изображением телефона. Практически это работает как «оверлей-подсказка» — окно можно положить поверх/рядом с scrcpy, а после просмотра закрыть любой клавишей в окне overlay. Также будут созданы:

   * `run_outputs\adb_capture.json` — распознанное состояние и список лучших ходов.
   * `run_outputs\adb_capture_DEBUG.jpg` — отладочная разметка распознанных клеток.
   * `run_outputs\adb_MOVE.jpg` — скриншот с рассчитанным ходом.

Если любой ADB bat-скрипт пишет, что не найден `adb.exe`, установите Android platform-tools или задайте путь вручную:

```bat
set AIPONCHIK_ADB=C:\Android\platform-tools\adb.exe
scripts\get_move_adb.bat
```

Если подключено несколько устройств, задайте серийник перед запуском:

```bat
set AIPONCHIK_SERIAL=YOUR_DEVICE_SERIAL
scripts\get_move_adb.bat
```

## Проверка на сохранённом скриншоте

```bat
scripts\get_move_from_image.bat test_images\Screenshot_2026-06-01-13-30-11-51.jpg
```

Или напрямую через Python:

```bash
python game_parser.py --image test_images/Screenshot_2026-06-01-13-30-11-51.jpg --out run_outputs/check.json --overlay run_outputs/check_MOVE.jpg
```

## Вариант через scrcpy/V4L2-поток на Linux

Если нужен именно OpenCV-поток из scrcpy, на Linux можно использовать V4L2 loopback:

```bash
sudo modprobe v4l2loopback video_nr=2 card_label=scrcpy exclusive_caps=1
scrcpy --v4l2-sink=/dev/video2 --no-control --video-bit-rate=32M
python game_parser.py --stream /dev/video2 --out run_outputs/live.json --overlay run_outputs/live_MOVE.jpg
```

`--no-control` важен для безопасного режима: scrcpy показывает экран, но не принимает ввод мышью/клавиатурой как управление телефоном.

## Вариант напрямую через ADB screencap

Если не нужны bat-скрипты, можно запустить Python-команду напрямую:

```bash
python game_parser.py --adb --out run_outputs/adb_capture.json --overlay run_outputs/adb_MOVE.jpg --show-overlay-window
```

Этот режим ничего не нажимает на телефоне: используется только `adb exec-out screencap -p` для чтения изображения. Если `adb.exe` не в `PATH`, добавьте `--adb-path C:\Android\platform-tools\adb.exe`.

В консоли будет напечатан лучший ход, а в JSON в поле `bestMove.path` будет путь по клеткам `row`/`col`.
