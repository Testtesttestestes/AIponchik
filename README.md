# AIponchik: подсказчик хода без автокликов

Проект разбирает скриншот поля Cookie Cats-подобной игры, оценивает цепочки и показывает рассчитанный лучший ход. Скрипт **не отправляет на телефон нажатия**: он только читает кадр и сохраняет JSON, debug-изображение и изображение с оверлеем рекомендуемой цепочки.

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

Если скрипт пишет, что не найден `adb.exe`, установите Android platform-tools или задайте путь вручную:

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
