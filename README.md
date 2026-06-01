# AIponchik: подсказчик хода без автокликов

Проект разбирает скриншот поля Cookie Cats-подобной игры, оценивает цепочки и показывает рассчитанный лучший ход. Скрипт **не отправляет на телефон нажатия**: он только читает кадр и сохраняет JSON, debug-изображение и изображение с оверлеем рекомендуемой цепочки.

## Вариант 1: через scrcpy-трансляцию

1. Подключите телефон по USB, включите USB debugging и проверьте, что устройство видно:

   ```bash
   adb devices
   ```

2. Поднимите виртуальное V4L2-устройство и запустите scrcpy без управления телефоном:

   ```bash
   sudo modprobe v4l2loopback video_nr=2 card_label=scrcpy exclusive_caps=1
   scrcpy --v4l2-sink=/dev/video2 --no-control
   ```

   `--no-control` важен для безопасного режима: scrcpy показывает экран, но не принимает ввод мышью/клавиатурой как управление телефоном.

3. Откройте игру на телефоне и остановитесь на поле, где нужен совет. Затем в отдельном терминале запустите анализ одного стабильного кадра из scrcpy-потока:

   ```bash
   python game_parser.py --stream /dev/video2 --out test_images/live.json --overlay test_images/live_MOVE.jpg
   ```

4. Результаты:

   * `test_images/live.json` — распознанное состояние и список лучших ходов.
   * `test_images/live_DEBUG.jpg` — отладочная разметка распознанных клеток.
   * `test_images/live_MOVE.jpg` — скриншот с нарисованной рекомендуемой цепочкой; это и есть рассчитанный ход, без автотапов.

## Вариант 2: напрямую через ADB screencap

Если V4L2 недоступен, можно не подключаться к scrcpy-потоку, а взять стабильный кадр напрямую через ADB:

```bash
python game_parser.py --adb --out test_images/adb_capture.json --overlay test_images/adb_MOVE.jpg
```

Этот режим тоже ничего не нажимает на телефоне: используется только `adb exec-out screencap -p` для чтения изображения.

## Проверка на сохранённом скриншоте

Для локальной проверки алгоритма без телефона:

```bash
python game_parser.py --image test_images/Screenshot_2026-06-01-13-30-11-51.jpg --out test_images/check.json --overlay test_images/check_MOVE.jpg
```

В консоли будет напечатан лучший ход, а в JSON в поле `bestMove.path` будет путь по клеткам `row`/`col`.
