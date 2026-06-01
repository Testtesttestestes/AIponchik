import cv2
import numpy as np
import pytesseract
import json
import os
import glob

# Если Tesseract не добавлен в PATH, раскомментируйте и укажите путь:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class GameBoardParser:
    def __init__(self):
        # Настройки для сетки (нужно будет подогнать под разрешение вашего телефона)
        # Координаты X, Y левого верхнего угла сетки, ширина и высота сетки
        self.grid_x = 100
        self.grid_y = 750
        self.grid_width = 880
        self.grid_height = 880
        self.cols = 7
        self.rows = 7
        
        # Размеры одной ячейки
        self.cell_w = self.grid_width // self.cols
        self.cell_h = self.grid_height // self.rows

    def extract_text(self, image, x, y, w, h):
        """Вырезает кусок картинки и пытается прочитать текст"""
        roi = image[y:y+h, x:x+w]
        # Переводим в ЧБ для лучшего распознавания
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Увеличиваем контрастность
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Используем tesseract, ограничиваем только цифрами
        config = '--psm 7 -c tessedit_char_whitelist=0123456789/'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip()

    def classify_cell(self, cell_img):
        """Определяет, что находится в ячейке, по среднему цвету в центре"""
        # Берем центр ячейки (отбрасываем края, чтобы избежать фона)
        h, w = cell_img.shape[:2]
        center = cell_img[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
        
        # Переводим в цветовое пространство HSV, оно лучше подходит для анализа цветов
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])

        # Простая эвристика по цветам (значения нужно будет откалибровать)
        # Примерные значения для распознавания:
        if avg_val < 50:
             # Если совсем темно, возможно это дырка (EMPTY)
             return "EMPTY"
        
        if avg_hue > 150 or avg_hue < 10: 
            return "red"        # Красные конфеты
        elif 10 <= avg_hue <= 25 and avg_sat > 100:
            return "muffin"     # Коричневый/Оранжевый кекс
        elif 25 < avg_hue <= 40:
            return "biscuit"    # Желтое печенье
        elif avg_val > 200 and avg_sat < 50:
            return "donut"      # Светлый пончик
        elif 10 <= avg_hue <= 20 and avg_sat < 100:
            return "chocolate"  # Темная конфета

        return "unknown" # Если цвет не подошел ни под одно правило

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {image_path}")
            return None

        # 1. Читаем метаданные (Координаты примерные! Нужно настроить)
        moves_text = self.extract_text(img, x=250, y=250, w=150, h=80)
        level_text = self.extract_text(img, x=600, y=250, w=150, h=80)

        # 2. Нарезаем сетку
        board = []
        for row in range(self.rows):
            row_data = []
            for col in range(self.cols):
                # Считаем координаты текущей ячейки
                cx = self.grid_x + (col * self.cell_w)
                cy = self.grid_y + (row * self.cell_h)
                
                # Вырезаем ячейку
                cell_img = img[cy:cy+self.cell_h, cx:cx+self.cell_w]
                
                # Классифицируем
                item_type = self.classify_cell(cell_img)
                row_data.append(item_type)
            board.append(row_data)

        # 3. Собираем JSON
        game_state = {
            "gameState": {
                "movesLeft": moves_text if moves_text else "0",
                "level": level_text if level_text else "0",
                "targets": {
                    "muffin": "TODO", # Аналогично вырезаем области с целями
                    "biscuit": "TODO",
                    "ice": "TODO"
                }
            },
            "board": board
        }
        
        return game_state

if __name__ == "__main__":
    parser = GameBoardParser()
    
    # Ищем все .jpg файлы в папке test_images
    images = glob.glob(os.path.join("test_images", "*.jpg"))
    
    for img_path in images:
        print(f"Обработка: {img_path}")
        result_json = parser.process_image(img_path)
        
        if result_json:
            # Сохраняем результат в файл с тем же именем, но .json
            out_path = img_path.replace(".jpg", ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
            print(f"Сохранено в: {out_path}")
