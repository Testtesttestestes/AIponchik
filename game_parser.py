import cv2
import numpy as np
import pytesseract
import json
import os
import glob

# Укажите путь к Tesseract, если используете Windows и он не добавлен в PATH:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class GameBoardParser:
    def __init__(self):
        self.cols = 7
        self.rows = 7
        
        # Координаты сетки (будут перезаписаны автоопределением)
        self.grid_x = 0
        self.grid_y = 0
        self.grid_width = 0
        self.grid_height = 0
        self.cell_w = 0
        self.cell_h = 0

        # Координаты для распознавания текста (Вам нужно настроить их под свой экран!)
        # Формат: (x, y, ширина, высота)
        self.moves_rect = (250, 250, 150, 80) 
        self.level_rect = (600, 250, 150, 80)

    def detect_grid_automatically(self, img):
        """Автоматически находит игровое поле и вычисляет его геометрию."""
        h_img, w_img = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_centers = []
        min_size = int(w_img * 0.08)
        max_size = int(w_img * 0.15)
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if (min_size < w < max_size) and (min_size < h < max_size) and (0.7 < aspect_ratio < 1.3):
                valid_centers.append((x + w//2, y + h//2))
                
        if not valid_centers:
            return False

        xs = [pt[0] for pt in valid_centers]
        ys = [pt[1] for pt in valid_centers]
        
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        self.cell_w = (max_x - min_x) // (self.cols - 1)
        self.cell_h = (max_y - min_y) // (self.rows - 1)
        self.grid_x = min_x - (self.cell_w // 2)
        self.grid_y = min_y - (self.cell_h // 2)
        self.grid_width = self.cell_w * self.cols
        self.grid_height = self.cell_h * self.rows
        
        return True

    def extract_text(self, image, rect):
        """Вырезает указанную область и читает цифры."""
        x, y, w, h = rect
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return "0"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(thresh, config=config)
        return text.strip()

    def classify_cell(self, cell_img):
        """Определяет тип фишки и наличие льда."""
        h, w = cell_img.shape[:2]
        # Вырезаем только центр ячейки для анализа цвета (убираем фон по краям)
        center = cell_img[int(h*0.25):int(h*0.75), int(w*0.25):int(w*0.75)]
        
        # Проверка на пустую клетку (однородный фон)
        # Если дисперсия (разброс) пикселей низкая, значит внутри нет рельефа/фишки
        gray_center = cv2.cvtColor(center, cv2.COLOR_BGR2GRAY)
        if np.var(gray_center) < 100:
            return "EMPTY"

        # Проверка на лед (ищем яркие белые трещины)
        # Если много белых пикселей (высокая яркость и низкая насыщенность)
        hsv_full = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv_full, (0, 0, 200), (180, 40, 255))
        ice_ratio = cv2.countNonZero(white_mask) / (h * w)
        has_ice = ice_ratio > 0.05 # Если больше 5% площади ячейки белые трещины

        # Определение цвета
        hsv_center = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        avg_hue = np.mean(hsv_center[:, :, 0])
        avg_sat = np.mean(hsv_center[:, :, 1])
        avg_val = np.mean(hsv_center[:, :, 2])

        item_type = "unknown"
        if avg_hue > 150 or avg_hue < 10: 
            item_type = "red"        
        elif 10 <= avg_hue <= 25 and avg_sat > 100:
            item_type = "muffin"     
        elif 25 < avg_hue <= 40:
            item_type = "biscuit"    
        elif avg_val > 180 and avg_sat < 80:
            item_type = "donut"      
        elif 10 <= avg_hue <= 25 and avg_sat < 100:
            item_type = "chocolate"  

        # Добавляем суффикс льда, если он обнаружен
        if has_ice and item_type != "unknown":
            item_type += "_ice"

        return item_type

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить: {image_path}")
            return None

        # 1. Автоматически ищем сетку
        if not self.detect_grid_automatically(img):
            print(f"[{image_path}] Ошибка: Не удалось найти игровое поле автоматически.")
            return None

        debug_img = img.copy()

        # 2. Читаем текст
        moves_text = self.extract_text(img, self.moves_rect)
        level_text = self.extract_text(img, self.level_rect)

        # Рисуем рамки текста (синие)
        for rect in [self.moves_rect, self.level_rect]:
            x, y, w, h = rect
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        # 3. Нарезаем и классифицируем ячейки
        board = []
        for row in range(self.rows):
            row_data = []
            for col in range(self.cols):
                cx = self.grid_x + (col * self.cell_w)
                cy = self.grid_y + (row * self.cell_h)
                
                # Рисуем зеленую рамку ячейки
                cv2.rectangle(debug_img, (cx, cy), (cx + self.cell_w, cy + self.cell_h), (0, 255, 0), 2)
                # Рисуем красную точку в центре
                cv2.circle(debug_img, (cx + self.cell_w//2, cy + self.cell_h//2), 5, (0, 0, 255), -1)

                cell_img = img[cy:cy+self.cell_h, cx:cx+self.cell_w]
                
                if cell_img.size == 0 or cell_img.shape[0] != self.cell_h or cell_img.shape[1] != self.cell_w:
                    row_data.append("ERROR")
                    continue

                item_type = self.classify_cell(cell_img)
                
                # Пишем распознанный класс поверх ячейки на отладочной картинке
                cv2.putText(debug_img, item_type, (cx + 5, cy + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(debug_img, item_type, (cx + 5, cy + 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                row_data.append(item_type)
            board.append(row_data)

        # 4. Сохраняем отладочную картинку
        debug_out_path = image_path.replace(".jpg", "_DEBUG.jpg")
        cv2.imwrite(debug_out_path, debug_img)

        # 5. Формируем финальный JSON
        game_state = {
            "gameState": {
                "movesLeft": moves_text if moves_text else "0",
                "level": level_text if level_text else "0"
            },
            "board": board
        }
        
        return game_state

if __name__ == "__main__":
    parser = GameBoardParser()
    
    # Папка с тестовыми изображениями должна находиться рядом со скриптом
    images = glob.glob(os.path.join("test_images", "*.jpg"))
    
    if not images:
        print("В папке test_images не найдено файлов .jpg!")
        
    for img_path in images:
        # Пропускаем отладочные файлы, если они уже есть в папке
        if "_DEBUG.jpg" in img_path:
            continue
            
        print(f"Анализ: {img_path} ...", end=" ")
        result_json = parser.process_image(img_path)
        
        if result_json:
            out_path = img_path.replace(".jpg", ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
            print(f"Готово! Сохранено в {out_path}")