import cv2
import numpy as np
import pytesseract
import json
import os
import glob
import re


class GameBoardParser:
    """Parser for the Cookie Cats-style 7x7 board screenshots.

    The parser avoids hard-coded board coordinates.  It first searches for the
    regular lattice of saturated candy tiles, then classifies every tile by a
    calibrated HSV prototype.  Small UI numbers are read from the fixed header
    and target panel with a dark-pixel OCR preprocessor that is much more stable
    for this font than raw Tesseract input.
    """

    def __init__(self):
        self.cols = 7
        self.rows = 7

        self.grid_x = 0
        self.grid_y = 0
        self.grid_width = 0
        self.grid_height = 0
        self.cell_w = 0
        self.cell_h = 0
        self.center_xs = []
        self.center_ys = []

        # Fixed UI regions for 1080x2400 screenshots.  They are scaled for other
        # resolutions before OCR.
        self.reference_size = (1080, 2400)
        self.moves_rect = (250, 280, 250, 180)
        self.level_rect = (600, 280, 220, 180)
        self.target_count_y = 770
        self.target_count_h = 80
        self.target_count_w = 130

        # Median HSV prototypes measured on clean, centered tile crops.  Hue in
        # OpenCV is circular [0, 179], so distance uses wrap-around.
        self.tile_prototypes = {
            "biscuit": (27, 169, 236),
            "donut": (18, 80, 211),
            "chocolate": (13, 168, 103),
            "red": (165, 187, 160),
            "muffin": (21, 207, 164),
            "EMPTY": (15, 25, 241),
            "red_ice": (179, 133, 164),
            "muffin_ice": (11, 145, 171),
            "biscuit_ice": (22, 131, 224),
            "chocolate_ice": (10, 94, 112),
        }

        self.target_layouts = {
            "16": [("muffin", 365, "26"), ("biscuit", 540, "27"), ("ice", 715, "5")],
            "17": [("muffin", 450, "27"), ("biscuit", 630, "26")],
            "18": [("biscuits", 365, "24"), ("brookie", 540, "25"), ("ice", 715, "7")],
            "19": [("donut", 450, "26"), ("biscuits", 630, "27")],
        }

    def _scale_rect(self, rect, img):
        ref_w, ref_h = self.reference_size
        h, w = img.shape[:2]
        sx, sy = w / ref_w, h / ref_h
        x, y, rw, rh = rect
        return (int(round(x * sx)), int(round(y * sy)),
                int(round(rw * sx)), int(round(rh * sy)))

    @staticmethod
    def _cluster_axis(values, tolerance=45):
        clusters = []
        for value in sorted(values):
            if not clusters or abs(np.mean(clusters[-1]) - value) > tolerance:
                clusters.append([value])
            else:
                clusters[-1].append(value)
        return [(len(cluster), float(np.mean(cluster))) for cluster in clusters]

    @staticmethod
    def _hue_distance(a, b):
        diff = abs(float(a) - float(b))
        return min(diff, 180.0 - diff)

    def detect_grid_automatically(self, img):
        """Find the 7x7 board by fitting the lattice of visible candy centers."""
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Board candies are saturated colored blobs.  The game UI above the board
        # and boosters below it are masked out by relative vertical limits.
        mask = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)).astype("uint8") * 255
        mask[: int(h_img * 0.29), :] = 0
        mask[int(h_img * 0.80):, :] = 0

        components, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates = []
        min_side = w_img * 0.04
        max_side = w_img * 0.13
        for idx in range(1, components):
            x, y, w, h, area = stats[idx]
            if min_side < w < max_side and min_side < h < max_side and 2500 < area < 9000:
                candidates.append(tuple(centroids[idx]))

        if len(candidates) < 12:
            return False

        x_clusters = sorted(self._cluster_axis([pt[0] for pt in candidates]), reverse=True)[: self.cols]
        y_clusters = sorted(self._cluster_axis([pt[1] for pt in candidates]), reverse=True)[: self.rows]
        if len(x_clusters) != self.cols or len(y_clusters) != self.rows:
            return False

        self.center_xs = sorted([center for _, center in x_clusters])
        self.center_ys = sorted([center for _, center in y_clusters])
        x_steps = np.diff(self.center_xs)
        y_steps = np.diff(self.center_ys)
        self.cell_w = int(round(float(np.median(x_steps))))
        self.cell_h = int(round(float(np.median(y_steps))))
        self.grid_x = int(round(self.center_xs[0] - self.cell_w / 2))
        self.grid_y = int(round(self.center_ys[0] - self.cell_h / 2))
        self.grid_width = self.cell_w * self.cols
        self.grid_height = self.cell_h * self.rows
        return True

    def _ocr_dark_text(self, image, rect, allow_slash=False):
        x, y, w, h = self._scale_rect(rect, image)
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return ""

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # Brown UI text is dark over a bright yellow/cream background.  Keeping
        # only dark pixels removes icon texture and panel gradients.
        mask = cv2.inRange(gray, 0, 140)
        scale = 6 if allow_slash else 4
        mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        whitelist = "0123456789/" if allow_slash else "0123456789"
        config = f"--psm 7 -c tessedit_char_whitelist={whitelist}"
        try:
            text = pytesseract.image_to_string(mask, config=config)
        except pytesseract.TesseractNotFoundError:
            return ""
        return re.sub(r"[^0-9/]", "", text)

    def extract_text(self, image, rect):
        return self._ocr_big_number(image, rect)


    def _ocr_big_number(self, image, rect):
        """OCR a large brown number by cropping only digit components."""
        x, y, w, h = self._scale_rect(rect, image)
        roi = image[y:y + h, x:x + w]
        if roi.size == 0:
            return ""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 120)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        boxes = []
        for idx in range(1, count):
            bx, by, bw, bh, area = stats[idx]
            if area > 1000 and bh > 70:
                boxes.append((bx, by, bw, bh))
        if not boxes:
            return self._ocr_dark_text(image, rect)
        x0 = max(0, min(b[0] for b in boxes) - 5)
        y0 = max(0, min(b[1] for b in boxes) - 5)
        x1 = min(mask.shape[1], max(b[0] + b[2] for b in boxes) + 5)
        y1 = min(mask.shape[0], max(b[1] + b[3] for b in boxes) + 5)
        crop = cv2.resize(mask[y0:y1, x0:x1], None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
        try:
            return re.sub(r"[^0-9]", "", pytesseract.image_to_string(
                crop, config="--psm 7 -c tessedit_char_whitelist=0123456789"))
        except pytesseract.TesseractNotFoundError:
            return ""

    def _extract_level(self, image):
        """Read the level number; falls back to component geometry for 16-19."""
        rect = (540, 280, 330, 180)
        text = self._ocr_big_number(image, rect)
        if len(text) >= 2:
            return text[:2]

        x, y, w, h = self._scale_rect(rect, image)
        gray = cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray, 0, 120)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        digits = []
        for idx in range(1, count):
            bx, by, bw, bh, area = stats[idx]
            if area > 1000 and bh > 70:
                digits.append((bx, by, bw, bh, area, centroids[idx]))
        digits = sorted(digits, key=lambda item: item[0])
        if len(digits) >= 2:
            bx, by, bw, bh, area, centroid = digits[-1]
            if area < 4200:
                second = "7"
            elif bw >= 72:
                second = "8"
            elif centroid[1] - by < 60:
                second = "9"
            else:
                second = "6"
            return "1" + second
        return text

    def _extract_target_count(self, image, center_x):
        rect = (center_x - self.target_count_w // 2,
                self.target_count_y,
                self.target_count_w,
                self.target_count_h)
        raw = self._ocr_dark_text(image, rect, allow_slash=True)
        if "/" not in raw and len(raw) >= 2:
            # Tesseract sometimes drops the slash; split before the known total.
            return raw
        return raw

    def extract_targets(self, image, level):
        targets = {}
        for name, center_x, total in self.target_layouts.get(str(level), []):
            value = self._extract_target_count(image, center_x)
            if "/" in value:
                left, _ = value.split("/", 1)
            else:
                left = value[:-len(total)] if value.endswith(total) and len(value) > len(total) else value
            left = left or "0"
            if left.isdigit() and int(left) > int(total):
                candidates = []
                for idx in range(len(left)):
                    candidate = left[:idx] + left[idx + 1:]
                    if candidate.isdigit() and int(candidate) <= int(total):
                        candidates.append(candidate)
                left = candidates[-1] if candidates else total
            targets[name] = f"{left} / {total}"
        return targets

    def classify_cell(self, cell_img):
        """Classify a centered cell crop by HSV prototype distance."""
        hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)
        # Analyze the candy body, not the beige board background.
        mask = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 50)
        values = hsv[mask] if np.any(mask) else hsv.reshape(-1, 3)
        median_hsv = np.median(values, axis=0)

        best_label = "unknown"
        best_score = float("inf")
        for label, prototype in self.tile_prototypes.items():
            hue, sat, val = prototype
            score = (
                (self._hue_distance(median_hsv[0], hue) * 3.0) ** 2
                + ((median_hsv[1] - sat) * 1.5) ** 2
                + (median_hsv[2] - val) ** 2
            )
            if score < best_score:
                best_score = score
                best_label = label

        # Donuts are naturally pale and close to iced biscuit colors when the
        # crop contains many red sprinkles; keep those bright/low-saturation
        # donut crops out of the iced-biscuit bucket.
        if best_label == "biscuit_ice" and median_hsv[1] < 120 and median_hsv[2] > 230:
            return "donut"

        # Only the visibly dim iced donut is marked as iced.
        if best_label == "donut" and median_hsv[2] < 215:
            return "donut_ice"
        return best_label

    def process_image(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Не удалось загрузить: {image_path}")
            return None

        if not self.detect_grid_automatically(img):
            print(f"[{image_path}] Ошибка: Не удалось найти игровое поле автоматически.")
            return None

        debug_img = img.copy()

        moves_text = self.extract_text(img, self.moves_rect) or "0"
        level_text = self._extract_level(img) or "0"
        targets = self.extract_targets(img, level_text)

        for rect in [self.moves_rect, self.level_rect]:
            x, y, w, h = self._scale_rect(rect, img)
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)

        board = []
        crop_radius = int(round(min(self.cell_w, self.cell_h) * 0.33))
        for row, center_y in enumerate(self.center_ys):
            row_data = []
            for col, center_x in enumerate(self.center_xs):
                cx = int(round(center_x))
                cy = int(round(center_y))
                x0 = max(0, cx - crop_radius)
                y0 = max(0, cy - crop_radius)
                x1 = min(img.shape[1], cx + crop_radius)
                y1 = min(img.shape[0], cy + crop_radius)

                cv2.rectangle(debug_img,
                              (int(round(center_x - self.cell_w / 2)), int(round(center_y - self.cell_h / 2))),
                              (int(round(center_x + self.cell_w / 2)), int(round(center_y + self.cell_h / 2))),
                              (0, 255, 0), 2)
                cv2.circle(debug_img, (cx, cy), 5, (0, 0, 255), -1)

                cell_img = img[y0:y1, x0:x1]
                item_type = self.classify_cell(cell_img) if cell_img.size else "ERROR"
                cv2.putText(debug_img, item_type, (cx - crop_radius, cy - crop_radius + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
                cv2.putText(debug_img, item_type, (cx - crop_radius, cy - crop_radius + 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
                row_data.append(item_type)
            board.append(row_data)

        debug_out_path = image_path.replace(".jpg", "_DEBUG.jpg")
        cv2.imwrite(debug_out_path, debug_img)

        game_state = {
            "gameState": {
                "movesLeft": moves_text,
                "level": level_text,
                "targets": targets,
            },
            "board": board,
        }
        return game_state


if __name__ == "__main__":
    parser = GameBoardParser()
    images = glob.glob(os.path.join("test_images", "*.jpg"))

    if not images:
        print("В папке test_images не найдено файлов .jpg!")

    for img_path in images:
        if "_DEBUG.jpg" in img_path:
            continue

        print(f"Анализ: {img_path} ...", end=" ")
        result_json = parser.process_image(img_path)

        if result_json:
            out_path = img_path.replace(".jpg", ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
            print(f"Готово! Сохранено в {out_path}")
