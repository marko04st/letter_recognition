import cv2
import numpy as np
from tensorflow.keras.models import load_model

model_path_latin = 'model_latin.h5'
model_path_cyrillic = 'model_kiril.h5'
IMG_SIZE = 28
LATIN_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CYRILLIC_CHARS = "АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЬЮЯ"
CANVAS_SIZE = 400
BRUSH_SIZE = 12
MARGIN = 15
is_cyrillic = True
drawing = False
ix, iy = -1, -1

# Load models
print("[INFO] Loading models...")
model_latin = load_model(model_path_latin)
model_cyrillic = load_model(model_path_cyrillic)
print("[INFO] Models loaded.")


def draw_circle(event, x, y, flags, param):
    global drawing, ix, iy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), BRUSH_SIZE)
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, (ix, iy), (x, y), (255, 255, 255), BRUSH_SIZE)


def predict_character(captured_img, use_cyrillic=False):
    gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)

    if coords is None:
        print("[DEBUG] No drawing detected.")
        return ""

    x, y, w, h = cv2.boundingRect(coords)
    roi = gray[max(0, y - MARGIN): min(CANVAS_SIZE, y + h + MARGIN),
          max(0, x - MARGIN): min(CANVAS_SIZE, x + w + MARGIN)]

    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    roi_reshaped = roi_resized.astype('float32') / 255.0
    roi_reshaped = roi_reshaped.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    model = model_cyrillic if use_cyrillic else model_latin
    preds = model.predict(roi_reshaped)[0]
    label_index = np.argmax(preds)

    return CYRILLIC_CHARS[label_index] if use_cyrillic else LATIN_CHARS[label_index]


if __name__ == "__main__":
    img = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
    cv2.namedWindow('Draw & Predict')
    cv2.setMouseCallback('Draw & Predict', draw_circle)
    predicted_text = ""

    while True:
        display_img = img.copy()
        alphabet = "Cyrillic" if is_cyrillic else "Latin"
        cv2.putText(display_img, f"Prediction: {predicted_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display_img, f"Alphabet: {alphabet} (Press 't' to toggle, +/- to adjust brush size)", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('Draw & Predict', display_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            predicted_text = predict_character(img, use_cyrillic=is_cyrillic)
            print(f"Predicted: {predicted_text}")
        elif key == ord('c'):
            img.fill(0)
            predicted_text = ""
        elif key == ord('t'):
            is_cyrillic = not is_cyrillic
            predicted_text = ""
        elif key == ord('+'):
            BRUSH_SIZE = min(BRUSH_SIZE + 2, 30)
        elif key == ord('-'):
            BRUSH_SIZE = max(BRUSH_SIZE - 2, 2)
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
