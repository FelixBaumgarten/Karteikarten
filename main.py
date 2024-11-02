import cv2
import numpy as np
import os

def crop_card(image_path, output_path):
    # Bild laden
    image = cv2.imread(image_path)
    if image is None:
        print(f"Fehler beim Laden des Bildes: {image_path}")
        return

    # Bild in Graustufen umwandeln und den Schwellenwert anwenden
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Konturen finden
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Größte Kontur finden (die Karteikarte)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)  # Gibt den rotierten Rechteckrahmen zurück
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Bild begradigen
        width, height = rect[1][0], rect[1][1]
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, M, (int(width), int(height)))

        cv2.imwrite(output_path, warped)
        print(f"Karteikarte zugeschnitten und gespeichert: {output_path}")

def process_folder(folder_path):
    # Alle Dateien im Ordner durchlaufen
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            output_path = os.path.join(folder_path, "processed_" + file_name)
            crop_card(file_path, output_path)

# Pfad zu eurem Bilderordner
image_folder_path = '/Users/dein/Name/Ordner'
process_folder(image_folder_path)
