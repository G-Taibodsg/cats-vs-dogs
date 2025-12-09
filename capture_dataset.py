# optional: simple webcam capturer to collect images for each class interactively
import cv2
import os

def capture(out_dir='raw_data', class_name='cats', max_images=200):
    os.makedirs(os.path.join(out_dir, class_name), exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0
    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('capture - press s to save, q to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            path = os.path.join(out_dir, class_name, f'{count:04d}.jpg')
            cv2.imwrite(path, frame)
            print('Saved', path)
            count += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture()
