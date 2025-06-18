from icrawler.builtin import GoogleImageCrawler
import os
import cv2


emotions = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']

print("🔍 Starting image scraping...")

for emotion in emotions:
    folder = f"emotion_dataset_raw/{emotion}"
    os.makedirs(folder, exist_ok=True)

    google_crawler = GoogleImageCrawler(storage={'root_dir': folder})
    google_crawler.crawl(keyword=f"{emotion} face expression", max_num=500)
    print(f"✅ Downloaded images for: {emotion}")

print("🎉 Image scraping completed!")

print("\n🧹 Starting preprocessing...")

# Load face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

input_dir = 'emotion_dataset_raw'
output_dir = 'emotion_dataset_cleaned'
os.makedirs(output_dir, exist_ok=True)

for emotion in os.listdir(input_dir):
    input_emotion_path = os.path.join(input_dir, emotion)
    output_emotion_path = os.path.join(output_dir, emotion)
    os.makedirs(output_emotion_path, exist_ok=True)

    count = 0
    for file in os.listdir(input_emotion_path):
        try:
            img_path = os.path.join(input_emotion_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                face = cv2.resize(face, (48, 48))
                output_path = os.path.join(output_emotion_path, f"{emotion}_{count}.jpg")
                cv2.imwrite(output_path, face)
                count += 1
                break  # Save only the first face
        except Exception as e:
            continue

    print(f"✅ Processed {count} face images for: {emotion}")

print("\n🎯 Dataset ready in: emotion_dataset_cleaned/")
print("You can now use it for training your ML model! 🎉")
