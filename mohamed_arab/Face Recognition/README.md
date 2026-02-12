📌 Face & Object Detection + Face Embedding System
📖 Overview
هذا المشروع يقوم بـ:
✅ Face Detection باستخدام YOLOv8
✅ Object Detection باستخدام YOLO
✅ Face Embedding باستخدام FaceNet (512-D Vector)
✅ تخزين الـ Embeddings داخل ملف JSON
❌ لا يوجد Tracking حالياً
🛠️ Requirements
Python 3.9+
Git
(اختياري) GPU + CUDA لتسريع المعالجة
📦 1️⃣ تحميل المشروع
نسخ التعليمات البرمجية
Bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
🧪 2️⃣ إنشاء Virtual Environment
🔹 Windows
نسخ التعليمات البرمجية
Bash
python -m venv venv
venv\Scripts\activate
🔹 Linux / Mac
نسخ التعليمات البرمجية
Bash
python3 -m venv venv
source venv/bin/activate
📥 3️⃣ تثبيت المكتبات
أنشئ ملف requirements.txt وضع به:
نسخ التعليمات البرمجية
Txt
ultralytics
torch
torchvision
opencv-python
facenet-pytorch
numpy
ثم ثبتها:
نسخ التعليمات البرمجية
Bash
pip install -r requirements.txt
لو تستخدم GPU:
نسخ التعليمات البرمجية
Bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
▶️ 4️⃣ تشغيل المشروع
نسخ التعليمات البرمجية
Bash
python main.py
أو:
نسخ التعليمات البرمجية
Bash
python detect.py
📂 Project Structure
نسخ التعليمات البرمجية

🧠 How It Works
🔹 Step 1: Face Detection
YOLOv8 يحدد أماكن الوجوه داخل الصورة أو الفيديو.
🔹 Step 2: Crop Face
يتم قص الوجه من الإطار.
🔹 Step 3: Face Embedding
يتم تمرير الوجه إلى FaceNet.
يتم استخراج Vector بطول 512 قيمة.
🔹 Step 4: Save Embedding
يتم حفظه في JSON بهذا الشكل:
نسخ التعليمات البرمجية
Json
{
  "person_1": [0.0213, -0.5521, 0.3378, ...]
}
🔎 Important Notes
✅ FaceNet يجب أن يرجع 512 قيمة
تأكد أن الكود يحتوي على:
نسخ التعليمات البرمجية
Python
embedding = embedding.detach().cpu().numpy().flatten().tolist()
لو عدد القيم أقل:
ممكن يكون حصل slicing
أو حصل تقليل أبعاد
أو تم حفظه بشكل مختصر
🚀 Future Improvements (Planned)
🔄 إضافة Tracking (ByteTrack أو DeepSORT)
🧠 إضافة Face Recognition (مقارنة embeddings)
☁️ ربط Embeddings بـ ChromaDB Cloud
🌐 إنشاء API باستخدام FastAPI
👨‍💻 Author
Mohamed Arab