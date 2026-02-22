# ✅ GitHub Upload Checklist — TalkNet-ASD

## الملفات الجاهزة للرفع:

### 📁 Root Folder
- [x] finaltalknet.py       ← الملف الرئيسي (webcam real-time)
- [x] demoTalkNet.py        ← الأصلي (video file demo)
- [x] talkNet.py            ← المودل
- [x] loss.py               ← دالة الخسارة
- [x] dataLoader.py         ← تحميل البيانات
- [x] trainTalkNet.py       ← التدريب
- [x] requirement.txt       ← المكتبات
- [x] README.md             ← شرح البروجكت
- [x] .gitignore            ← الملفات المستثناة
- [x] CONTRIBUTING.md       ← دليل المساهمة

### 📁 Folders (ارفعهم كما هم)
- [x] model/                ← كود المودل والـ face detector
- [x] utils/                ← أدوات مساعدة

### ❌ لا ترفعهم
- [ ] .venv/ أو talknet_env/    ← البيئة الافتراضية
- [ ] __pycache__/              ← Python cache
- [ ] pretrain_TalkSet.model    ← كبير جداً (في README رابط التحميل)
- [ ] test.mp4                  ← ملف فيديو
- [ ] exps/                     ← نتائج تجارب

## أوامر الرفع:
```bash
git init
git add .
git commit -m "Initial commit - TalkNet ASD Real-time Webcam"
git remote add origin https://github.com/YOUR_USERNAME/TalkNet-ASD.git
git push -u origin main
```
