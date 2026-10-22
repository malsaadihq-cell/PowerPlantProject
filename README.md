# ♨️ Ideal Reheat Rankine Cycle Simulator

محاكاة تفاعلية لدورة **Reheat Rankine** المثالية، مبنية على المثال 10-4 من كتاب
Cengel (Thermodynamics). تحسب الحالات الست، موازنات الطاقة، الكفاءة الحرارية،
والرطوبة عند خروج التوربين، وترسم مخطط **T-s** تلقائيًا.

An interactive simulator for the **ideal reheat Rankine cycle** (based on Cengel,
Example 10-4). It computes the six states, energy balances, thermal efficiency, and
exit moisture, and draws the **T-s diagram** automatically.

---

## 🚀 Live Demo
<!-- بعد النشر على Streamlit Cloud، حط رابط التطبيق هنا -->
<!-- After deploying, paste your app URL here -->
> أضِف الرابط هنا بعد النشر.

---

## ✨ المميزات / Features
- إدخال مباشر لـ 5 متغيرات: ضغط الغلاية، حرارة الدخول، ضغط إعادة التسخين، حرارة إعادة التسخين، ضغط المكثّف.
- جدول بالحالات الست (P, T, h, s, x).
- حساب `q_in`, `q_out`, `w_net`, `η_th` ونسبة الرطوبة.
- تحذير تلقائي إذا تجاوزت الرطوبة الحد المسموح.
- مخطط T-s مع قبة التشبّع ومسارات الدورة.

خصائص البخار محسوبة من مكتبة **CoolProp** (IAPWS-95).

---

## 💻 التشغيل محليًا / Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## ☁️ النشر على Streamlit Community Cloud
1. ارفع هذا المستودع إلى GitHub.
2. ادخل على [share.streamlit.io](https://share.streamlit.io) وسجّل بحساب GitHub.
3. اختر **New app** → حدّد المستودع والفرع والملف `streamlit_app.py`.
4. اضغط **Deploy**. الـ Cloud بيقرأ `requirements.txt` ويركّب كل المكتبات تلقائيًا.

---

## 📁 محتويات المستودع / Repo structure
```
.
├── streamlit_app.py     # التطبيق الرئيسي
├── requirements.txt     # المكتبات المطلوبة
├── README.md
└── .gitignore
```

## 📝 ملاحظة
القيم تطابق المثال 10-4 تقريبًا (η = 45.0%، الرطوبة = 10.4%). الفروق البسيطة عن
الكتاب سببها اختلاف جداول البخار المستخدمة (CoolProp مقابل جداول الكتاب).
