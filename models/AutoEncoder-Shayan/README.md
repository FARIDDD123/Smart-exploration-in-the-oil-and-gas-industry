# اتوانکودر لرزه‌نگاری ساده
## توضیحات
این پروژه یک اتوانکودر ساده برای پردازش داده‌های لرزه‌نگاری با استفاده از scikit-learn پیاده‌سازی کرده است. این ابزار برای کاهش نویز در داده‌های لرزه‌نگاری طراحی شده است و از فرمت‌های SEG-Y و CSV پشتیبانی می‌کند.

## پیش‌نیازها
```
numpy>=1.19.2
scikit-learn>=1.3.0
pandas>=1.3.0
matplotlib>=3.4.3
segyio>=1.9.7
```

## نحوه نصب
۱. ابتدا مخزن را کلون کنید:
```bash
git clone [آدرس مخزن]
cd [نام پوشه پروژه]
```

۲. محیط مجازی پایتون را ایجاد و فعال کنید:
```bash
python -m venv venv
# در ویندوز
venv\Scripts\activate
# در لینوکس/مک
source venv/bin/activate
```

۳. وابستگی‌ها را نصب کنید:
```bash
pip install -r requirements.txt
```

## نحوه استفاده
برای اجرای برنامه با داده‌های واقعی:
```bash
# برای فایل‌های SEG-Y
python simple_seismic_autoencoder.py --input data.segy --noise 0.2

# برای فایل‌های CSV
python simple_seismic_autoencoder.py --input data.csv --noise 0.2
```

برای اجرا با داده‌های مصنوعی:
```bash
python simple_seismic_autoencoder.py --synthetic
```

## فرمت‌های ورودی پشتیبانی شده
- فایل‌های SEG-Y (`.segy` یا `.sgy`)
- فایل‌های CSV (`.csv`)
  - هر ستون یک ردلرزه است
  - اگر هر سطر یک ردلرزه باشد، داده به صورت خودکار ترانهاده می‌شود

## خروجی‌ها
برنامه دو فایل خروجی تولید می‌کند:
- `denoising_results.png`: نمودار مقایسه‌ای داده اصلی، نویزی و بازسازی شده
- `example_results.npz`: فایل حاوی داده‌های عددی نتایج

## ساختار کد
- `SimpleSeismicAutoencoder`: کلاس اصلی برای پیاده‌سازی اتوانکودر
- `read_segy_file`: تابع خواندن فایل‌های SEG-Y
- `read_csv_file`: تابع خواندن فایل‌های CSV
- `generate_synthetic_seismic`: تابع تولید داده‌های مصنوعی لرزه‌نگاری
- `add_noise`: تابع اضافه کردن نویز گوسی به داده‌ها
- `plot_results`: تابع رسم نتایج
- `process_seismic_data`: تابع اصلی برای پردازش داده‌ها

## پارامترهای قابل تنظیم
- `trace_length`: طول هر ردلرزه (پیش‌فرض: ۱۰۰۰)
- `num_traces`: تعداد ردلرزه‌ها (پیش‌فرض: ۱۰۰)
- `latent_dim`: ابعاد فضای نهان (پیش‌فرض: ۳۲)
- `noise_level`: سطح نویز (پیش‌فرض: ۰.۲) 