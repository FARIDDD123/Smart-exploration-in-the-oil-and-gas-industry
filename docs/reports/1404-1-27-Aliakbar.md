

**تاریخ: 27 فروردین 1404**  

## مهارت افزایی مستمر

#### تحلیل روی دیتاست titanic

در این پروژه، از دیتاست معروف تایتانیک استفاده شده که یکی از دیتاست‌های کلاسیک در حوزه یادگیری ماشین است. این دیتاست شامل اطلاعات مسافران کشتی تایتانیک می‌باشد و هدف پروژه پیش‌بینی احتمال نجات‌یافتن مسافران با توجه به ویژگی‌های آن‌ها است.

### داده های استفاده شده

فایل این پروژه که شامل اطلاعات زیر برای هر مسافر است:

Survived: آیا مسافر زنده مانده یا خیر (0 = خیر، 1 = بله)

Pclass: کلاس بلیط (1، 2 یا 3)

Sex: جنسیت

Age: سن

SibSp: تعداد خواهر و برادر یا همسر همراه

Parch: تعداد والدین یا فرزندان همراه

Fare: مبلغ پرداختی بلیط

Embarked: بندری که مسافر از آن سوار شده است


### پیش پردازش و مدل سازی

حذف ستون‌های غیرضروری 

تبدیل مقادیر متنی به عددی 

پر کردن مقادیر گمشده با مقادیر مناسب

میانگین برای ستون سن

پرکردن ستون براساس داده های پرتکرار

**تبدیل کردن داده ها به داده های آموزش و تست**

استفاده از گرید سرچ برای بهترین پارامتر برای مدل هایی که در نظر داریم و استفاده از مدل ها با استفاده از بهترین پارامتر ها و بررسی دقت مدل ها در پایان کار