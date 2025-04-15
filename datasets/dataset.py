import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# تابع برای تولید داده‌های سری‌زمانی
def generate_time_series_data(num_rows, start_date, freq="10s"):
    """
    Generates a time series index for the given number of rows.
    """
    date_range = pd.date_range(start=start_date, periods=num_rows, freq=freq)
    return date_range

# تابع برای تولید داده‌های سنسورهای لرزه‌ای
def generate_seismic_data(num_rows):
    """
    Generates synthetic seismic sensor data (e.g., vibration amplitude).
    """
    timestamps = generate_time_series_data(num_rows, "2025-01-01")
    vibration_amplitude = np.random.normal(loc=0.5, scale=0.1, size=num_rows)  # مقادیر نرمال شده
    noise_level = np.random.uniform(low=0.01, high=0.05, size=num_rows)  # نویز تصادفی
    return pd.DataFrame({
        "timestamp": timestamps,
        "vibration_amplitude": vibration_amplitude + noise_level
    })

# تابع برای تولید داده‌های سنسورهای درون‌چاهی
def generate_downhole_data(num_rows):
    """
    Generates synthetic downhole sensor data (e.g., pressure and temperature).
    """
    timestamps = generate_time_series_data(num_rows, "2025-01-01")
    pressure = np.random.uniform(low=500, high=1000, size=num_rows)  # فشار در PSI
    temperature = np.random.uniform(low=50, high=200, size=num_rows)  # دما در درجه سانتی‌گراد
    return pd.DataFrame({
        "timestamp": timestamps,
        "pressure_psi": pressure,
        "temperature_c": temperature
    })

# تابع برای تولید داده‌های IoT و محیطی
def generate_iot_environmental_data(num_rows):
    """
    Generates synthetic IoT and environmental sensor data.
    """
    timestamps = generate_time_series_data(num_rows, "2025-01-01")
    gas_leak_level = np.random.exponential(scale=0.1, size=num_rows)  # سطح نشت گاز
    soil_moisture = np.random.uniform(low=10, high=50, size=num_rows)  # رطوبت خاک
    return pd.DataFrame({
        "timestamp": timestamps,
        "gas_leak_level": gas_leak_level,
        "soil_moisture": soil_moisture
    })

# ترکیب همه داده‌ها
def generate_combined_dataset(num_rows):
    """
    Combines seismic, downhole, and IoT/environmental data into one dataset.
    """
    seismic_data = generate_seismic_data(num_rows)
    downhole_data = generate_downhole_data(num_rows)
    iot_data = generate_iot_environmental_data(num_rows)

    # ترکیب داده‌ها بر اساس timestamp
    combined_data = pd.concat([seismic_data, downhole_data.drop(columns=["timestamp"]), iot_data.drop(columns=["timestamp"])], axis=1)
    return combined_data

# تولید ۱۰ میلیون داده
num_rows = 10_000_000
dataset = generate_combined_dataset(num_rows)

# ذخیره داده‌ها در فایل CSV
dataset.to_csv("synthetic_sensor_data.csv", index=False)

print(f"Dataset with {num_rows} rows has been generated and saved to 'synthetic_sensor_data.csv'.")