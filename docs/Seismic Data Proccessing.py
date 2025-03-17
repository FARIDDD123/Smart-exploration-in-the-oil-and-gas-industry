from obspy import read, Stream
import numpy as np
import os


base_output_dir="C:/Users/Christopher/Desktop/New folder/PetroPalas/smart-exploration-in-the-oil-and-gass/data"


def get_region(x_coord, y_coord):
    
    if 60 <= x_coord <= 150 and 0 <= y_coord <= 90: 
        return "asia"
    elif -10 <= x_coord <= 60 and 35 <= y_coord <= 70: 
        return "europe"
    else:
        return "other"

snr_threshold=10
def calculate_snr(trace_data):
    
    signal = np.mean(np.abs(trace_data)**2)
    
    noise = np.var(trace_data)
    if noise == 0:  
        return float('inf')
    snr = 10 * np.log10(signal / noise) 
    return snr


try:
 segy_file = read(r"I:\SEAM_Interpretation_Challenge_1_Depth.sgy", format="SEGY")
 cleaned_stream = Stream()
 regions = {}

 for trace in segy_file:
  
    header=trace.stats.segy.trace_header
    data = trace.data

    try:
        x_coord=header.source_coordinate_x
        y_coord=header.source_coordinate_y
#        print(f"X: {x_coord}, Y: {y_coord}")
    except AttributeError:
        print(f"coordinate Not found:")
        x_coord, y_coord = 0, 0

    sampling_rate = trace.stats.sampling_rate
#    print(f"sampling rate {sampling_rate} hz")

    start_time = trace.stats.starttime
#    print(f"Date and Time of Start: {start_time}")
  
    raw_data = trace.data
  
    filtered_trace = trace.copy()  
    filtered_trace.filter("lowpass", freq=10.0) 
    filtered_data = filtered_trace.data
#    print(f"داده فیلترشده: {filtered_data[:10]}")

    region = get_region(x_coord, y_coord)


    if region not in regions:
        regions[region] = Stream()
    regions[region] += trace

    for region, region_stream in regions.items():
        region_dir = os.path.join(base_output_dir, region)
        os.makedirs(region_dir, exist_ok=True)
    
        output_file = os.path.join(region_dir, f"{region}_3d_2023.segy")
    
    
        region_stream.write(output_file, format='SEGY')
        print(f"File Saved: {output_file} با {len(region_stream)} trace")

    if np.any(np.isnan(data)) or np.all(data == 0):
        print(f"Trace {trace.stats.station} حذف شد: داده ناقص")
        continue

    snr = calculate_snr(data)
    print(f"Trace {trace.stats.station} - SNR: {snr:.2f} dB")
  
    if snr < snr_threshold:
        print(f"Trace {trace.stats.station} حذف شد: SNR پایین ({snr:.2f} dB)")
        continue
  
    cleaned_stream += trace 
    print(f"تعداد trace‌های باقی‌مونده: {len(cleaned_stream)}")

    if len(cleaned_stream) > 0:
        cleaned_stream.write(output_file, format='SEGY')
        print(f"داده‌های تمیز ذخیره شدند در: {output_file}")
    else:
        print("هیچ trace‌ای باقی نموند!")
    
except Exception as e:
 print(f"An error occurred: {e}")
 

# print(seismic)