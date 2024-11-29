import os
import pandas as pd
import seisbench.models as sbm
from obspy import read, read_inventory, UTCDateTime
from scipy import signal
import glob
import re
import numpy as np
from obspy import Stream
import torch


df_earthquake = pd.read_csv('/media/justin/Data2/AGU_Discrime/blast_SNR_WQSZ.csv')
filtered_df = df_earthquake[(df_earthquake['SNR_denoise'] >= 1.8)]

model = sbm.DeepDenoiser.from_pretrained("original")


fs = 100  # Hz
# Calculate nperseg and noverlap
nperseg = 256  # This is a simple approximation
noverlap = 128  # 50% overlap
temp_all = []
all_hour = []
for time, station in zip(filtered_df['Event'], filtered_df['Station']):
    os.chdir('/media/justin/Data2/AGU_Discrime/WQSZ_station_blast')
    t = UTCDateTime(time)
    print(t)
    event = t.strftime('%Y%m%d%H%M%S')
    os.chdir(event)
    file = '*' + station + '*'
    inv = read_inventory(file)
    os.chdir('/media/justin/Data2/AGU_Discrime/WQSZ_blast')
    os.chdir(event)
    st = read(file)
    for tr in st.select(station="M53A"):
        st.remove(tr)
    for tr in st.select(station="O53A"):
        st.remove(tr)
    for tr in st.select(station="P52A"):
        st.remove(tr)            
    for tr in st:
        if round(tr.stats.sampling_rate) == 100:
            pre_filt = [0.001, 0.002, 45, 50]
        if round(tr.stats.sampling_rate) == 40:
            pre_filt = [0.001, 0.002, 18, 20]
        tr.remove_response(inventory=inv,pre_filt=pre_filt,output="DISP",water_level=60,taper=True,taper_fraction=0.00001)
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("highpass", freq=1)
    st2 = model.annotate(st)
    temp_array = []
    for tr in st2:
        if tr.stats.npts > 11990:
            frequencies, times, Sxx = signal.spectrogram(tr.data, fs, nperseg=nperseg, noverlap=noverlap)
            temp_array.append(Sxx)
        else:
            continue
    if len(temp_array) == 3:
        compoenet_array = np.stack(temp_array, axis=-1)
        epsilon = 1e-40  # Small constant to avoid log(0)
        log_array = np.log10(compoenet_array + epsilon)
        log_array_min = log_array.min()
        log_array_max = log_array.max()
        scaled_array = (log_array - log_array_min) / (log_array_max - log_array_min)
        temp_all.append(scaled_array)
        all_hour.append(t.hour)
    os.chdir('..')
df_earthquake2 = pd.read_csv('/media/justin/Data2/AGU_Discrime/blast_SNR_LSL.csv')
filtered_df2 = df_earthquake2[(df_earthquake2['SNR_denoise'] >= 1.8)]

for time, station in zip(filtered_df2['Event'], filtered_df2['Station']):
    os.chdir('/media/justin/Data2/AGU_Discrime/LSL_station_blast')
    t = UTCDateTime(time)
    print(t)
    event = t.strftime('%Y%m%d%H%M%S')
    os.chdir(event)
    file = '*' + station + '*'
    inv = read_inventory(file)
    os.chdir('/media/justin/Data2/AGU_Discrime/LSL_blast')
    os.chdir(event)
    st = read(file)
    for tr in st.select(station="M53A"):
        st.remove(tr)
    for tr in st.select(station="O53A"):
        st.remove(tr)
    for tr in st.select(station="P52A"):
        st.remove(tr)            
    for tr in st:
        if round(tr.stats.sampling_rate) == 100:
            pre_filt = [0.001, 0.002, 45, 50]
        if round(tr.stats.sampling_rate) == 40:
            pre_filt = [0.001, 0.002, 18, 20]
        tr.remove_response(inventory=inv,pre_filt=pre_filt,output="DISP",water_level=60,taper=True,taper_fraction=0.00001)
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("highpass", freq=1)
    st2 = model.annotate(st)
    temp_array = []
    for tr in st2:
        if tr.stats.npts > 11990:
            frequencies, times, Sxx = signal.spectrogram(tr.data, fs, nperseg=nperseg, noverlap=noverlap)
            temp_array.append(Sxx)
        else:
            continue
    if len(temp_array) == 3:
        compoenet_array = np.stack(temp_array, axis=-1)
        epsilon = 1e-40  # Small constant to avoid log(0)
        log_array = np.log10(compoenet_array + epsilon)
        log_array_min = log_array.min()
        log_array_max = log_array.max()
        scaled_array = (log_array - log_array_min) / (log_array_max - log_array_min)
        temp_all.append(scaled_array)
        all_hour.append(t.hour)
    os.chdir('..')

all_array = np.stack(temp_all, axis=-1)
aaa = np.array(all_hour)     

# Convert to a PyTorch tensor and reshape
reshaped_tensor = torch.tensor(all_array).permute(3, 2, 0, 1)

# Save as a tensor file
os.chdir('/media/justin/Data2/AGU_Discrime')
torch.save(reshaped_tensor, 'blast_tensor_SNR18.pt')
np.save('blast_tensor_SNR18_hour.npy', aaa)
