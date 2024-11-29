from obspy import read, read_inventory, UTCDateTime
import seisbench.models as sbm
import os
import pandas as pd
import math
import numpy as np
from obspy.core.stream import Stream
import glob
def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))

    # Radius of Earth in kilometers. Use 3956 for miles
    r = 6371

    # Calculate the result
    return c * r

df_earthquake = pd.read_csv('/media/justin/Data2/AGU_Discrime/all_blast.csv')
df_earthquake['Time'] = pd.to_datetime(df_earthquake['Time'], errors='coerce')
# Step 3b: Reformat 'Time' column
df_earthquake['Time'] = df_earthquake['Time'].dt.strftime('%Y-%m-%dT%H:%M:%S.000Z')
model = sbm.DeepDenoiser.from_pretrained("original")
os.chdir('/media/justin/Data2/AGU_Discrime/WQSZ_station_blast')
all_events = sorted(os.listdir())
df_station = []
df_distance=[]
df_event = []
df_SNR = []
df_SNR_denoise = []
ddof = 1
for event in all_events:
    print(event)
    os.chdir('/media/justin/Data2/AGU_Discrime/WQSZ_station_blast')
    os.chdir(event)
    files = glob.glob('*')
    if not files:  # If the list is empty, no files match the pattern
        print("No files found. Continuing to next operation.")
    else:
        inv = read_inventory('*')
        os.chdir('/media/justin/Data2/AGU_Discrime/WQSZ_blast')
        os.chdir(event)
        t = UTCDateTime(event)
        ttt = t.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        row = df_earthquake[df_earthquake['Time'] == ttt]
        row_values = row.iloc[0].to_dict()
        env_lat = row_values['Latitude']
        env_lon = row_values['Longitude']
        env_mag = row_values['Magnitude']
        st_begin = read('*')
        stations = []
        traces=[]
        for tr in st_begin:
            if round(tr.stats.sampling_rate) == 100 and tr.stats.npts > 10000:
                traces.append(tr)
            elif round(tr.stats.sampling_rate) == 40 and tr.stats.npts > 4000:
                traces.append(tr)
            else:
                continue


        st = Stream(traces=traces)
        for tr in st:
            if tr.stats.station == 'M53A':
                continue
            else:
                if round(tr.stats.sampling_rate) == 100:
                    pre_filt = [0.001, 0.002, 45, 50]
                if round(tr.stats.sampling_rate) == 40:
                    pre_filt = [0.001, 0.002, 18, 20]
                tr.remove_response(inventory=inv,pre_filt=pre_filt,output="DISP",water_level=60,taper=True,taper_fraction=0.00001)
                stations.append(tr.stats.station)
        st.detrend("linear")
        st.taper(max_percentage=0.05, type="hann")
        st.filter("highpass", freq=1.0)
        annotations = model.annotate(st)
        stations_all = list(set(stations))


        for sta in stations_all:
            ##print(sta)
            st2 = st.select(station=sta)
            if len(st2) == 3:
                tr = st2[0]
                location = inv.get_coordinates(tr.stats.network+'.' + tr.stats.station + '.' + tr.stats.location + '.' + tr.stats.channel, t)
                sta_lat = location['latitude']
                sta_lon = location['longitude']
                distance = haversine(sta_lat, sta_lon, env_lat, env_lon)
                t1  = distance/6.5 - 1 + 30
                t2 = distance/3.7 + 5 + 30
                temp_snr = []
                if round(tr.stats.sampling_rate) == 100:
                    fs = 100
                if round(tr.stats.sampling_rate) == 40:
                    fs = 40
                for tr_cal in st2:
                    data = tr_cal.data
                    # Define signal window in seconds
                    # Convert signal window to samples
                    start_sample = round(t1 * fs)
                    end_sample = round(t2 * fs)

                    # Extract signal and noise
                    signal = data[start_sample:end_sample]
                    # Assuming noise is the rest of the data; adjust as necessary
                    noise = np.concatenate([data[:start_sample], data[end_sample:]])
                    if data.size > 0:  # Ensuring the array isn't empty
                        if data.size > ddof:
                            # Ensuring there are enough data points for the chosen ddof
                            power_signal = np.var(signal, ddof=1)
                            power_noise = np.var(noise, ddof=1)
                        else:
                            print("Not enough data points for the chosen degrees of freedom.")

                    # Compute SNR
                    if power_signal == 0 or power_noise ==0:
                        snr = 0.00000001
                    else:
                        snr = power_signal / power_noise

                    # Convert SNR to decibels
                    snr_db_all = 10 * np.log10(snr)
                    temp_snr.append(snr_db_all)
                snr_db = sum(temp_snr)/3
                #df_station.append(tr.stats.station)
                df_SNR.append(snr_db)
                df_distance.append(distance)
                df_station.append(sta)
                df_event.append(ttt)

            st2 = annotations.select(station=sta)
            if len(st2) == 3:
                temp_snr = []
                fs = 100
                for tr_cal in st2:
                    data = tr_cal.data
                    # Define signal window in seconds
                    # Convert signal window to samples
                    start_sample = round(t1 * fs)
                    end_sample = round(t2 * fs)

                    # Extract signal and noise
                    signal = data[start_sample:end_sample]
                    # Assuming noise is the rest of the data; adjust as necessary
                    noise = np.concatenate([data[:start_sample], data[end_sample:]])
                    if data.size > 0:  # Ensuring the array isn't empty
                        if data.size > ddof:
                            # Ensuring there are enough data points for the chosen ddof
                            power_signal = np.var(signal, ddof=1)
                            power_noise = np.var(noise, ddof=1)
                        else:
                            print("Not enough data points for the chosen degrees of freedom.")

                    # Compute SNR
                    if power_signal == 0 or power_noise ==0:
                        snr = 0.00000001
                    else:
                        snr = power_signal / power_noise


                    # Convert SNR to decibels
                    snr_db_all = 10 * np.log10(snr)
                    temp_snr.append(snr_db_all)
                snr_db = sum(temp_snr)/3
                #df_station.append(tr.stats.station)
                df_SNR_denoise.append(snr_db)
df = pd.DataFrame({'Event': df_event,'Station':df_station, 'Distance': df_distance, 'SNR':df_SNR, 'SNR_denoise':df_SNR_denoise})
os.chdir('/media/justin/Data2/AGU_Discrime')
df.to_csv('blast_SNR_WQSZ.csv', index=False)