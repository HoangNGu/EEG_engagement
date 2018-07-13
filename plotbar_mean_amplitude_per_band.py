import numpy as np
import csv

fs = 250                                # Sampling rate (512 Hz)
csvEEG = "..\Dataset\Zeynep_EEG_Experience\Person0_2018-02-23_eeg.csv"

def OpenCsvFile (csvfile):
    
   f=open(csvfile,'r') # opens file for reading
   reader = csv.reader(f, delimiter=',')
   timestamp = []
   channel = [[] for j in range(4)]
   for row in reader:
      timestamp.append(float(row[0]))
      for i in range(4):
          channel[i].append(float(row[i+1]))
   f.close()
   return (timestamp,channel[0])

timestamp, data = OpenCsvFile (csvEEG)
# Get real amplitudes of FFT (only in postive frequencies)
fft_vals = np.absolute(np.fft.rfft(data))

# Get frequencies for amplitudes in Hz
fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)

# Define EEG bands
eeg_bands = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

# Take the mean of the fft amplitude for each EEG band
eeg_band_fft = dict()
for band in eeg_bands:  
    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                       (fft_freq <= eeg_bands[band][1]))[0]
    eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

# Plot the data (using pandas here cause it's easy)
import pandas as pd
df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")