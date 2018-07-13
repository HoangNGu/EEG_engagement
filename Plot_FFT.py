import csv
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

csvEEG = "..\Dataset\Zeynep_EEG_Experience\Person0_2018-02-23_eeg.csv"

# Define EEG bands
eeg_bands = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

timestamp, channel = [],[]

csvfile = csvEEG
fs_Hz = 250 # sampling rate
Ts = 1.0/fs_Hz # sampling interval
t = np.arange(0,1,Ts) # time vector
#===========================================================
#======================== FUNCTIONS ========================
#===========================================================

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
   return (timestamp,channel)

def filterCreation(fs_Hz):
#    Notch filter 60Hz, Bandpassfilter:5-50Hz
    if fs_Hz == 125:
        notch_b = [0.931378858122982, 3.70081291785747, 5.53903191270520,
                   3.70081291785747, 0.931378858122982]
        notch_a = [1, 3.83246204081167, 5.53431749515949,
                   3.56916379490328, 0.867472133791669]
        bandpass_b = [0.529967227069348, 0, -1.05993445413870, 0, 0.529967227069348]
        bandpass_a = [1, -0.517003774490767, -0.734318454224823, 0.103843398397761, 0.294636527587914]
    elif fs_Hz == 200:
        notch_b = [0.956543225556877, 1.18293615779028, 2.27881429174348, 1.18293615779028, 0.956543225556877]
        notch_a = [1, 1.20922304075909, 2.27692490805580, 1.15664927482146, 0.914975834801436]
        bandpass_b = [0.248341078962541, 0, -0.496682157925081, 0, 0.248341078962541]
        bandpass_a = [1, -1.86549482213123, 1.17757811892770, -0.460665534278457, 0.177578118927698]
    elif fs_Hz == 250:
        notch_b = [0.965080986344733, -0.242468320175764, 1.94539149412878, -0.242468320175764, 0.965080986344733]
        notch_a = [1, -0.246778261129785, 1.94417178469135, -0.238158379221743, 0.931381682126902]
        bandpass_b = [0.175087643672101, 0, -0.350175287344202, 0, 0.175087643672101]
        bandpass_a = [1, -2.29905535603850, 1.96749775998445, -0.874805556449481, 0.219653983913695]
    elif fs_Hz == 500:
        notch_b = [0.982385438526095, -2.86473884662109, 4.05324051877773, -2.86473884662109, 0.982385438526095]
        notch_a = [1, -2.89019558531207, 4.05293022193077, -2.83928210793009, 0.965081173899134]
        bandpass_b = [0.0564484622607352, 0, -0.112896924521470, 0, 0.0564484622607352]
        bandpass_a = [1, -3.15946330211917, 3.79268442285094, -2.08257331718360, 0.450445430056042]
    elif fs_Hz == 1000:
        notch_b = [0.991153595101611, -3.68627799048791, 5.40978944177152, -3.68627799048791, 0.991153595101611]
        notch_a = [1, -3.70265590760266, 5.40971118136100, -3.66990007337352, 0.982385450614122]
        bandpass_b = [0.0165819316692804, 0, -0.0331638633385608, 0, 0.0165819316692804]
        bandpass_a = [1, -3.58623980811691, 4.84628980428803, -2.93042721682014, 0.670457905953175]
    elif fs_Hz == 1600:
        notch_b = [0.994461788958027, -3.86796874670208, 5.75004904085114, -3.86796874670208, 0.994461788958027]
        notch_a = [1, -3.87870938463296, 5.75001836883538, -3.85722810877252, 0.988954249933128]
        bandpass_b = [0.00692579317243661, 0, -0.0138515863448732, 0, 0.00692579317243661]
        bandpass_a = [1, -3.74392328264678, 5.26758817627966, -3.30252568902969, 0.778873972655117]
    else:
        print("Error: only 125Hz, 200Hz, 250Hz, 500Hz, 1000Hz or 1600Hz")
        notch_b = [1.0]
        notch_a = [1.0]
        bandpass_b = [1.0]
        bandpass_a = [1.0]
    return(notch_b,notch_a,bandpass_b,bandpass_a)
    
def my_range(start, end, step):
    while start <= end:
        yield start
        start += step
#===========================================================
#=========================== CODE ==========================
#===========================================================

timestamp, channel = OpenCsvFile(csvfile)
n = len(channel[0]) # length of the signal (same for all channel)
k = np.arange(n)
T = n/fs_Hz

frq = k/T # two sides frequency range
frq = frq[range(n//2)] # one side frequency range

channel_filter = [[] for j in range(4)]
channel_FFT = [[] for j in range(4)]
#Define b and a for notch, bandpass
notch_b, notch_a,bandpass_b, bandpass_a = filterCreation(fs_Hz)

#Filter 667-683 - on all channel
##NOTCH,BANDBASS
for i in range(len(channel)):
    channel_filter[i] = signal.lfilter(notch_b,notch_a,channel[i])
    channel_filter[i] = signal.lfilter(bandpass_b,bandpass_a,channel[i])
    
#FFT 686-749 - on all channel
#            channel_FFT[i] = np.fft.fft(channel_filter[i:i+132])/132 # fft computing and normalization
for i in range(len(channel)):
    channel_FFT[i] = np.fft.fft(channel_filter[i])/n # fft computing and normalization
    channel_FFT[i] = channel_FFT[i][range(n//2)]
    
fig, ax = plt.subplots(2, 1)
ax[0].plot(frq,abs(channel_FFT[0]),'r') # plotting the spectrum
ax[0].set_xlabel('Freq (Hz)')
ax[0].set_ylabel('|Y(freq)|')
ax[1].plot(frq,abs(channel_FFT[1]),'r') # plotting the spectrum
ax[1].set_xlabel('Freq (Hz)')
ax[1].set_ylabel('|Y(freq)|')
#Get headwide power 784
