import numpy as np
import csv
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib import animation
#===========================================================
#======================== FUNCTIONS ========================
#===========================================================
def get_err_bar_xy(x, y, x0, xf, x_binsize, illustrate = False):
    x_edges = np.arange(x0 , xf, x_binsize)
    pos = np.digitize(x, x_edges) 
    y_mean = []
    y_std = []
    for i in range(len(x_edges)):
        vals = np.array(y)[(np.where(pos == i))]
        if not len(vals):
            y_mean.append(0)
            y_std.append(0)
        else:
            y_mean.append(np.mean(vals))
            y_std.append(np.std(vals))
    if illustrate:
        plt.figure("error_bars")
#        plt.plot(x, y, 'b.')
        plt.errorbar(x_edges ,  y_mean, yerr = y_std, fmt="ro--", ecolor = 'r')
        # if you need to play with any of the following, feel free            
        #        axes = plt.gca()
        #        axes.set_xlim([0,10000])       
        #        axes.set_ylim([0,6000])       
        plt.xlabel('time (sec)')
        plt.ylabel('Engagement index (E)')
        plt.grid(True)
        plt.show()

#==============================================
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
    
#csvEEG = "..\Dataset\\2018_02_eeg_attention\\2018_02_eeg_attention\Person0_2018-02-13_1_eeg_esra.csv" #participant a, exp 1
#csvEEG = "..\Dataset\\2018_02_eeg_attention\\2018_02_eeg_attention\Person0_2018-02-15_1_eeg_buse.csv" #participant b, exp 2
csvEEG = "..\Dataset\\2018_02_eeg_attention\\2018_02_eeg_attention\Person0_2018-02-28_1_eeg_gokhan.csv" #participant c, exp 3
fs_Hz = 250                                # Sampling rate (512 Hz)
timestamp, channel = OpenCsvFile(csvEEG)
n = len(channel[0]) # length of the signal (same for all channel)
n_window = 132 # 132 = 500 ms

#Filter 667-683 - on all channel
##NOTCH,BANDBASS
channel_filter = [[] for j in range(4)]
#Define b and a for notch, bandpass
notch_b, notch_a,bandpass_b, bandpass_a = filterCreation(fs_Hz)
for i in range(len(channel)):
    channel_filter[i] = signal.lfilter(notch_b,notch_a,channel[i])
    channel_filter[i] = signal.lfilter(bandpass_b,bandpass_a,channel[i])
    
channel_cut = [[] for j in range(4)]
for i in range(len(channel)):
    for j in my_range(0,n-n_window,n_window):
        channel_cut[i].append(channel_filter[i][j:j+n_window]) # fft computing and normalization, 132 points = 500ms

# Get real amplitudes of FFT (only in postive frequencies)
fft_vals = [[] for j in range(4)]
fft_freq = [[] for j in range(4)]
for i in range(len(channel_cut)):
    for k in range(len(channel_cut[0])):        
        fft_vals[i].append(np.absolute(np.fft.rfft(channel_cut[i][k])))

        # Get frequencies for amplitudes in Hz
        fft_freq[i].append(np.fft.rfftfreq(len(channel_cut[i][k]), 1.0/fs_Hz))

# Define EEG bands
eeg_bands = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

EEG_bands = [[] for j in range(4)]
# Get indices for frequency band
for i in range(4): 
    for band in eeg_bands:
        for j in range(len(fft_freq[i])):
            freq_ix = np.where((fft_freq[i][j] >= eeg_bands[band][0]) & 
                               (fft_freq[i][j] <= eeg_bands[band][1]))
            
            if(j==0):
                EEG_bands[i].append(freq_ix)
            elif (not np.array_equal(freq_ix,EEG_bands[i][-1])):
                EEG_bands[i].append(freq_ix)

EEG_bands = list(EEG_bands[0])
for i in range(5):
    EEG_bands[i] = np.ndarray.tolist(EEG_bands[i][0])
    
EEG_power = [[[] for j in range(5)] for j in range(4)]
for i in range(4):
    for j in range(len(fft_vals[i])):
        for l in range(5):
            mean = 0.0
            total = 0.0
            for k in range(len(EEG_bands[l])):
                total += fft_vals[i][j][EEG_bands[l][k]]
            mean = total/len(EEG_bands[l])
            EEG_power[i][l].append(mean)

# Compute total head power ( mean(channel_Power))
EEG_total_head_power = [[] for i in range(5)]      
for k in range(len(EEG_power[i][0])):
    for j in range(5):
        total = 0
        mean = 0
        for i in range(4):
            total += EEG_power[i][j][k]
        mean = total/4 #4 channels
        EEG_total_head_power[j].append(mean)
# Get engagement
# E = Beta / (Alpha + Theta)
# Engagement per channel
EngagementPerChannel = [[] for j in range(4)]
for i in range(4):
    for j in range(len(EEG_power[i][0])):
        E = EEG_power[i][3][j] / (EEG_power[i][1][j] + EEG_power[i][2][j])
        EngagementPerChannel[i].append(E)

# Engagement global (all channel)
EngagementGlobal = []
for i in range(len(EEG_total_head_power[0])):
    E = EEG_total_head_power[3][i]/(EEG_total_head_power[1][i] + EEG_total_head_power[2][i])
    EngagementGlobal.append(E)

plt.figure()
#===========================================================
#=========================== PLOT ==========================
#===========================================================   
#fig, ax = plt.subplots(5, 1)
#for i in range(4):
#    ax[i].plot(EngagementPerChannel[i],'r') # plotting the engagement per channel
#    ax[i].set_xlabel('Index')
#    ax[i].set_ylabel('Eng. chan.' + str(i))
#ax[4].plot(EngagementGlobal,'b') # plotting the global engagement
#ax[4].set_xlabel('Index')
#ax[4].set_ylabel('Global Eng.')

#######################################################################
#x = [i+1 for i in range(18200)]
#get_err_bar_xy(x, EngagementGlobal, 1000, 11800, 600, illustrate = True)
#

# plot with various axes scales
fig = plt.figure(1)
ax = fig.add_subplot(221)
ax1 = fig.add_subplot(222)
ax2 = fig.add_subplot(223)
ax3 = fig.add_subplot(224)
ax11 = fig.add_subplot(111)

ax.set_xlabel('Time (s)',fontsize=25)
ax.set_ylabel('EEG amplitude (μV)',fontsize=25)
ax2.set_xlabel('Time (s)',fontsize=25)
ax2.set_ylabel('EEG amplitude (μV)',fontsize=25)

ax1.set_xlabel('Frequency (Hz)',fontsize=25)
ax1.set_ylabel('Relative Amplitude',fontsize=25)
ax3.set_xlabel('Frequency (Hz)',fontsize=25)
ax3.set_ylabel('Relative Amplitude',fontsize=25)

# EEG1
plt.subplot(221)
plt.plot(channel_cut[0][69])
plt.xlim(xmin=0 ,xmax=132) 
plt.title('(a)',fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylabel('EEG amplitude (μV)',fontsize=20)
plt.grid(True)

# FFT1
plt.subplot(222)
plt.plot(fft_freq[0][69],fft_vals[0][69])
plt.title('(b)',fontsize=20)
plt.xlabel('Frequency (Hz)',fontsize=20)
plt.ylabel('Relative Amplitude',fontsize=20)
xcoords = [4, 8, 12, 30, 45]
plt.xticks(xcoords, ('4', '8', '12', '30', '45'), fontsize=20)
for xc in xcoords:
    plt.axvline(x=xc, ls='--', c = 'r')
plt.xlim(xmin=0 ,xmax=50) 

plt.grid(True)

# EEG2
plt.subplot(223)
plt.xlim(xmin=0 ,xmax=132) 
plt.plot(channel_cut[2][69])
plt.title('(c)',fontsize=20)
plt.xlabel('Time (s)',fontsize=20)
plt.ylabel('EEG amplitude (μV)',fontsize=20)
plt.grid(True)


# FFT2
plt.subplot(224)
plt.plot(fft_freq[1][69],fft_vals[2][69])
xcoords = [4, 8, 12, 30, 45]
plt.xticks(xcoords, ('4', '8', '12', '30', '45'), fontsize=20)
for xc in xcoords:
    plt.axvline(x=xc, ls='--', c = 'r')
plt.xlim(xmin=0 ,xmax=50) 
plt.title('(d)',fontsize=20)
plt.xlabel('Frequency (Hz)',fontsize=20)
plt.ylabel('Relative Amplitude',fontsize=20)
plt.grid(True)

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)
plt.show()