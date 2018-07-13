import csv
import time
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()
#===========================================================
#======================== FUNCTIONS ========================
#===========================================================
def OpenCsvFileSpecial (csvfile):
    
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

def OpenCsvFile (csvfile):
    
   f=open(csvfile,'r') # opens file for reading
   reader = csv.reader(f, delimiter=',')
   timestamp = []
   channel = [[] for j in range(4)]
   for row in reader:
      timestamp.append(float(row[0]))
      for i in range(4):
          channel[i].append(float(row[i+2]))
   f.close()
   return (timestamp,channel)

def OpenCsvFile_offset (csvfile):
    
   f=open(csvfile,'r') # opens file for reading
   reader = csv.reader(f, delimiter=',')
   info = []
   for row in reader:
      info.append(row)
   f.close()
   return (info)

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
    
infoCsv = "..\Dataset\\2018_02_eeg_attention\\2018_02_eeg_attention\offsets.csv"
info = []
info_experiment = []
info = OpenCsvFile_offset(infoCsv)
fs_Hz = 250 

timewindowStarttime = [5, 10, 15, 25, 30, 40, 50, 60, 70, 80, 85, 90, 100, 105, 110]

engagement_timewindow = [[[] for k in range(len(timewindowStarttime))] for i in range(len(info))]
index = [[[] for k in range(len(timewindowStarttime))] for i in range(len(info))]

for p in range(len(info)):
#for p in range(1): # to test
    info_file = info[p]
    csvfile = "..\Dataset\\2018_02_eeg_attention\\2018_02_eeg_attention\Person0_"+info_file[0][:10].replace("_", "-")+"_1_eeg"+info_file[0][10:]+".csv"
    info_experiment.append(info_file[0])
    startTime = int(info_file[1])
    videoBegin = info_file[2]
    if len(videoBegin) == 5:
        videostart_sec = int(videoBegin[:1])*60 + int(videoBegin[2:4])
    else:
        videostart_sec = int(videoBegin[:2])*60 + int(videoBegin[3:5])
    
    if(info_file[0] == "2018_02_23_gokhan"):
        timestamp,channel = OpenCsvFileSpecial(csvfile)
    else:
        timestamp,channel = OpenCsvFile(csvfile)
    
    for s in range(len(timewindowStarttime)):
        channel_temp = list(channel)
#    for s in range(1):
        timewindowBegin = startTime - videostart_sec + timewindowStarttime[s]*60
        timewindowEnd = startTime - videostart_sec + timewindowStarttime[s]*60 +10
        current_index = 0
        find = False
        while(int(timestamp[current_index]) != timewindowEnd and current_index < len(timestamp) -1 ):
            if( find == False and int(timestamp[current_index]) == timewindowBegin):
                find = True
                index_start = current_index
            current_index+=1
        index_stop = current_index
        index[p][s].append(index_start)
        index[p][s].append(index_stop)
        
        for m in range(4):
            channel_temp[m] = channel_temp[m][index_start:index_stop] # Crop original data inside time window
        
        #Apply FFT algorithm
        n = len(channel_temp[0]) # length of the signal (same for all channel)
        n_window = 132 # 132 = 500 ms
        
        #Filter 667-683 - on all channel
        ##NOTCH,BANDBASS
        channel_filter = [[] for j in range(4)]
        #Define b and a for notch, bandpass
        notch_b, notch_a,bandpass_b, bandpass_a = filterCreation(fs_Hz)
        for i in range(len(channel_temp)):
            channel_filter[i] = signal.lfilter(notch_b,notch_a,channel_temp[i])
            channel_filter[i] = signal.lfilter(bandpass_b,bandpass_a,channel_temp[i])
            
        channel_cut = [[] for j in range(4)]
        for i in range(len(channel_temp)):
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
        
        # Engagement global (all channel)
        EngagementGlobal = []
        for i in range(len(EEG_total_head_power[0])):
            E = EEG_total_head_power[3][i]/(EEG_total_head_power[1][i] + EEG_total_head_power[2][i])
            EngagementGlobal.append(E)
        engagement_timewindow[p][s] = (EngagementGlobal)
    print("File n°"+str(p+1)+"/"+str(len(info)+1)+" processed") 
    
print("--- %s seconds ---" % (time.time() - start_time))   
#Assuming res is a list of lists
outputcsvfile = "..\Dataset\\2018_02_eeg_attention\\2018_02_eeg_attention\outputfile.csv"
with open(outputcsvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(engagement_timewindow)
    
    engagement_mean = [[] for i in range(len(engagement_timewindow))]
    for i in range(len(engagement_timewindow)):
        for l in range(len(engagement_timewindow[i])):
            engagement_mean[i].append(np.mean(engagement_timewindow[i][l]))
            
    
for i in range(len(engagement_mean)):   
    x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    plt.figure()
    plt.title('Engagement of '+ info_experiment[i])
    plt.plot(x,engagement_mean[i], 'r')
    plt.xticks(x, timewindowStarttime, rotation='vertical')
    plt.grid()
    plt.show()
    
exp = {}
for i in range(len(info)):
    name_temp = info[i][0][11:]
    if name_temp in exp:
        exp[name_temp].append(i)
    else:
        exp[name_temp]=[i]

engagement_experiment = [[] for i in range(3)]



for name in exp:
    for i in range(len(exp[name])):
        engagement_experiment[i].append(engagement_mean[exp[name][i]])  

engagement_experiment_plot = [[] for i in range(3)]
std_engagement_experiment_plot = [[] for i in range(4)]


for i in range(len(engagement_experiment)):
    for j in range(1):
        list_temp = []
        for k in range(len(engagement_experiment[i])):           
            list_temp.append(engagement_experiment[i][k][j])
        engagement_experiment_plot[i].append(np.mean(list_temp))
        std_engagement_experiment_plot[i].append(np.std(list_temp))
        
engagement_experiment_plot_time_bin = [[] for i in range(3)]
std_engagement_experiment_plot_time_bin = [[] for i in range(4)]

for i in range(len(engagement_experiment_plot)):
    for j in my_range(0, len(engagement_experiment_plot[i])-3, 3):
        list_temp = []
        for k in range(3):
            list_temp.append(engagement_experiment_plot[i][j+k])
        engagement_experiment_plot_time_bin[i].append(np.mean(list_temp))
        std_engagement_experiment_plot_time_bin[i].append(np.std(list_temp))

engagement_experiment_plot_mean = []
engagement_experiment_plot_time_bin_mean = []


for i in range(len(engagement_experiment_plot[0])):
    list_temp = []
    for j in range(len(engagement_experiment_plot)):
        list_temp.append(engagement_experiment_plot[j][i])
    engagement_experiment_plot_mean.append(np.mean(list_temp))
    std_engagement_experiment_plot[3].append(np.std(list_temp))
    
for i in range(len(engagement_experiment_plot_time_bin[0])):
    list_temp = []
    for j in range(len(engagement_experiment_plot_time_bin)):
        list_temp.append(engagement_experiment_plot_time_bin[j][i])
    engagement_experiment_plot_time_bin_mean.append(np.mean(list_temp))
    std_engagement_experiment_plot_time_bin[3].append(np.std(list_temp))




#plt.figure()
colors =['r','b','g']
x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
for i in range(3):
    fig = plt.figure()
    plt.title('Engagement for each experiment for each time window ' + 'Exp '+str(i+1))
    plt.errorbar(x, engagement_experiment_plot[i], yerr=std_engagement_experiment_plot[i],color = colors[i], fmt = 'o--', capsize=5)
    plt.xticks(x, timewindowStarttime, rotation='vertical')
fig = plt.figure()
plt.title('Engagement for each experiment for each time window: Mean')
plt.errorbar(x, engagement_experiment_plot_mean, yerr=std_engagement_experiment_plot[3],color = 'c' , label='Mean',fmt = 'o--', capsize=5)

plt.grid()
plt.legend()
plt.show()


x_timebin = ['t1','t2','t3','t4','t5']
x_bin = [0,1,2,3,4]
colors =['r','b','g']

for i in range(3):
    plt.figure()
    plt.title('Engagement for each experiment for each time bin '+ 'Exp '+str(i+1))
    plt.errorbar(x_bin, engagement_experiment_plot_time_bin[i], yerr=std_engagement_experiment_plot_time_bin[i],color =colors[i], fmt = 'o--', capsize=5)
    plt.xticks(x_bin, x_timebin)
plt.figure()
plt.title('Engagement for each experiment for each time bin: Mean ')
plt.errorbar(x_bin, engagement_experiment_plot_time_bin_mean, yerr=std_engagement_experiment_plot_time_bin[3],color = 'c', fmt = 'o--', capsize=5)
plt.xticks(x_bin, x_timebin)
plt.legend()
plt.grid()
plt.show()