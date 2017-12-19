import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.close('all')


batch_nmse_train_log = np.loadtxt('batch_nmse_train_log.txt')

x = np.array([[i+1 for i in range(batch_nmse_train_log.shape[0])]]).T
y1 = batch_nmse_train_log[:,0]
y2 = batch_nmse_train_log[:,1]
y3 = batch_nmse_train_log[:,2]

N_filter_order       = 2
low_pass_cutoff_freq = 7.7
fs                   = 1.0;
b, a = signal.butter(N_filter_order, low_pass_cutoff_freq/(fs/2))
pp_y1 = signal.filtfilt(b, a, y1, padlen=150)
pp_y2 = signal.filtfilt(b, a, y2, padlen=150)
pp_y3 = signal.filtfilt(b, a, y3, padlen=150)

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(x, pp_y1)
ax1.set_title('Sharing both axes')
ax2.plot(x, pp_y2)
ax3.plot(x, pp_y3)
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)