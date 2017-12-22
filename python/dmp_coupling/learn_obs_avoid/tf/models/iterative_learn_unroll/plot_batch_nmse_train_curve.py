import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

plt.close('all')


batch_nmse_train_log = np.loadtxt('batch_nmse_train_log.txt')
N_iters_plot = 10000

x = np.array([[i for i in range(N_iters_plot)]]).T
y1 = batch_nmse_train_log[0:N_iters_plot,0]
y2 = batch_nmse_train_log[0:N_iters_plot,1]
y3 = batch_nmse_train_log[0:N_iters_plot,2]

N_filter_order = 7
Wn = 0.1
b, a = signal.butter(N_filter_order, Wn)
pp_y1 = signal.filtfilt(b, a, y1)
pp_y2 = signal.filtfilt(b, a, y2)
pp_y3 = signal.filtfilt(b, a, y3)

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.plot(x, pp_y1)
ax1.set_title('Batch NMSE Training Curve vs Iterations')
ax2.plot(x, pp_y2)
ax3.plot(x, pp_y3)

ax1.set_ylabel('NMSE: Cx (Coupling Term x-axis)')
ax2.set_ylabel('NMSE: Cy (Coupling Term y-axis)')
ax3.set_ylabel('NMSE: Cz (Coupling Term z-axis)')
ax3.set_xlabel('Iterations')
# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)