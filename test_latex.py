import matplotlib.pyplot as plt
import numpy as np
import os

# Generate some example data
n_valid_hits_per_shower = np.linspace(1, 10, 100)
incident_e_per_shower = np.linspace(1, 100, 100)
x_bin, y_bin = np.meshgrid(n_valid_hits_per_shower, incident_e_per_shower)
e_vs_nhits_prob = np.random.rand(100, 100)

# Create the figure and axis
fig0, ax0 = plt.subplots(ncols=1, sharey=True)

# Generate the heatmap
heatmap = ax0.pcolormesh(y_bin, x_bin, e_vs_nhits_prob, cmap='rainbow')

# Plot a diagonal line
ax0.plot(n_valid_hits_per_shower, n_valid_hits_per_shower, 'k-')

# Set limits
ax0.set_xlim(n_valid_hits_per_shower.min(), n_valid_hits_per_shower.max())
ax0.set_ylim(incident_e_per_shower.min(), incident_e_per_shower.max())

# Add colorbar
cbar = plt.colorbar(heatmap)

# Add grid
ax0.grid()

# Define the filename for saving the figure
savefigname = 'validhitsine2D_test.png'

# Save the figure
try:
    fig0.savefig(savefigname)
    print(f"Figure saved successfully as {savefigname}")
except Exception as e:
    print(f"Error saving figure: {e}")

# Close the figure to free memory
plt.close(fig0)
#import matplotlib.pyplot as plt

#import os
#
#
#
#plt.rcdefaults()
#plt.rcParams['text.usetex'] = False
#plt.rcParams['font.family'] = 'serif'  # Optional: Change to a LaTeX-compatible font
#plt.rcParams['mathtext.fontset'] = 'cm'
#
## Set up a minimal plot
#plt.figure()
#
## Example LaTeX-formatted labels
#plt.title(r"Sample Title with LaTeX: $\alpha + \beta = \gamma$")
#plt.xlabel(r"X-axis: $x^{2}$")
#plt.ylabel(r"Y-axis: $\sqrt{y}$")
#
## Plot some data
#plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
#
## Save the plot to a file
#output = '/eos/home-c/chenhua/copy_tdsm_encoder_sweep16'
#save_name = os.path.join(output,'hit2D')
#plt.savefig(save_name)
#
## Optional: Print a success message
#print("LaTeX plot generated and saved as latex_test_output.png")
#
