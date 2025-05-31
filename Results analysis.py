import zipfile
import os
import pandas as pd
import re
import statistics as stat
import numpy as np
import matplotlib.pyplot as plt


def analyze_results_energy(results_path):
  """Analyzes energy consumption from results.

  Parameters
  ----------
    results_path (String): Path to the results directory. This directory must be structured with two folders: Emissions (containing energy consumption output files) and Output (containing other output files).

  Returns
  -------
    emissions_data_avg (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the average of total energy consumption across 5 repetitions and summed over all epochs.
    gpu_data_avg (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the average of gpu energy consumption across 5 repetitions and summed over all epochs.
    ram_data_avg (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the average of ram energy consumption across 5 repetitions and summed over all epochs.
    cpu_data_avg (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the average of cpu energy consumption across 5 repetitions and summed over all epochs.
    emissions_data_sd (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the standard deviation of total energy consumption summed over all epochs across 5 repetitions.
    gpu_data_sd (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the standard deviation of gpu energy consumption summed over all epochs across 5 repetitions.
    ram_data_sd (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the standard deviation of ram energy consumption summed over all epochs across 5 repetitions.
    cpu_data_sd (dictionary): Has one entry for each threshold, with a key that is the threshold ID (from 0 to 29). Each value is the standard deviation of cpu energy consumption summed over all epochs across 5 repetitions.
  """

  emissions_dir = os.path.join(results_path, "Emissions")

  emissions_data_avg = {}
  gpu_data_avg = {}
  ram_data_avg = {}
  cpu_data_avg = {}
  emissions_data_sd = {}
  gpu_data_sd = {}
  ram_data_sd = {}
  cpu_data_sd = {}
  for threshold_nb in range(0, 150, 5):
    total_energy_threshold = []
    energy_gpu_threshold = []
    energy_ram_threshold = []
    energy_cpu_threshold = []
    for experience in range (0,5):
      filename = "emissions_" + str(threshold_nb+experience) + ".csv"
      filepath = os.path.join(emissions_dir, filename)
      df = pd.read_csv(filepath)
      total_energy = df["energy_consumed"].sum()
      total_energy_threshold.append(total_energy)
      energy_cpu = df["cpu_energy"].sum()
      energy_cpu_threshold.append(energy_cpu)
      energy_gpu = df["gpu_energy"].sum()
      energy_gpu_threshold.append(energy_gpu)
      energy_ram = df["ram_energy"].sum()
      energy_ram_threshold.append(energy_ram)
    avg_total_energy = stat.mean(total_energy_threshold)
    avg_energy_gpu = stat.mean(energy_gpu_threshold)
    avg_energy_ram = stat.mean(energy_ram_threshold)
    avg_energy_cpu = stat.mean(energy_cpu_threshold)
    sd_total_energy = stat.stdev(total_energy_threshold)
    sd_energy_gpu = stat.stdev(energy_gpu_threshold)
    sd_energy_ram = stat.stdev(energy_ram_threshold)
    sd_energy_cpu = stat.stdev(energy_cpu_threshold)
    emissions_data_avg[threshold_nb/5] = avg_total_energy
    gpu_data_avg[threshold_nb/5] = avg_energy_gpu
    ram_data_avg[threshold_nb/5] = avg_energy_ram
    cpu_data_avg[threshold_nb/5] = avg_energy_cpu
    emissions_data_sd[threshold_nb/5] = sd_total_energy
    gpu_data_sd[threshold_nb/5] = sd_energy_gpu
    ram_data_sd[threshold_nb/5] = sd_energy_ram
    cpu_data_sd[threshold_nb/5] = sd_energy_cpu

  return emissions_data_avg, gpu_data_avg, ram_data_avg, cpu_data_avg, emissions_data_sd, gpu_data_sd, ram_data_sd, cpu_data_sd

def analyze_results(results_path):
  """Analyzes training and testing accuracies, switched off channels and avoided computations from results.

    Parameters
    ----------
    results_path (String): Path to the results directory. This directory must be structured with two folders: Emissions (containing energy consumption output files) and Output (containing other output files).

    Returns
    -------
    all_results (dictionary): Is a dictionary of dictionaries. One dictionary corresponds to one threshold and contains the average training and testing accuracies, the average total number of channels switched off at training and testing, the average number of channels switched off at training and testing per layer, the average total number of computations avoided at training and testing, the average number of avoided computations per layer at training and testing, the average maximum training and testing accuracies over all epochs, and the standard deviations of all these values. Averages and standard deviations are computed across the 5 iterations of each experiment.
  """

  output_dir = os.path.join(results_path, "Output")
  all_results = []

  for threshold_nb in range(0, 150, 5):
    results = {}
    training_acc = []
    test_acc = []
    channels_training = []
    channels_testing = []
    values_training = []
    values_testing = []
    layers_training_detail = []
    layers_testing_detail = []
    layers_values_training_detail = []
    layers_values_testing_detail = []
    max_training_acc = []
    max_testing_acc = []
    for experience in range (0,5):
      layers_training_exp = []
      layers_testing_exp = []
      layers_value_training_exp = []
      layers_value_testing_exp = []
      filename = "Output_" + str(int(threshold_nb+experience)) + ".txt"
      filepath = os.path.join(output_dir, filename)
      with open(filepath, 'r') as f:
        lines = f.readlines()

      first_line = lines[0]
      lines = lines[-26:]
      max_training_acc.append(float(lines[-2].split(":")[1]))
      max_testing_acc.append(float(lines[-1].split(":")[1]))
      score_training = 0
      score_testing = 0
      score_value_training = 0
      score_value_testing = 0
      match = re.match(r"Threshold: ([\d.]+)", first_line.strip())
      if match:
        threshold = float(match.group(1))
        results["threshold"] = threshold
      for i, line in enumerate(lines):
        if "Training accuracy" in line:
           training_acc.append(float(line.split(":")[1]))
        elif "Test accuracy" in line:
           test_acc.append(float(line.split(":")[1]))
        elif "Percentage of switched off channels for layer" in line:
          match = re.match(r"Percentage of switched off channels for layer (\d+) at (training|testing): ([\d.]+)%", line)
          if match:
            layer = int(match.group(1))
            phase = match.group(2)
            percentage = float(match.group(3))
            if 1 <= layer <= 2:
              multiplier = 16
              multiplier_values = 16384
            elif layer == 3:
              multiplier = 16
              multiplier_values = 8192
            elif 4 <= layer <= 5:
              multiplier = 32
              multiplier_values = 8192
            elif layer == 6:
              multiplier = 32
              multiplier_values = 4096
            elif 7 <= layer <= 8:
              multiplier = 64
              multiplier_values = 4096
            elif layer == 9:
              multiplier = 64
              multiplier_values = 1
            else:
              multiplier = 0
              multiplier_values = 0
            contribution = (percentage/100) * multiplier
            if phase == "training":
                score_training += contribution
                score_value_training += round(contribution) * multiplier_values
                layers_training_exp.append(contribution)
                layers_value_training_exp.append(round(contribution) * multiplier_values)
            elif phase == "testing":
                score_testing += contribution
                score_value_testing += round(contribution) * multiplier_values
                layers_testing_exp.append(contribution)
                layers_value_testing_exp.append(round(contribution) * multiplier_values)
      layers_training_detail.append(layers_training_exp)
      layers_testing_detail.append(layers_testing_exp)
      layers_values_training_detail.append(layers_value_training_exp)
      layers_values_testing_detail.append(layers_value_testing_exp)
      channels_training.append(score_training)
      channels_testing.append(score_testing)
      values_training.append(score_value_training)
      values_testing.append(score_value_testing)


    results["training_accuracy_avg"] = round(stat.mean(training_acc),4)
    results["test_accuracy_avg"] = round(stat.mean(test_acc),4)
    results["training_accuracy_sd"] = round(stat.stdev(training_acc),4)
    results["test_accuracy_sd"] = round(stat.stdev(test_acc),4)
    results["channels_off_training_avg"] = round(stat.mean(channels_training),4)
    results["channels_off_testing_avg"] = round(stat.mean(channels_testing),4)
    results["channels_off_training_sd"] = round(stat.stdev(channels_training),4)
    results["channels_off_testing_sd"] = round(stat.stdev(channels_testing),4)
    results["values_off_training_avg"] = round(stat.mean(values_training),4)
    results["values_off_testing_avg"] = round(stat.mean(values_testing),4)
    results["values_off_training_sd"] = round(stat.stdev(values_training),4)
    results["values_off_testing_sd"] = round(stat.stdev(values_testing),4)
    results["layers_training_detail_avg"] = np.round(np.mean(np.array(layers_training_detail), axis=0),4)
    results["layers_testing_detail_avg"] = np.round(np.mean(np.array(layers_testing_detail), axis=0),4)
    results["layers_training_detail_sd"] = np.round(np.std(np.array(layers_training_detail), axis=0),4)
    results["layers_testing_detail_sd"] = np.round(np.std(np.array(layers_testing_detail), axis=0),4)
    results["values_training_detail_avg"] = np.round(np.mean(np.array(layers_values_training_detail), axis=0),4)
    results["values_testing_detail_avg"] = np.round(np.mean(np.array(layers_values_testing_detail), axis=0),4)
    results["values_training_detail_sd"] = np.round(np.std(np.array(layers_values_training_detail), axis=0),4)
    results["values_testing_detail_sd"] = np.round(np.std(np.array(layers_values_testing_detail), axis=0),4)
    results["max_training_acc_avg"] = round(stat.mean(max_training_acc),4)
    results["max_testing_acc_avg"] = round(stat.mean(max_testing_acc),4)
    results["max_training_acc_sd"] = round(stat.stdev(max_training_acc),4)
    results["max_testing_acc_sd"] = round(stat.stdev(max_testing_acc),4)

    all_results.append(results)

  return all_results

#Unzipping the results folder. The name of the folder can be changed.
with zipfile.ZipFile("results_cifar10.zip", 'r') as zip_ref:
    zip_ref.extractall("results_cifar10")

initial_folder = "results_cifar10"
#Path is to be changed if different
os.chdir("/content/%s/%s"%(initial_folder, initial_folder))
results_path = os.getcwd()

#Analyzing energy consumption results
emissions_results = analyze_results_energy(results_path)

#Transforming the dictionaries into lists
energy_list_avg = []
cpu_list_avg = []
gpu_list_avg = []
ram_list_avg = []
energy_list_sd = []
cpu_list_sd = []
gpu_list_sd = []
ram_list_sd = []
for result in emissions_results[0].values():
    energy_list_avg.append(round(float(result),4))
for result in emissions_results[1].values():
    gpu_list_avg.append(round(float(result),4))
for result in emissions_results[2].values():
    ram_list_avg.append(round(float(result),4))
for result in emissions_results[3].values():
    cpu_list_avg.append(round(float(result),4))
for result in emissions_results[4].values():
    energy_list_sd.append(round(float(result),4))
for result in emissions_results[5].values():
    gpu_list_sd.append(round(float(result),4))
for result in emissions_results[6].values():
    ram_list_sd.append(round(float(result),4))
for result in emissions_results[7].values():
    cpu_list_sd.append(round(float(result),4))

#Printing all energy consumption results
print(energy_list_avg)
print(gpu_list_avg)
print(ram_list_avg)
print(cpu_list_avg)
print(energy_list_sd)
print(gpu_list_sd)
print(ram_list_sd)
print(cpu_list_sd)

#Analyzing other results
results = analyze_results(results_path)

#Transforming dictionaries into lists
switched_off_training_avg = [entry['channels_off_training_avg'] for entry in results]
switched_off_testing_avg = [entry['channels_off_testing_avg'] for entry in results]
switched_off_training_sd = [entry['channels_off_training_sd'] for entry in results]
switched_off_testing_sd = [entry['channels_off_testing_sd'] for entry in results]
switched_off_training_detail_avg = [entry['layers_training_detail_avg'].tolist() for entry in results]
switched_off_testing_detail_avg = [entry['layers_testing_detail_avg'].tolist() for entry in results]
switched_off_training_detail_sd = [entry['layers_training_detail_sd'].tolist() for entry in results]
switched_off_testing_detail_sd = [entry['layers_testing_detail_sd'].tolist() for entry in results]

values_training_avg = [entry['values_off_training_avg'] for entry in results]
values_testing_avg = [entry['values_off_testing_avg'] for entry in results]
values_training_sd = [entry['values_off_training_sd'] for entry in results]
values_testing_sd = [entry['values_off_testing_sd'] for entry in results]
values_training_detail_avg = [entry['values_training_detail_avg'].tolist() for entry in results]
values_testing_detail_avg = [entry['values_testing_detail_avg'].tolist() for entry in results]
values_training_detail_sd = [entry['values_training_detail_sd'].tolist() for entry in results]
values_testing_detail_sd = [entry['values_testing_detail_sd'].tolist() for entry in results]

training_accuracies_avg = [entry['training_accuracy_avg'] for entry in results]
training_accuracies_sd = [entry['training_accuracy_sd'] for entry in results]
test_accuracies_avg = [entry['test_accuracy_avg'] for entry in results]
test_accuracies_sd = [entry['test_accuracy_sd'] for entry in results]
threshold_list = [round(entry['threshold'],4) for entry in results]

max_training_acc_avg = [entry['max_training_acc_avg'] for entry in results]
max_testing_acc_avg = [entry['max_testing_acc_avg'] for entry in results]
max_training_acc_sd = [entry['max_training_acc_sd'] for entry in results]
max_testing_acc_sd = [entry['max_testing_acc_sd'] for entry in results]

#Printing other results
print(switched_off_training_avg)
print(switched_off_testing_avg)
print(switched_off_training_sd)
print(switched_off_testing_sd)
print(switched_off_training_detail_avg)
print(switched_off_testing_detail_avg)
print(switched_off_training_detail_sd)
print(switched_off_testing_detail_sd)


print(values_training_avg)
print(values_testing_avg)
print(values_training_sd)
print(values_testing_sd)
print(values_training_detail_avg)
print(values_testing_detail_avg)
print(values_training_detail_sd)
print(values_testing_detail_sd)


print(training_accuracies_avg)
print(training_accuracies_sd)
print(test_accuracies_avg)
print(test_accuracies_sd)
print(threshold_list)

print(max_training_acc_avg)
print(max_testing_acc_avg)
print(max_training_acc_sd)
print(max_testing_acc_sd)

#Putting the lists in a dataframe to export it to an Excel file
table_data = [
    energy_list_avg,
    gpu_list_avg,
    ram_list_avg,
    cpu_list_avg,
    energy_list_sd,
    gpu_list_sd,
    ram_list_sd,
    cpu_list_sd,
    switched_off_training_avg,
    switched_off_testing_avg,
    switched_off_training_sd,
    switched_off_testing_sd,
    switched_off_training_detail_avg,
    switched_off_testing_detail_avg,
    switched_off_training_detail_sd,
    switched_off_testing_detail_sd,
    values_training_avg,
    values_testing_avg,
    values_training_sd,
    values_testing_sd,
    values_training_detail_avg,
    values_testing_detail_avg,
    values_training_detail_sd,
    values_testing_detail_sd,
    training_accuracies_avg,
    training_accuracies_sd,
    test_accuracies_avg,
    test_accuracies_sd,
    threshold_list,
    max_training_acc_avg,
    max_testing_acc_avg,
    max_training_acc_sd,
    max_testing_acc_sd
]

row_names = [
    "energy_list_avg", "gpu_list_avg", "ram_list_avg", "cpu_list_avg",
    "energy_list_sd", "gpu_list_sd", "ram_list_sd", "cpu_list_sd",
    "switched_off_training_avg", "switched_off_testing_avg",
    "switched_off_training_sd", "switched_off_testing_sd",
    "switched_off_training_detail_avg", "switched_off_testing_detail_avg",
    "switched_off_training_detail_sd", "switched_off_testing_detail_sd",
    "values_training_avg", "values_testing_avg",
    "values_training_sd", "values_testing_sd",
    "values_training_detail_avg", "values_testing_detail_avg",
    "values_training_detail_sd", "values_testing_detail_sd",
    "training_accuracies_avg", "training_accuracies_sd",
    "test_accuracies_avg", "test_accuracies_sd",
    "threshold_list",
    "max_training_acc_avg", "max_testing_acc_avg",
    "max_training_acc_sd", "max_testing_acc_sd"
]

column_names = list(range(1, 31))

df = pd.DataFrame(table_data, index=row_names, columns=column_names)

df.to_excel("results_table.xlsx")

#####################################################
#####################################################
################# GRAPHS ############################
#####################################################

#####################################################
################## Accuracy #########################
#####################################################

def plot_with_error_bars(ax, x, y, y_sd, marker, label, color):
    ax.errorbar(x, y, yerr=y_sd, fmt=marker, label=label, color=color,
                ecolor=(0, 0, 0, 0.3), capsize=5, linestyle='None', markersize=7)
    
def plot_with_std(ax, x, y, y_sd, style, label, color):
    ax.plot(x, y, style, label=label, color=color)
    ax.fill_between(x, [m - s for m, s in zip(y, y_sd)],
                       [m + s for m, s in zip(y, y_sd)],
                    color=color, alpha=0.2)
    
fig, axs = plt.subplots(1, 3, figsize=(21, 6))

# Accuracy vs switched off channels (Training)
ax1 = axs[0]

plot_with_error_bars(ax1, switched_off_training_avg, training_accuracies_avg, training_accuracies_sd, 'o', 'Training Accuracy', 'blue')
ax1.axhline(y=1, linestyle='--', color='black', alpha=1, label='Baseline accuracy',  linewidth=2)

ax1.set_xlabel('Switched Off Channels (Training)', fontsize = 12)
ax1.set_ylabel('Accuracy', fontsize = 12)
ax1.set_title('Accuracy vs Number of switched off channels (Training)', fontsize = 16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True)
ax1.legend(loc='lower left')

# Accuracy vs switched off channels (Testing)
ax1 = axs[1]

plot_with_error_bars(ax1, switched_off_testing_avg, test_accuracies_avg, test_accuracies_sd, 's', 'Testing Accuracy', 'red')
ax1.axhline(y=0.8579, linestyle='--', color='black', alpha=1, label='Baseline Accuracy',  linewidth=2)

ax1.set_xlabel('Switched Off Channels (Testing)', fontsize = 12)
ax1.set_ylabel('Accuracy', fontsize = 12)
ax1.set_title('Accuracy vs Number of switched off channels (Testing)', fontsize = 16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True)
ax1.legend(loc='lower left')

# Accuracy vs threshold
ax1 = axs[2]

plot_with_std(ax1, threshold_list, training_accuracies_avg, training_accuracies_sd, 'o-', 'Training Accuracy', 'blue')
plot_with_std(ax1, threshold_list, test_accuracies_avg, test_accuracies_sd, 's--', 'Testing Accuracy', 'red')

ax1.set_xlabel('Threshold', fontsize = 12)
ax1.set_ylabel('Accuracy', fontsize = 12)
ax1.set_title('Accuracy vs Threshold', fontsize = 16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True)
ax1.legend(loc='lower left')

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 2 graphes au lieu de 3

# Accuracy vs avoided computations (Training)
ax1 = axs[0]

plot_with_error_bars(ax1, values_training_avg, training_accuracies_avg, training_accuracies_sd, 'o', 'Training Accuracy', 'blue')
ax1.axhline(y=1, linestyle='--', color='black', alpha=1, label='Baseline accuracy', linewidth=2)

ax1.set_xlabel('Avoided computations for training (millions)', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Accuracy vs Number of avoided computations (Training)', fontsize=16)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True)
ax1.legend(loc='lower left')

# Accuracy vs avoided computations (Testing)
ax2 = axs[1]

plot_with_error_bars(ax2, values_testing_avg, test_accuracies_avg, test_accuracies_sd, 's', 'Testing Accuracy', 'red')
ax2.axhline(y=0.8579, linestyle='--', color='black', alpha=1, label='Baseline Accuracy', linewidth=2)

ax2.set_xlabel('Avoided computations for testing (millions)', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy vs Number of avoided computations (Testing)', fontsize=16)
ax2.tick_params(axis='both', which='major', labelsize=12)
ax2.grid(True)
ax2.legend(loc='lower left')

plt.tight_layout()
plt.show()

#####################################################
############## Energy Consumption ###################
#####################################################

def plot_with_std_energy(ax, x, y, y_sd, style, label, color, baseline, baseline_label):
    ax.plot(x, y, style, label=label, color=color)
    ax.fill_between(x, [m - s for m, s in zip(y, y_sd)],
                       [m + s for m, s in zip(y, y_sd)],
                    color=color, alpha=0.2)
    ax.axhline(y=baseline, linestyle='--', color='black', alpha=0.7,
               label=baseline_label, linewidth=2)
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Energy (kWh)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.05))

def plot_with_error_bars_channels_training(ax, x, y, y_sd, marker, label, color, baseline, baseline_label):
    ax.errorbar(x, y, yerr=y_sd, fmt=marker, label=label, color=color,
                ecolor=(0, 0, 0, 0.3), capsize=5, linestyle='None', markersize=7)
    ax.axhline(y=baseline, linestyle='--', color='black', alpha=0.7,
               label=baseline_label, linewidth=2)
    ax.set_xlabel('Channels switched off at training', fontsize=12)
    ax.set_ylabel('Energy (kWh)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.05))

def plot_with_error_bars_channels_testing(ax, x, y, y_sd, marker, label, color, baseline, baseline_label):
    ax.errorbar(x, y, yerr=y_sd, fmt=marker, label=label, color=color,
                ecolor=(0, 0, 0, 0.3), capsize=5, linestyle='None', markersize=7)
    ax.axhline(y=baseline, linestyle='--', color='black', alpha=0.7,
               label=baseline_label, linewidth=2)
    ax.set_xlabel('Channels switched off at testing', fontsize=12)
    ax.set_ylabel('Energy (kWh)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.05))

def plot_with_error_bars_computations_training(ax, x, y, y_sd, marker, label, color, baseline, baseline_label):
    ax.errorbar(x, y, yerr=y_sd, fmt=marker, label=label, color=color,
                ecolor=(0, 0, 0, 0.3), capsize=5, linestyle='None', markersize=7)
    ax.axhline(y=baseline, linestyle='--', color='black', alpha=0.7,
               label=baseline_label, linewidth=2)
    ax.set_xlabel('Avoided computations at training (millions)', fontsize=12)
    ax.set_ylabel('Energy (kWh)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.05))

fig, axs = plt.subplots(1, 2, figsize=(14, 6)) 

def plot_with_error_bars_computations_testing(ax, x, y, y_sd, marker, label, color, baseline, baseline_label):
    ax.errorbar(x, y, yerr=y_sd, fmt=marker, label=label, color=color,
                ecolor=(0, 0, 0, 0.3), capsize=5, linestyle='None', markersize=7)
    ax.axhline(y=baseline, linestyle='--', color='black', alpha=0.7,
               label=baseline_label, linewidth=2)
    ax.set_xlabel('Avoided computations at testing (millions)', fontsize=12)
    ax.set_ylabel('Energy (kWh)', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 0.05))

# GPU energy vs threshold
plot_with_std_energy(axs[0], threshold_list, gpu_list_avg, gpu_list_sd, 'd-', 'Energy (GPU)', 'green',
              baseline=0.0294, baseline_label='Baseline GPU energy')
axs[0].set_title('Energy (GPU) vs Threshold', fontsize=16)

# CPU energy vs threshold
plot_with_std_energy(axs[1], threshold_list, cpu_list_avg, cpu_list_sd, 'd-', 'Energy (CPU)', 'green',
              baseline=0.0144, baseline_label='Baseline CPU energy')
axs[1].set_title('Energy (CPU) vs Threshold', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# RAM energy vs threshold
plot_with_std_energy(axs[0], threshold_list, ram_list_avg, ram_list_sd, 'd-', 'Energy (RAM)', 'green',
              baseline=0.0128, baseline_label='Baseline RAM energy')
axs[0].set_title('Energy (RAM) vs Threshold', fontsize=16)

# Total energy vs threshold
plot_with_std_energy(axs[1], threshold_list, energy_list_avg, energy_list_sd, 'd-', 'Total energy', 'green',
              baseline=0.0566, baseline_label='Baseline total energy')
axs[1].set_title('Total energy vs Threshold', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# GPU energy vs switched off channels (Training)
plot_with_error_bars_channels_training(axs[0], switched_off_training_avg, gpu_list_avg, gpu_list_sd,
                     'd', 'Energy (GPU)', 'green', baseline=0.0294, baseline_label='Baseline GPU energy')
axs[0].set_title('Energy (GPU) vs Number of channels switched off at training', fontsize=16)

# CPU energy vs switched off channels (Training)
plot_with_error_bars_channels_training(axs[1], switched_off_training_avg, cpu_list_avg, cpu_list_sd,
                     'd', 'Energy (CPU)', 'green', baseline=0.0144, baseline_label='Baseline CPU energy')
axs[1].set_title('Energy (CPU) vs Number of channels switched off at training', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# RAM energy vs switched off channels (Training)
plot_with_error_bars_channels_training(axs[0], switched_off_training_avg, ram_list_avg, ram_list_sd,
                     'd', 'Energy (RAM)', 'green', baseline=0.0128, baseline_label='Baseline RAM energy')
axs[0].set_title('Energy (RAM) vs Number of channels switched off at training', fontsize=16)

# Total energy vs switched off channels (Training)
plot_with_error_bars_channels_training(axs[1], switched_off_training_avg, energy_list_avg, energy_list_sd,
                     'd', 'Total energy', 'green', baseline=0.0566, baseline_label='Baseline total energy')
axs[1].set_title('Total energy vs Number of channels switched off at training', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 2 graphs side by side

# GPU energy vs switched off channels (Testing)
plot_with_error_bars_channels_testing(axs[0], switched_off_testing_avg, gpu_list_avg, gpu_list_sd,
                     'd', 'Energy (GPU)', 'green', baseline=0.0294, baseline_label='Baseline GPU energy')
axs[0].set_title('Energy (GPU) vs Number of channels switched off at testing', fontsize=16)

# CPU energy vs switched off channels (Testing)
plot_with_error_bars_channels_testing(axs[1], switched_off_testing_avg, cpu_list_avg, cpu_list_sd,
                     'd', 'Energy (CPU)', 'green', baseline=0.0144, baseline_label='Baseline CPU energy')
axs[1].set_title('Energy (CPU) vs Number of channels switched off at testing', fontsize=16)

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# RAM energy vs switched off channels (Testing)
plot_with_error_bars_channels_testing(axs[0], switched_off_testing_avg, ram_list_avg, ram_list_sd,
                     'd', 'Energy (RAM)', 'green', baseline=0.0128, baseline_label='Baseline RAM energy')
axs[0].set_title('Energy (RAM) vs Number of channels switched off at testing', fontsize=16)

# Total energy vs switched off channels (Testing)
plot_with_error_bars_channels_testing(axs[1], switched_off_testing_avg, energy_list_avg, energy_list_sd,
                     'd', 'Total energy', 'green', baseline=0.0566, baseline_label='Baseline total energy')
axs[1].set_title('Total energy vs Number of channels switched off at testing', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# GPU energy vs avoided computations (Training)
plot_with_error_bars_computations_training(axs[0], values_training_avg, gpu_list_avg, gpu_list_sd,
                     'd', 'Energy (GPU)', 'green', baseline=0.0294, baseline_label='Baseline GPU energy')
axs[0].set_title('Energy (GPU) vs Avoided computations at training', fontsize=16)

# CPU energy vs avoided computations (Training)
plot_with_error_bars_computations_training(axs[1], values_training_avg, cpu_list_avg, cpu_list_sd,
                     'd', 'Energy (CPU)', 'green', baseline=0.0144, baseline_label='Baseline CPU energy')
axs[1].set_title('Energy (CPU) vs Avoided computations at training', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

# RAM energy vs avoided computations (Training)
plot_with_error_bars(axs[0], values_training_avg, ram_list_avg, ram_list_sd,
                     'd', 'Energy (RAM)', 'green', baseline=0.0128, baseline_label='Baseline RAM energy')
axs[0].set_title('Energy (RAM) vs Avoided computations at training', fontsize=16)

# Total energy vs avoided computations (Training)
plot_with_error_bars(axs[1], values_training_avg, energy_list_avg, energy_list_sd,
                     'd', 'Total energy', 'green', baseline=0.0566, baseline_label='Baseline total energy')
axs[1].set_title('Total energy vs Avoided computations at training', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# GPU energy vs avoided computations (Testing)
plot_with_error_bars_computations_testing(axs[0], values_testing_avg, gpu_list_avg, gpu_list_sd,
                     'd', 'Energy (GPU)', 'green', baseline=0.0294, baseline_label='Baseline GPU energy')
axs[0].set_title('Energy (GPU) vs Avoided computations at testing', fontsize=16)

# CPU energy vs avoided computations (Testing)
plot_with_error_bars_computations_testing(axs[1], values_testing_avg, cpu_list_avg, cpu_list_sd,
                     'd', 'Energy (CPU)', 'green', baseline=0.0144, baseline_label='Baseline CPU energy')
axs[1].set_title('Energy (CPU) vs Avoided computations at testing', fontsize=16)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# RAM energy vs avoided computations (Testing)
plot_with_error_bars(axs[0], values_testing_avg, ram_list_avg, ram_list_sd,
                     'd', 'Energy (RAM)', 'green', baseline=0.0128, baseline_label='Baseline RAM energy')
axs[0].set_title('Energy (RAM) vs Avoided computations at testing', fontsize=16)

# Total energy vs avoided computations (Testing)
plot_with_error_bars(axs[1], values_testing_avg, energy_list_avg, energy_list_sd,
                     'd', 'Total energy', 'green', baseline=0.0566, baseline_label='Baseline total energy')
axs[1].set_title('Total energy vs Avoided computations at testing', fontsize=16)

plt.tight_layout()
plt.show()

###############################################################
# Threshold VS switched off channels and avoided computations #
###############################################################

def plot_with_std_treshold(ax, x, y, y_sd, label, color):
    ax.plot(x, y, 'o-', label=label, color=color)
    ax.fill_between(x, [m - s for m, s in zip(y, y_sd)],
                       [m + s for m, s in zip(y, y_sd)],
                    color=color, alpha=0.2)

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Threshold vs channels switched off (training)
plot_with_std_treshold(axs[0], threshold_list, switched_off_training_avg, switched_off_training_sd, 'Switched Off Channels (Training)', 'purple')
axs[0].set_xlabel('Threshold')
axs[0].set_ylabel('Switched Off Channels')
axs[0].set_title('Training - Number of switched Off Channels vs Threshold')
axs[0].grid(True)
axs[0].legend()

# Threshold vs channels switched off (testing)
plot_with_std_treshold(axs[1], threshold_list, switched_off_testing_avg, switched_off_testing_sd, 'Switched Off Channels (Testing)', 'brown')
axs[1].set_xlabel('Threshold')
axs[1].set_ylabel('Switched Off Channels')
axs[1].set_title('Testing - Number of switched Off Channels vs Threshold')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Threshold vs avoided computations (training)
plot_with_std(axs[0], threshold_list, values_training_avg, values_training_sd, 'Avoided computations for training (millions)', 'purple')
axs[0].set_xlabel('Threshold')
axs[0].set_ylabel('Avoided computations (millions)')
axs[0].set_title('Training - Number of avoided computations vs Threshold')
axs[0].grid(True)
axs[0].legend()

# Threshold vs avoided computations (testing)
plot_with_std(axs[1], threshold_list, values_testing_avg, values_testing_sd, 'Avoided computations for testing (millions)', 'brown')
axs[1].set_xlabel('Threshold')
axs[1].set_ylabel('Avoided computations (millions)')
axs[1].set_title('Testing - Number of avoided computations vs Threshold')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

#####################################################
############## 3D graphs per layer ##################
#####################################################

# Switched off channels per layer per threshold (training)
fig = plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
ax = fig.add_subplot(projection='3d')

colors = ['m', 'b', 'c', 'r', 'k', 'g']
yticks = threshold_list[13:18]
channels_switched_off_training_graph = switched_off_training_detail_avg[13:18]

for c, k, ys in zip(colors, yticks, channels_switched_off_training_graph):
    xs = [1,2,3,4,5,6,7,8,9]
    cs = [c] * len(xs)

    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

fig.suptitle('Number of switched off channels at training for each SE layer\nand for most relevant thresholds', fontsize = 14, y = 0.92, x = 0.58)
ax.set_xlabel('Number of the SE Layer',fontsize=10, labelpad=5)
ax.set_ylabel('Threshold', fontsize=10, labelpad=10)
ax.set_zlabel('Switched off channels at training', fontsize=10, labelpad=5)

ax.set_yticks(yticks)
ax.set_xticks(xs)
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Switched off channels per layer per threshold (testing)
fig = plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
ax = fig.add_subplot(projection='3d')

colors = ['m', 'b', 'c', 'r', 'k', 'g']
yticks = threshold_list[13:18]
channels_switched_off_testing_graph = switched_off_testing_detail_avg[13:18]

for c, k, ys in zip(colors, yticks, channels_switched_off_testing_graph):
    xs = [1,2,3,4,5,6,7,8,9]
    cs = [c] * len(xs)

    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

fig.suptitle('Number of switched off channels at testing for each SE layer\nand for most relevant thresholds', fontsize = 14, y = 0.92, x = 0.58)
ax.set_xlabel('Number of the SE Layer',fontsize=10, labelpad=5)
ax.set_ylabel('Threshold', fontsize=10, labelpad=10)
ax.set_zlabel('Switched off channels at testing', fontsize=10, labelpad=5)

ax.set_yticks(yticks)
ax.set_xticks(xs)
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Avoided computations per layer per threshold (training)
fig = plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
ax = fig.add_subplot(projection='3d')

colors = ['m', 'b', 'c', 'r', 'k', 'g']
yticks = threshold_list[13:18]
values_training_graph = values_training_detail_avg[13:18]

for c, k, ys in zip(colors, yticks, values_training_graph):
    xs = [1,2,3,4,5,6,7,8,9]
    cs = [c] * len(xs)

    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

fig.suptitle('Number of avoided computations at training for each SE layer\nand for most relevant thresholds', fontsize = 14, y = 0.92, x = 0.58)
ax.set_xlabel('Number of the SE Layer',fontsize=10, labelpad=5)
ax.set_ylabel('Threshold', fontsize=10, labelpad=10)
ax.set_zlabel('Avoided computations at training', fontsize=10, labelpad=18)

ax.set_yticks(yticks)
ax.set_xticks(xs)
ax.tick_params(axis='z', pad=10)
ax.set_box_aspect(None, zoom=0.85)

plt.show()

# Avoided computations per layer per threshold (testing)
fig = plt.figure(figsize=(8, 6))
plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
ax = fig.add_subplot(projection='3d')

colors = ['m', 'b', 'c', 'r', 'k', 'g']
yticks = threshold_list[13:18]
values_testing_graph = values_testing_detail_avg[13:18]

for c, k, ys in zip(colors, yticks, values_testing_graph):
    xs = [1,2,3,4,5,6,7,8,9]
    cs = [c] * len(xs)

    ax.bar(xs, ys, zs=k, zdir='y', color=cs, alpha=0.8)

fig.suptitle('Number of avoided computations at testing for each SE layer\nand for most relevant thresholds', fontsize = 14, y = 0.92, x = 0.58)
ax.set_xlabel('Number of the SE Layer',fontsize=10, labelpad=5)
ax.set_ylabel('Threshold', fontsize=10, labelpad=10)
ax.set_zlabel('Avoided computations at testing', fontsize=10, labelpad=18)

ax.set_yticks(yticks)
ax.set_xticks(xs)
ax.tick_params(axis='z', pad=10)
ax.set_box_aspect(None, zoom=0.85)

plt.show()