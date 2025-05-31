import zipfile
import os
import pandas as pd
import re
import statistics as stat

def analyze_results_energy(results_path):
  """Analyzes energy consumption from results.

  Parameters
  ----------
  results_path (String): Path to the results directory. This directory must be structured with two folders: Emissions (containing energy consumption output files) and Output (containing other output files).

  Returns
  -------
  avg_total_energy (float): Average of total energy consumption across 5 repetitions and summed over all epochs.
  avg_energy_gpu (float): Average of gpu energy consumption across 5 repetitions and summed over all epochs.
  avg_energy_ram (float): Average of ram energy consumption across 5 repetitions and summed over all epochs.
  avg_energy_cpu (float): Average of cpu energy consumption across 5 repetitions and summed over all epochs.
  sd_total_energy (float): Standard deviation of total energy consumption across 5 repetitions and summed over all epochs.
  sd_energy_gpu (float): Standard deviation of gpu energy consumption across 5 repetitions and summed over all epochs.
  sd_energy_ram (float): Standard deviation of ram energy consumption across 5 repetitions and summed over all epochs.
  sd_energy_cpu (float): Standard deviation of cpu energy consumption across 5 repetitions and summed over all epochs.
  """

  emissions_dir = os.path.join(results_path, "Emissions")

  total_energy_list = []
  energy_gpu_list = []
  energy_ram_list = []
  energy_cpu_list = []
  for experience in range (0,5):
    filename = "emissions_" + str(experience) + ".csv"
    filepath = os.path.join(emissions_dir, filename)
    df = pd.read_csv(filepath)
    total_energy = df["energy_consumed"].sum()
    total_energy_list.append(total_energy)
    energy_cpu = df["cpu_energy"].sum()
    energy_cpu_list.append(energy_cpu)
    energy_gpu = df["gpu_energy"].sum()
    energy_gpu_list.append(energy_gpu)
    energy_ram = df["ram_energy"].sum()
    energy_ram_list.append(energy_ram)
  avg_total_energy = stat.mean(total_energy_list)
  avg_energy_gpu = stat.mean(energy_gpu_list)
  avg_energy_ram = stat.mean(energy_ram_list)
  avg_energy_cpu = stat.mean(energy_cpu_list)
  sd_total_energy = stat.stdev(total_energy_list)
  sd_energy_gpu = stat.stdev(energy_gpu_list)
  sd_energy_ram = stat.stdev(energy_ram_list)
  sd_energy_cpu = stat.stdev(energy_cpu_list)

  return avg_total_energy, avg_energy_gpu, avg_energy_ram, avg_energy_cpu, sd_total_energy, sd_energy_gpu, sd_energy_ram, sd_energy_cpu

def analyze_results(results_path):
  """Analyzes training and testing accuracies from results.

  Parameters
  ----------
  results_path (String): Path to the results directory. This directory must be structured with two folders: Emissions (containing energy consumption output files) and Output (containing other output files).

  Returns
  -------
  results (dictionary): Contains the average of training and testing accuracies, the average maximum training and testing accuraciess across all epochs and the standard deviation of all these values. Averages and standard deviations are computed across the 5 iterations of each experiment.
  """

  output_dir = os.path.join(results_path, "Output")

  results = {}
  training_acc = []
  test_acc = []
  max_training_acc = []
  max_testing_acc = []
  for experience in range (0,5):
    filename = "Output_" + str(int(experience)) + ".txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'r') as f:
      lines = f.readlines()

      lines = lines[-8:]
      training_acc.append(float(lines[2].split(":")[1]))
      test_acc.append(float(lines[4].split(":")[1]))
      max_training_acc.append(float(lines[6].split(":")[1]))
      max_testing_acc.append(float(lines[7].split(":")[1]))


  results["training_accuracy_avg"] = round(stat.mean(training_acc),4)
  results["test_accuracy_avg"] = round(stat.mean(test_acc),4)
  results["training_accuracy_sd"] = round(stat.stdev(training_acc),4)
  results["test_accuracy_sd"] = round(stat.stdev(test_acc),4)
  results["max_training_acc_avg"] = round(stat.mean(max_training_acc),4)
  results["max_testing_acc_avg"] = round(stat.mean(max_testing_acc),4)
  results["max_training_acc_sd"] = round(stat.stdev(max_training_acc),4)
  results["max_testing_acc_sd"] = round(stat.stdev(max_testing_acc),4)


  return results

#Unzipping the results folder. The name of the folder can be changed.
with zipfile.ZipFile("ResultsBaseline.zip", 'r') as zip_ref:
    zip_ref.extractall("ResultsBaseline")

initial_folder = "ResultsBaseline"
#Path is to be changed if different
os.chdir("/content/%s/%s"%(initial_folder, initial_folder))
results_path = os.getcwd()

#Analyzing energy consumption results
emissions_results = analyze_results_energy(results_path)

#Rounding energy consumption results
energy_list_avg = (round(float(emissions_results[0]),4))
gpu_list_avg = (round(float(emissions_results[1]),4))
ram_list_avg = (round(float(emissions_results[2]),4))
cpu_list_avg = (round(float(emissions_results[3]),4))
energy_list_sd = (round(float(emissions_results[4]),4))
gpu_list_sd = (round(float(emissions_results[5]),4))
ram_list_sd = (round(float(emissions_results[6]),4))
cpu_list_sd = (round(float(emissions_results[7]),4))

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

#Extracting values from the dictionary
training_accuracies_avg = results["training_accuracy_avg"]
training_accuracies_sd = results['training_accuracy_sd']
test_accuracies_avg = results['test_accuracy_avg']
test_accuracies_sd = results['test_accuracy_sd']

max_training_acc_avg = results['max_training_acc_avg']
max_testing_acc_avg = results['max_testing_acc_avg']
max_training_acc_sd = results['max_training_acc_sd']
max_testing_acc_sd = results['max_testing_acc_sd']

#Printing other results
print(training_accuracies_avg)
print(training_accuracies_sd)
print(test_accuracies_avg)
print(test_accuracies_sd)

print(max_training_acc_avg)
print(max_testing_acc_avg)
print(max_training_acc_sd)
print(max_testing_acc_sd)

#Putting the values in a dataframe to export it to an Excel file
table_data = [
    energy_list_avg,
    gpu_list_avg,
    ram_list_avg,
    cpu_list_avg,
    energy_list_sd,
    gpu_list_sd,
    ram_list_sd,
    cpu_list_sd,
    training_accuracies_avg,
    training_accuracies_sd,
    test_accuracies_avg,
    test_accuracies_sd,
    max_training_acc_avg,
    max_testing_acc_avg,
    max_training_acc_sd,
    max_testing_acc_sd
]

row_names = [
    "energy_list_avg", "gpu_list_avg", "ram_list_avg", "cpu_list_avg",
    "energy_list_sd", "gpu_list_sd", "ram_list_sd", "cpu_list_sd",
    "training_accuracies_avg", "training_accuracies_sd",
    "test_accuracies_avg", "test_accuracies_sd",
    "max_training_acc_avg", "max_testing_acc_avg",
    "max_training_acc_sd", "max_testing_acc_sd"
]

column_names = list(range(1, 2))

df = pd.DataFrame(table_data, index=row_names, columns=column_names)

df.to_excel("baseline_results_table.xlsx")
