import numpy as np
import scipy.io.wavfile as w
import csv


def compute_rms(data):
    return np.sqrt(1/len(data) * sum(data**2))


def compute_par(data):
    return max(abs(data)) / compute_rms(data)


def compute_zcr(data):
    data_mul = data[1:len(data)] * data[0:len(data)-1]
    data_mul[data_mul >= 0] = 0
    data_mul[data_mul < 0] = 1
    return np.mean(data_mul)


def compute_mad(data):
    data_med = np.median(data)
    return np.median(abs(data - data_med))


def compute_mean_ad(data):
    return np.mean(abs(data - np.mean(data)))


# Open CSV file to write to
csv_file_name = "TanLaiChianAlan_A0174404L_A1a.csv"
csv_file = open(csv_file_name, 'w')
writer = csv.writer(csv_file, delimiter=" ", quoting=csv.QUOTE_NONE)

# Read ground truth data set
gt_file_name = "music_speech.mf"
gt_file = open(gt_file_name, 'r')
files = gt_file.readlines()
gt_file.close()
for i in range(len(files)):
    files[i] = files[i].strip('\n')
    files[i] = files[i].split('\t')

# Iterate through files and perform computation
for j in files:
    # Read wav file
    [rate, data] = w.read(j[0])

    # Convert to floats by dividing 32768.0
    data = data / 32768.0

    # Compute Root-mean-squared (RMS)
    rms = compute_rms(data)

    # Compute Peak-to-average-ratio (PAR)
    par = compute_par(data)

    # Compute Zero crossings (ZCR)
    zcr = compute_zcr(data)

    # Compute Median absolute deviation (MAD)
    mad = compute_mad(data)

    # Compute Mean Absolute Deviation (MEAN-AD)
    mean_ad = compute_mean_ad(data)

    # String output
    output = "{},{:0.6f},{:0.6f},{:0.6f},{:0.6f},{:0.6f}".format(j[0], rms, par, zcr, mad, mean_ad)

    # Write to csv file
    writer.writerow([output])

# Close file
csv_file.close()
