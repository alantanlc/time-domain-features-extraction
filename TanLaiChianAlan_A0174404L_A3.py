import numpy as np
import scipy.io.wavfile as w
import scipy.signal as sig
import scipy.fftpack as f
import pylab


def plot_triangular_windows_overall(fs, nfilt, nfft):
    low_freq_mel = 0
    high_freq_mel = 1127 * np.log10(1 + (fs / 2) / 700)   # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) # Equally spaced in Mel scale
    hz_points = 700 * (10**(mel_points / 1127) - 1)   # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / fs)

    pylab.figure()
    pylab.title('26 Triangular MFCC filters, 22050Hz signal, window size 1024')
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Amplitude')
    pylab.xlim([0, 12000])
    pylab.ylim([0, 1])
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m-1])   # left
        f_m = int(bin[m])           # center
        f_m_plus = int(bin[m + 1])  # right

        pylab.plot([f_m_minus, f_m, f_m_plus], [0, 1, 0])
    pylab.show()


def plot_triangular_windows_300hz(fs, nfilt, nfft):
    low_freq_mel = 0
    high_freq_mel = 1127 * np.log10(1 + (fs / 2) / 700)   # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) # Equally spaced in Mel scale
    hz_points = 700 * (10**(mel_points / 1127) - 1)   # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / fs)

    pylab.figure()
    pylab.title('26 Triangular MFCC filters, 22050Hz signal, window size 1024')
    pylab.xlabel('Frequency (Hz)')
    pylab.ylabel('Amplitude')
    pylab.xlim([0, 300])
    pylab.ylim([0, 1])
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m-1])   # left
        f_m = int(bin[m])           # center
        f_m_plus = int(bin[m + 1])  # right

        pylab.plot([f_m_minus, f_m, f_m_plus], [0, 1, 0])
    pylab.show()


def compute_filter_bank(fs, nfilt, nfft):
    low_freq_mel = 0
    high_freq_mel = 1127 * np.log10(1 + (fs / 2) / 700)   # Convert Hz to Mel
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) # Equally spaced in Mel scale
    hz_points = 700 * (10**(mel_points / 1127) - 1)   # Convert Mel to Hz
    bin = np.floor((nfft + 1) * hz_points / fs)

    fbank = np.zeros((nfilt, int(np.floor(nfft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m-1])   # left
        f_m = int(bin[m])           # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank


def compute_uncorrected_std(data):
    return np.sqrt(sum((data - np.mean(data))**2) / (len(data)))


# Open ARFF file to write to
filename = "TanLaiChianAlan_A0174404L_A3.arff"
arff_file = open(filename, 'w')

# Write header to ARFF file
arff_file.write("@RELATION music_speech\n@ATTRIBUTE MFCC_0 NUMERIC\n@ATTRIBUTE MFCC_1 NUMERIC\n@ATTRIBUTE MFCC_2 NUMERIC\n@ATTRIBUTE MFCC_3 NUMERIC\n@ATTRIBUTE MFCC_4 NUMERIC\n@ATTRIBUTE MFCC_5 NUMERIC\n@ATTRIBUTE MFCC_6 NUMERIC\n@ATTRIBUTE MFCC_7 NUMERIC\n@ATTRIBUTE MFCC_8 NUMERIC\n@ATTRIBUTE MFCC_9 NUMERIC\n@ATTRIBUTE MFCC_10 NUMERIC\n@ATTRIBUTE MFCC_11 NUMERIC\n@ATTRIBUTE MFCC_12 NUMERIC\n@ATTRIBUTE MFCC_13 NUMERIC\n@ATTRIBUTE MFCC_14 NUMERIC\n@ATTRIBUTE MFCC_15 NUMERIC\n@ATTRIBUTE MFCC_16 NUMERIC\n@ATTRIBUTE MFCC_17 NUMERIC\n@ATTRIBUTE MFCC_18 NUMERIC\n@ATTRIBUTE MFCC_19 NUMERIC\n@ATTRIBUTE MFCC_20 NUMERIC\n@ATTRIBUTE MFCC_21 NUMERIC\n@ATTRIBUTE MFCC_22 NUMERIC\n@ATTRIBUTE MFCC_23 NUMERIC\n@ATTRIBUTE MFCC_24 NUMERIC\n@ATTRIBUTE MFCC_25 NUMERIC\n@ATTRIBUTE MFCC_26 NUMERIC\n@ATTRIBUTE MFCC_27 NUMERIC\n@ATTRIBUTE MFCC_28 NUMERIC\n@ATTRIBUTE MFCC_29 NUMERIC\n@ATTRIBUTE MFCC_30 NUMERIC\n@ATTRIBUTE MFCC_31 NUMERIC\n@ATTRIBUTE MFCC_32 NUMERIC\n@ATTRIBUTE MFCC_33 NUMERIC\n@ATTRIBUTE MFCC_34 NUMERIC\n@ATTRIBUTE MFCC_35 NUMERIC\n@ATTRIBUTE MFCC_36 NUMERIC\n@ATTRIBUTE MFCC_37 NUMERIC\n@ATTRIBUTE MFCC_38 NUMERIC\n@ATTRIBUTE MFCC_39 NUMERIC\n@ATTRIBUTE MFCC_40 NUMERIC\n@ATTRIBUTE MFCC_41 NUMERIC\n@ATTRIBUTE MFCC_42 NUMERIC\n@ATTRIBUTE MFCC_43 NUMERIC\n@ATTRIBUTE MFCC_44 NUMERIC\n@ATTRIBUTE MFCC_45 NUMERIC\n@ATTRIBUTE MFCC_46 NUMERIC\n@ATTRIBUTE MFCC_47 NUMERIC\n@ATTRIBUTE MFCC_48 NUMERIC\n@ATTRIBUTE MFCC_49 NUMERIC\n@ATTRIBUTE MFCC_50 NUMERIC\n@ATTRIBUTE MFCC_51 NUMERIC\n@ATTRIBUTE class {music,speech}\n\n@DATA\n");

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
    [rate, signal] = w.read(j[0])

    # Convert to floats by dividing 32768.0
    # signal = signal / 32768.0

    # Apply pre-emphasis filter
    pre_emphasis = 0.95
    data = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

    # Split data into 1290 buffers of length 1024 with 50% overlap (or a hop size of 512)
    num_buffers = 1290  # rows
    buffer_size = 1024  # columns
    hop_size = 512
    buffer_data = np.zeros((num_buffers, buffer_size))
    for i in range(num_buffers):
        buffer_data[i] = data[i*hop_size:i*hop_size+buffer_size]

    # Compute hamming window
    hamming = sig.hamming(buffer_size)

    # Compute filter bank
    num_mel_filters = 26
    fbank = compute_filter_bank(rate, num_mel_filters, buffer_size)
    # plot_triangular_windows_overall(rate, num_mel_filters, rate)
    # plot_triangular_windows_300hz(rate, num_mel_filters, rate)

    # Calculate the MFCCs for each window
    i = 0
    mfcc = np.zeros((num_buffers, num_mel_filters))
    for data in buffer_data:
        # Apply hamming window
        data_hamming = data * hamming

        # Discrete Fourier Transform
        data_fft = f.fft(np.fft.rfft(data_hamming, buffer_size))

        # Convert spectrum to absolute values
        data_mag = np.abs(data_fft)

        # Apply mel-frequency filtering
        data_mel = np.dot(data_mag, fbank.T)

        # Apply log
        data_log = np.log10(data_mel)

        # Apply DCT
        mfcc[i] = f.dct(data_log)

        # Increment i
        i = i + 1

    # Compute mean and uncorrected sample standard deviation for each feature
    result = np.zeros(num_mel_filters * 2)
    for i in range(num_mel_filters):
        result[i] = np.mean(mfcc[:, i])
        result[i + num_mel_filters] = compute_uncorrected_std(mfcc[:, i])

    # Write to file
    for i in range(num_mel_filters):
        arff_file.write("{:0.6f},".format(result[i]))
    for i in range(num_mel_filters):
        arff_file.write("{:0.6f},".format(result[i + num_mel_filters]))
    arff_file.write("{}\n".format(j[1]))

# Close file
arff_file.close()