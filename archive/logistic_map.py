import numpy as np
import matplotlib.pyplot as plt

def logistic_map(r, x0, num_iterations):
    result = []
    x = x0
    for _ in range(num_iterations):
        result.append(x)
        x = r * x * (1 - x)

    return result




signal = np.sin(np.array(np.linspace(-np.pi, np.pi*10, 201)))

plt.plot(signal)
plt.show()



signal = np.sin(np.array(np.linspace(-np.pi, np.pi*10, 201)))

plt.plot(signal)
plt.show()

# Generating a signal (sine wave for example)
fs = 1000  # Sampling frequency
f = 5      # Frequency of the sine wave
t = np.arange(0, 1, 1/fs)  # Time vector
signal = np.sin(2*np.pi*f*t)

# Generating white noise
noise_amplitude = 0.2
white_noise = np.random.normal(0, noise_amplitude, len(t))

# Adding white noise to the signal
noisy_signal = signal + white_noise

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(t, signal, label='Original Signal')
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Signal with White Noise')
plt.legend()
plt.grid(True)
plt.show()