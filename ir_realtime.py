import numpy as np
from scipy.fft import fft, ifft
from scipy import signal as sg
import soundfile as sf
import pathlib as pl

BUFFER_SIZE = 256
DRY_WET = 0.5
REVERB_HEADROOM_CONSTANT = 0.1
IR_FFT_SIZE = 2 * BUFFER_SIZE
BITRATE = 48000

# Default impulse response if none specified
def impulse():
    impulse = np.zeros((1, 2))
    impulse[0, :] = 1
    return impulse

# Impulse response reverb class
#
# This version is usable as a module in other
# files and can perform in realtime on an input
class IR():
    def __init__(self, ir: str = None, 
                 bitrate: int = BITRATE, buffer_size: int = BUFFER_SIZE,
                 dry_wet: float = DRY_WET):

        # Select appropriate impulse response
        match ir:
            case "0" | "" | None:
                self._ir = impulse();
                self._ir_bitrate = bitrate
            case "1" | "noise-1" | "massive":
                self._ir, self._ir_bitrate = sf.read(file="ir_demo_1_filtered_white_noise_massive.wav", always_2d=True)
            case "2" | "noise-2" | "surge" | "surge-xt":
                self._ir, self._ir_bitrate = sf.read(file="ir_demo_2_filtered_noise_surge_xt.wav", always_2d=True)
            case _:
                assert pl.Path(ir).resolve().is_file(), f"File {ir} does not exist!"
                self._ir, self._ir_bitrate  = sf.read(file=ir, always_2d=True)

        # If bitrates of IR does not match bitrate provided, fix it with resampling
        if bitrate != self._ir_bitrate:
            resamp_left = sg.resample_poly(
                self._ir[:, 0],
                up=bitrate,
                down=self._ir_bitrate)
            resamp_right = sg.resample_poly(
                self._ir[:, 1],
                up=bitrate,
                down=self._ir_bitrate)
            self._ir = np.column_stack([resamp_left, resamp_right])

        # Store parameters as class fields
        self._bitrate = bitrate
        self._buffer_size = buffer_size
        self._tail = np.zeros((0, 2))
        self._dry_wet = dry_wet

        # Normalize impulse response by both peak and rms
        ir_max = np.max(np.abs(self._ir))
        if ir_max > 0:
            self._ir /= (ir_max + 1e-12)            # add a very small constant to prevent div0
        ir_rms = np.sqrt(np.mean(self._ir ** 2))
        self._ir /= (ir_rms + 1e-12)                # add a very small constant to prevent div0
        self._ir *= REVERB_HEADROOM_CONSTANT

        # Create IR partitions
        self._num_partitions = int(np.ceil(len(self._ir) / BUFFER_SIZE)) 
        self._ir_partitions_left = self._partition_ir(0)
        self._ir_partitions_right = self._partition_ir(1)

        # Previous input tracking
        self._input_history_left = np.zeros((self._num_partitions, IR_FFT_SIZE), dtype=complex)
        self._input_history_right = np.zeros((self._num_partitions, IR_FFT_SIZE), dtype=complex)

        # Overlap tracking
        self._overlap_left = np.zeros(BUFFER_SIZE)
        self._overlap_right = np.zeros(BUFFER_SIZE)

    # Used for splitting impulse responses into partitions
    def _partition_ir(self, channel: int = 0):

        partitions = np.zeros((self._num_partitions, IR_FFT_SIZE), dtype=complex)
        for i in range(self._num_partitions):
            start = i * BUFFER_SIZE
            end = min(start + BUFFER_SIZE, len(self._ir))

            # Precomputing ffts of IR chunks ahead of time to save compute power in the use() function
            ir_chunk = self._ir[start:end, channel]
            ir_padded = np.zeros(IR_FFT_SIZE)
            ir_padded[:len(ir_chunk)] = ir_chunk
            partitions[i] = fft(ir_padded)

        return partitions

    def updateDryWet(self, newDW):
        self._dry_wet = newDW
        
    def use(self, input_signal: np.array) -> np.array:

        # Convert to stereo if mono
        if input_signal.ndim == 1:
            input_signal = np.column_stack([input_signal, input_signal])

        # Zero-pad input
        zero_padded_input_left = np.zeros(IR_FFT_SIZE)
        zero_padded_input_left[:BUFFER_SIZE] = input_signal[:, 0]
        zero_padded_input_right = np.zeros(IR_FFT_SIZE)
        zero_padded_input_right[:BUFFER_SIZE] = input_signal[:, 1]

        # take fft of padded input
        fft_left = fft(zero_padded_input_left)
        fft_right = fft(zero_padded_input_right)

        # Roll the input history buffer
        self._input_history_left = np.roll(self._input_history_left, 1, axis=0)
        self._input_history_right = np.roll(self._input_history_right, 1, axis=0)

        # Store newest fft
        self._input_history_left[0] = fft_left
        self._input_history_right[0] = fft_right

        # Initialize output arrays
        output_freq_left = np.zeros(IR_FFT_SIZE, dtype=complex)
        output_freq_right = np.zeros(IR_FFT_SIZE, dtype=complex)

        # Partitioned Convolution
        for i in range(self._num_partitions):
                
            # Add to the accumulator data
            output_freq_left += (self._input_history_left[i] * self._ir_partitions_left[i])
            output_freq_right += (self._input_history_right[i] * self._ir_partitions_right[i])

        # Inverse Fourier Transform on the resultant frequency, discarding unneeded portion
        output_time_left = np.real(ifft(output_freq_left, n=IR_FFT_SIZE))
        output_time_right = np.real(ifft(output_freq_right, n=IR_FFT_SIZE))
            
        # Overlap-add
        output_block_left = (output_time_left[:BUFFER_SIZE] + self._overlap_left)
        output_block_right = (output_time_right[:BUFFER_SIZE] + self._overlap_right)
            
        # Save overlap
        self._overlap_left = output_time_left[BUFFER_SIZE:]
        self._overlap_right = output_time_right[BUFFER_SIZE:]
            
        output = np.column_stack([output_block_left, output_block_right])

        # Attenuate to prevent clipping
        output *= REVERB_HEADROOM_CONSTANT

        # Dry-Wet Mix
        output[:, 0] = (self._dry_wet * output[:, 0]) + (((1 - self._dry_wet) * input_signal[:, 0]))
        output[:, 1] = (self._dry_wet * output[:, 1]) + (((1 - self._dry_wet) * input_signal[:, 1]))
        
        return(output)

if __name__ == "__main__":
    print("This version of the reverb is meant to be used as a module in another program and is not useful when run on its own.")
    quit()