import numpy as np
from scipy import signal as sg
import soundfile as sf
import pathlib as pl

IR_CHUNK_SIZE = 1024
BITRATE = 48000
BUFFER_SIZE = 256
DRY_WET = 0.5
REVERB_ON = True


# Impulse response reverb class
#
# This version is usable as a module in other
# files and can perform in realtime on an input
class IR():
    def __init__(self, ir: str = "1", 
                 bitrate: int = BITRATE, buffer_size: int = BUFFER_SIZE,
                 dry_wet: float = DRY_WET):

        # Select appropriate impulse response
        match ir:
            case "1" | "noise-1" | "massive":
                self._ir, self._ir_bitrate = sf.read(file="ir_demo_1_filtered_white_noise_massive.wav", always_2d=True)
            case "2" | "noise-2" | "surge" | "surge-xt":
                self._ir, self._ir_bitrate = sf.read(file="ir_demo_2_filtered_noise_surge_xt.wav", always_2d=True)
            case _:
                assert pl.Path(ir).resolve().is_file(), f"File {ir} does not exist!"
                self._ir, self._ir_bitrate  = sf.read(file=ir, always_2d=True)

        # If bitrates of IR does not match bitrate provided, fix it with resampling
        if bitrate > self._ir_bitrate:
            resamp_left = sg.resample_poly(self._ir[:, 0], up=bitrate, down=self._ir_bitrate)
            resamp_right = sg.resample_poly(self._ir[:, 1], up=bitrate, down=self._ir_bitrate)
            self._ir = np.column_stack([resamp_left, resamp_right])
        elif bitrate < self._ir_bitrate:
            resamp_left = sg.resample_poly(self._ir[:, 0], up=self._ir_bitrate, down=bitrate)
            resamp_right = sg.resample_poly(self._ir[:, 1], up=self._ir_bitrate, down=bitrate)
            self._ir = np.column_stack([resamp_left, resamp_right])

        # Store parameters as class fields
        self._bitrate = bitrate
        self._buffer_size = buffer_size
        self._tail = np.zeros((0, 2))
        self._dry_wet = dry_wet
        self._ir_max = np.max(np.abs(self._ir))
        assert self._ir_max > 0, "IR files may not be silent"

        # Create IR partitions
        self._num_partitions = int(np.ceil(len(self._ir) / IR_CHUNK_SIZE)) 
        self._ir_partitions_left = self._partition_ir(0)
        self._ir_partitions_right = self._partition_ir(1)
        self._partition_index = 0

        # Pending output tracking
        self._accumulator_left = np.zeros(len(self._ir) + IR_CHUNK_SIZE + BUFFER_SIZE)  
        self._accumulator_right = np.zeros(len(self._ir) + IR_CHUNK_SIZE + BUFFER_SIZE)      
        self._output_position = 0 

    # Used for splitting impulse responses into partitions
    def _partition_ir(self, channel: int = 0):

        partitions = np.zeros((self._num_partitions, IR_CHUNK_SIZE, 2))
        for i in range(self._num_partitions):
            start = i * IR_CHUNK_SIZE
            end = min(start + IR_CHUNK_SIZE, len(self._ir))
            chunk_length = end - start
            partitions[i, :chunk_length, 0] = self._ir[start:end, 0]
            partitions[i, :chunk_length, 1] = self._ir[start:end, 1]

        return partitions[:, :, channel]

    def updateDryWet(self, newDW):
        self._dry_wet = newDW
        
    def use(self, input_signal: np.array) -> np.array:

        # FIXME implement reverb toggle better once working
        if REVERB_ON:
                
            # Convert to stereo if mono
            if input_signal.ndim == 1:
                input_signal = np.column_stack([input_signal, input_signal])

            # Convolve with current ir partition
            convolved_input_left = sg.convolve(input_signal[:, 0], self._ir_partitions_left[self._partition_index])
            convolved_input_right = sg.convolve(input_signal[:, 1], self._ir_partitions_right[self._partition_index])

            # Place in accumulator at correct time offset
            start_position = self._partition_index * IR_CHUNK_SIZE
            self._accumulator_left[start_position : start_position + len(convolved_input_left)] += convolved_input_left
            self._accumulator_right[start_position : start_position + len(convolved_input_right)] += convolved_input_right

            # Extract buffer from accumulator
            output = np.column_stack([self._accumulator_left[self._output_position : self._output_position + BUFFER_SIZE],
                                     self._accumulator_right[self._output_position : self._output_position + BUFFER_SIZE]])
            
            # Right after extracting output, before dry-wet mix:
            if self._partition_index == 0:  # Only print on partition wraparound
                print(f"Read position {self._output_position}: "
                    f"wet_left max={np.max(np.abs(output[:, 0])):.6f}, "
                    f"wet_right max={np.max(np.abs(output[:, 1])):.6f}")
                
            # Advance partition and output indices
            self._output_position += BUFFER_SIZE 
            self._partition_index = (self._partition_index + 1) % self._num_partitions

            # Reset accumulator if fully read
            if self._output_position > len(self._ir):
                self._accumulator_left[:] = 0 
                self._accumulator_right[:]
                self._output_position = 0
                    
            # Dry-Wet Mix
            output[:, 0] = (self._dry_wet * output[:, 0]) + (((1 - self._dry_wet) * input_signal[:, 0]))
            output[:, 1] = (self._dry_wet * output[:, 1]) + (((1 - self._dry_wet) * input_signal[:, 1]))
                
            # Interleave for pyAudio output format
            output_flat = output.flatten('F')
            
            return(output_flat)

        else:
            return input_signal

if __name__ == "__main__":
    print("This version of the reverb is meant to be used as a module in another program and is not useful when run on its own.")
    quit()