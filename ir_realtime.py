import numpy as np
from scipy import signal as sg
import soundfile as sf
import pathlib as pl
import pyaudio as pa

DEFAULT_BITRATE = 48000

# FIXME implement
DRY_WET = 0.5               # 0 = Dry
                            # 1 = Wet

# Impulse response reverb class
#
# This version is usable as a module in other
# files and can perform in realtime on an input
class IR():
    def __init__(self, ir: str = "", input: str = "", bitrate: int = DEFAULT_BITRATE):

        # Select appropriate impulse response
        match ir:
            case "" | "1" | "noise-1" | "massive":
                self._ir, self._ir_bitrate = sf.read(file="ir_demo_1_filtered_white_noise_massive.wav", always_2d=True)
            case "2" | "noise-2" | "surge" | "surge-xt":
                self._ir, self._ir_bitrate = sf.read(file="ir_demo_2_filtered_noise_surge_xt.wav", always_2d=True)
            case _:
                assert pl.Path(input).resolve().is_file(), f"File {input} does not exist!"
                self._ir, self._ir_bitrate  = sf.read(file=input, always_2d=True)

        # Import the input signal as a SoundFile object, or use a sine wave if none provided
        match input:
            case "" | "sine":
                #self._input = sine_440(bitrate)
                self._input_bitrate = bitrate
            case "arp":
                self._input, self._input_bitrate = sf.read(file="input_demo_1_arp.wav", always_2d=True)
            case _:
                assert pl.Path(input).resolve().is_file(), f"File {input} does not exist!"
                self._input, self._input_bitrate = sf.read(file=input, always_2d=True)
                

        # If bitrates do not match, do not proceed
        # FIXME convert ir to match input bitrate (low priority)
        assert self._input_bitrate == self._ir_bitrate, f"Input bitrate ({self._input_bitrate}) does not match IR bitrate ({self._ir_bitrate})"
        # If numeric types do not match, do not proceed
        # FIXME convert ir to match input type (low priority)
        assert type(self._ir[0][0]) == type(self._input[0][0]), f"Input type ({type(self._ir[0][0])}) does not match IR type ({type(self._input[0][0])})"

        # Normalize
        # FIXME switch to a realtime audio library
        non_normalized_output = sg.convolve(in1=self._ir, in2=self._input)
        self._output = non_normalized_output / np.max(np.abs(non_normalized_output))

        # Write to output file
        # FIXME Switch to a realtime audio library
        input_name = input.replace("\\", "/").split("/")[-1].replace(".wav", "").replace(".WAV", "")
        file_name = f"output_{input_name}.wav"
        sf.write(file=file_name, 
                data=(self._output), 
                samplerate=self._input_bitrate)
        
if __name__ == "__main__":
    ir = IR(ir="surge", input="sine")
    ir = IR(ir="surge", input="arp")
    quit()