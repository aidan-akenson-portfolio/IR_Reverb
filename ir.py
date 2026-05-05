import numpy as np
from scipy import signal as sg
import soundfile as sf
import pathlib as pl

DEFAULT_BITRATE = 48000

# FIXME implement
DRY_WET = 0.5               # 0 = Dry
                            # 1 = Wet

# Generate a variable-length 440 Hz sine wave. Used if user does not supply an input.
SINE_LEN_SECONDS = 2
def sine_440(bitrate: int = DEFAULT_BITRATE):
    sine_wave = (np.sin(2 * np.pi * 440 * np.arange(bitrate * SINE_LEN_SECONDS) / bitrate))

    # Smooth heads and tails
    fade_len = bitrate // 10
    for i in range(fade_len):
        sine_wave[i] *= (i % fade_len) / fade_len
        sine_wave[-i] *= (i % fade_len) / fade_len

    # Make stereo. 
    # 
    # Soundfile uses format (samples, channels) so transposition is needed.
    # If audio library changes when converting to realtime, this transposition
    # may become unnecessary.
    sine_wave = np.array([sine_wave, sine_wave]).T

    # FIXME remove once no longer needed
    sf.write(file="sine_440.wav", data=sine_wave, samplerate=bitrate)

    return sine_wave

class IR():
    def __init__(self, ir: str = "", input: str = ""):

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
                self._input = sine_440()
                self._input_bitrate = DEFAULT_BITRATE
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