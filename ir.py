import numpy as np
from scipy import signal as sg
import soundfile as sf


# Generate a variable-length 440 Hz sine wave. Used if user does not supply an input.
SINE_LEN_SECONDS = 2
def sine_440(bitrate: int = 44800):
    sine_wave = (np.sin(2 * np.pi * 440 * np.arange(bitrate * SINE_LEN_SECONDS) / bitrate) * 32767).astype(np.int16)

    # Smooth heads and tails
    fade_len = bitrate // 10
    for i in range(fade_len):
        sine_wave[i] *= (i % fade_len) / fade_len
        sine_wave[-i] *= (i % fade_len) / fade_len

    sf.write(file="sine_440.wav", data=sine_wave, samplerate=bitrate)
    return sf.SoundFile(file="sine_440.wav")


class IR():
    def __init__(self, ir_file: str = "", input: str = ""):

        # Select appropriate impulse response
        match ir_file:
            case "":
                self._ir = sf.SoundFile(file="white_noise_massive.wav")
            case _:
                print("Command line arguments and user-supplied impulse responses are currently unsupported!")

        # Import the input signal as a SoundFile object, or use a sine wave if none provided
        if input != "":
            self._input = sf.SoundFile(file=input)
        else:
            self._input = sine_440()
        self._bitrate = self._input.samplerate

        # FIXME do the actual convolution operations!
        self._output = self._input
        #self.convolve()

    # Play the dry input signal
    # FIXME make realtime instead of writing a file!
    def play_dry(self):
        sf.write(file="dry_output.wav", data=self._input.read(dtype=np.int16), samplerate=self._bitrate)   # FIXME add autodetection for inputs that are not int16!

    # Play the wet output signal 
    # FIXME make realtime instead of writing a file!
    # FIXME add dry-wet control!
    def play_wet(self):
        sf.write(file="wet_output.wav", data=self._output.read(dtype=np.int16), samplerate=self._bitrate)   # FIXME why is only this one creating a 0 second wav file?

    # FIXME do the actual convolution operations!
    def convolve(self):
        pass
        
if __name__ == "__main__":
    ir = IR()
    ir.play_dry()
    ir.play_wet()
    quit()