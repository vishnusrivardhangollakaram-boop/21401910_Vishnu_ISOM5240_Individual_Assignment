"""Stand-in gTTS: makes a speech-like MP3 with ffmpeg (length proportional to text) and logs request times."""
import subprocess, numpy as np, io, time
from scipy.io import wavfile
REQUESTS = []
class gTTS:
    def __init__(self, text, lang="en", tld="com", timeout=None): self.text = text; self.tld = tld
    def write_to_fp(self, fp):
        REQUESTS.append((time.perf_counter(), self.tld, self.text))
        sr = 24000; dur = max(0.5, len(self.text.split()) * 0.33)
        t = np.arange(int(sr*dur))/sr
        f0 = 210 + 30*np.sin(2*np.pi*0.7*t)
        sig = sum(np.sin(2*np.pi*k*np.cumsum(f0)/sr)/k for k in range(1,8)) * (0.5+0.5*np.sin(2*np.pi*3.5*t))**2 * 0.2
        b = io.BytesIO(); wavfile.write(b, sr, (sig*32767).astype(np.int16))
        mp3 = subprocess.run(["ffmpeg","-loglevel","error","-f","wav","-i","pipe:0","-f","mp3","pipe:1"], input=b.getvalue(), capture_output=True, check=True).stdout
        fp.write(mp3)
