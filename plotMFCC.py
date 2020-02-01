from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os
#os.system('ffmpeg -i FluteTone.mp3 FluteTone.wav')


(rate,sig) = wav.read("violin.wav")
mfcc_feat = mfcc(sig,rate)

fig = plt.figure()
ax1 = fig.add_subplot(311)
print(mfcc_feat)
plt.plot(mfcc_feat)
#plt.title('violin Tone')
plt.xlabel('Violin Frame Number')
plt.ylabel('Violin MFCC Coefficients')

(rate,sig) = wav.read("piano.wav")
mfcc_feat = mfcc(sig,rate)
ax2 = fig.add_subplot(312)
print(mfcc_feat)
plt.plot(mfcc_feat)
#plt.title('Piano Tone')
plt.xlabel('Piano Frame Number')
plt.ylabel('Piano MFCC Coefficients')


(rate,sig) = wav.read("sitar.wav")
mfcc_feat = mfcc(sig,rate)
ax3 = fig.add_subplot(313)
print(mfcc_feat)
plt.plot(mfcc_feat)
#plt.title('Sitar Tone')
plt.xlabel('Sitar Frame Number')
plt.ylabel('Sitar MFCC Coefficients')

plt.show()
