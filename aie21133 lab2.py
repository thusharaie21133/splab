#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import librosa
import matplotlib.pyplot as plt

speech_file = r"C:\Users\Hp\Downloads\speech\Recording.wav" 
y, sr = librosa.load(speech_file)

t = np.arange(0, len(y)) / sr

dy = np.diff(y)

dy = np.append(dy, dy[-1])

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, y, 'b')
plt.title('Original Speech Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(t, dy, 'r')
plt.title('First Derivative of Speech Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()


# In[13]:


pip install librosa


# In[18]:


import numpy as np
import librosa
import matplotlib.pyplot as plt

speech_file = r"C:\Users\Hp\Downloads\speech\Recording.wav"  
y, sr = librosa.load(speech_file)

dy = np.diff(y)

zero_crossings = np.where(np.diff(np.sign(dy)))[0]


lengths = np.diff(zero_crossings) / sr  

speech_threshold = 0.05 
speech_indices = np.where(np.abs(y) > speech_threshold)[0]
silence_indices = np.where(np.abs(y) <= speech_threshold)[0]

speech_zero_crossings = np.intersect1d(zero_crossings, speech_indices)
silence_zero_crossings = np.intersect1d(zero_crossings, silence_indices)

avg_length_speech = np.mean(np.diff(speech_zero_crossings)) / sr
avg_length_silence = np.mean(np.diff(silence_zero_crossings)) / sr

print("Average length between consecutive zero crossings for speech: {:.2f} seconds".format(avg_length_speech))
print("Average length between consecutive zero crossings for silence: {:.2f} seconds".format(avg_length_silence))


# In[19]:


import librosa


your_audio_file = r"C:\Users\Hp\Downloads\speech\THUSHAR.wav"  
y, sr = librosa.load(your_audio_file)


duration_y = librosa.get_duration(y=y, sr=sr)


teammate_audio_file = r"C:\Users\Hp\Downloads\speech\MOKI.wav"  
y_teammate, sr_teammate = librosa.load(teammate_audio_file)
duration_y_teammate = librosa.get_duration(y=y_teammate, sr=sr_teammate)

print("Your speech duration:", duration_y, "seconds")
print("Teammate's speech duration:", duration_y_teammate, "seconds")


# In[20]:


import librosa
import matplotlib.pyplot as plt

def plot_waveform(y, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


statement_file = r"C:\Users\Hp\Downloads\speech\statement.wav"
question_file = r"C:\Users\Hp\Downloads\speech\question.wav"   

y_statement, sr_statement = librosa.load(statement_file, sr=None)
y_question, sr_question = librosa.load(question_file, sr=None)

plot_waveform(y_statement, sr_statement, title='Statement Waveform')
plot_waveform(y_question, sr_question, title='Question Waveform')


# In[ ]:




