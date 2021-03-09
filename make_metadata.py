import os
import pickle
import numpy as np

rootDir = 'assets/spmel'
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)


speakers, speakers_val = [], []
for speaker in sorted(subdirList):
    print('Processing speaker: %s' % speaker)
    utterances = []
    utterances.append(speaker)
    _, _, fileList = next(os.walk(os.path.join(dirName,speaker)))
    
    # use hardcoded onehot embeddings in order to be cosistent with the test speakers
    # modify as needed
    # may use generalized speaker embedding for zero-shot conversion
    idx = int(speaker[-2:])
    spkid = np.zeros((65,), dtype=np.float32)
    spkid[idx - 1] = 1.0

    #if speaker == 'p226':
    #    spkid[1] = 1.0
    #else:
    #    spkid[7] = 1.0
    utterances.append(spkid)
    
    # create train file list
    for fileName in sorted(fileList[:-1]):
        utterances.append(os.path.join(speaker,fileName))
    speakers.append(utterances)

    # create val file list
    utterances = [utterances[0]]
    for fileName in sorted(fileList[-1:]):
        utterances.append(os.path.join(speaker,fileName))
    speakers_val.append(utterances)
    
with open(os.path.join(rootDir, 'train.pkl'), 'wb') as handle:
    pickle.dump(speakers, handle)    

with open(os.path.join(rootDir, 'val.pkl'), 'wb') as handle:
    pickle.dump(speakers_val, handle)