### Modules / Components
* Dependency:
    - `torch, torchaudio`
* Audio loading
   - `torchaudio.load` 
* Audio augmentor
* Configuration
   - `torch.device`
* Audio pre-processing
   - `torch`
* Audio feature extraction
   - `torchaudio`

`pyspid_aug` = `[one_of[reverberate, additive_noise(music), ...], specaug(p=0.6) ]`
`pyspid_feat` = `[melspec(...), spectrogram]`
`pyspid_devices` = `[cpu]` / `[gpu0, gpu1]`

### Data Pipeline
```
pyspid_datapipeline(
    list_of_filenames (list of str), list_of_speaker_names (list of str),
    augmentations = pyspid_aug,
    features = pyspid_feat {concatenated channel-wise},
    devices = pyspid_devices,
    num_workers = 0,
    sampling_rate = 16000
) 
```

**N.B:** *A glorified torch data-loader, additional options to load everything
at once (for non-batch training models).*

### `pyspid_algorithms`
 * gmm_ubm
 * svm (pca) https://github.com/kazuto1011/svm-pytorch/blob/master/main.py
 * xvector
 * ivector 
      - http://people.csail.mit.edu/sshum/talks/ivector_tutorial_interspeech_27Aug2011.pdf
      - https://github.com/Anwarvic/Speaker-Recognition/tree/master/sidekit
       - https://projets-lium.univ-lemans.fr/sidekit/api/index.html
   - https://www.researchgate.net/publication/323178008_Fusing_discriminative_and_generative_methods_for_speaker_recognition_experiments_on_switchboard_and_NFITNO_field_data
          

**(NOT_IMPLEMENTED)**

 * use intermediate networks for embedding extraction. save the embeddings.
 * for models where batching is not possible or no `partial_fit`, create a list of model on all the segments of the data and make a mean ensemble model