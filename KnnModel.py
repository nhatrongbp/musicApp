import numpy as np
import pickle as pk
import librosa
import joblib
import warnings
from crudFeatureCsv import *


class KnnModel:
    def __init__(self, scaler_path, pca_path, model_path):
        self.loaded_scaler = pk.load(open(scaler_path, 'rb'))
        self.loaded_pca = pk.load(open(pca_path, 'rb'))
        self.loaded_knn = joblib.load(model_path)

    def predict_wav_file(self, filepath):
        y, sr = librosa.load(filepath, mono=True, duration=3)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        y_harm, y_perc = librosa.effects.hpss(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        pandas_data = [[np.mean(chroma_stft), np.var(chroma_stft), np.mean(rms),
                        np.var(rms), np.mean(spec_cent), np.var(spec_cent),
                        np.mean(spec_bw), np.var(spec_bw), np.mean(rolloff), np.var(rolloff),
                        np.mean(zcr), np.var(zcr), np.mean(y_harm), np.var(y_harm),
                        np.mean(y_perc), np.var(y_perc), tempo]]
        for e in mfcc:
            pandas_data[0].append(np.mean(e))
            pandas_data[0].append(np.var(e))
        df = pd.DataFrame(pandas_data)

        warnings.simplefilter('ignore')
        df_scaled = self.loaded_scaler.transform(df)
        df = pd.DataFrame(df_scaled)
        df_pca = self.loaded_pca.transform(df)
        df = pd.DataFrame(df_pca)
        prediction = self.loaded_knn.predict(df)
        # print(prediction)
        add_song_features_to_data_csv(filepath, pandas_data, prediction[0])
        return prediction[0]
