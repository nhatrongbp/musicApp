# -- coding: utf-8
import csv
import random
import warnings

import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class KmeansModel:
    def __init__(self, user_history_data_path):
        self.filepath = user_history_data_path
        self.df = pd.read_csv(user_history_data_path)

    def train(self):
        self.generate_features()
        self.X = self.df.iloc[:, 1:11]  # 1t for rows and second for columns
        cols = self.X.columns
        min_max_scaler = preprocessing.MinMaxScaler()
        np_scaled = min_max_scaler.fit_transform(self.X)

        # new data frame with the new scaled data.
        self.X = pd.DataFrame(np_scaled, columns=cols)

        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(self.X)
        header = ''
        for i in range(1, 4):
            header += f' pca_{i}'
        header = header.split()
        self.X = pd.DataFrame(X_pca, columns=header)
        warnings.simplefilter('ignore')

        self.kmeans = KMeans(10)
        self.kmeans.fit(self.X)

    def predict(self):
        identified_clusters = self.kmeans.fit_predict(self.X)
        self.df.insert(loc=11, column='cluster', value=identified_clusters)
        self.df.to_csv(self.filepath, index=False)

    def recommend_genre_of_user_by_id(self, user_id, number_of_genres):
        # df1 = pd.read_csv(self.filepath)
        df3 = self.df.loc[self.df['userId'] == user_id]
        results = []
        print(self.df.shape, df3.shape)
        if df3.shape[0] == 0:
            print('This is a new user and his history does not have much to explore')
            return results
        user_cluster = df3.iloc[0]['cluster']
        print('This user belongs to cluster ', user_cluster)
        df2 = self.df.loc[self.df['cluster'] == user_cluster]
        sorted_views = df2.iloc[:, 1:11].sum().sort_values(ascending=False)
        for x, y in sorted_views.items():
            print(x, y)
        i = 0
        for x, y in sorted_views.items():
            results.append(x)
            print('This user may like the ', x, ' genre!')
            i += 1
            if i == number_of_genres:
                break
        return results

    def generate_features(self):
        header = 'userId blues classical country disco hiphop jazz metal pop reggae rock'
        header = header.split()
        file = open(self.filepath, 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        for i in range(1, 501):
            to_append = f'{i}'
            for j in range(0, 10):
                random_float = random.uniform(0.0, 1.0)
                if random_float < 0.4:
                    to_append += f' {random.randint(0, 10)}'
                elif random_float > 0.6:
                    to_append += f' {random.randint(80, 100)}'
                else:
                    to_append += f' {random.randint(10, 80)}'
            file = open(self.filepath, 'a', newline='')
            with file:
                writer = csv.writer(file)
                writer.writerow(to_append.split())
        self.df = pd.read_csv(self.filepath)
