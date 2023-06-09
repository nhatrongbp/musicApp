# -- coding: utf-8
from flask import Flask, jsonify, request, send_file
import os

from KmeansModel import KmeansModel
from KnnModel import KnnModel
from RecommenderModel import *
from crudFeatureCsv import *

app = Flask(__name__)
app.config["UPLOAD_DIR"] = "Data/unlabeled/"
knn = KnnModel('MLmodels/minMaxScaler.pkl', 'MLmodels/pca_21.pkl', 'MLmodels/knn1.joblib')
km = KmeansModel('csvData/userFavouriteData.csv')


@app.route('/hello', methods=['GET'])
def helloworld():
    if request.method == 'GET':
        return send_file('Data\\tempSong\\temp.mp3')
    # data = {"data": "Hello World"}
    # return jsonify(data)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == 'POST':
        # file = request.files['file']
        # if file.filename.split('.')[-1] != 'wav' and file.filename.split('.')[-1] != 'mp3':
        #     data = {"error": "the song must be in mp3 or wav format"}
        #     return jsonify(data)
        # temp_path = 'Data\\tempSong\\temp.' + file.filename.split('.')[-1]
        # temp_path = app.config['UPLOAD_DIR'] + file.filename
        # file.save(temp_path)

        # LƯU Ý: cần truyền vào đường dẫn chính xác
        song_name = request.args.get('songName')
        temp_path = app.config['UPLOAD_DIR'] + song_name
        result = knn.predict_wav_file(temp_path)
        data = {"label": result}
        return jsonify(data)


@app.route("/recommend", methods=["GET"])
def recommend_songs_by_song():
    if request.method == 'GET':
        song_name = request.args.get('songName')
        n = int(request.args.get('minN'))
        print(song_name, n)
        rm = RecommenderModel('csvData/data.csv')
        # LƯU Ý: chỉ cần truyền vào tên file nhạc, không cần đường dẫn chính xác
        results = rm.find_similar_songs(song_name, min_n=n)
        data = {"data": {}}
        for index, value in enumerate(results):
            # print((index, value))
            data['data'].update({"song" + str(index): value})

        return jsonify(data)


@app.route("/recommendGenresByUserId", methods=["GET"])
def recommend_genres_by_user_id():
    if request.method == 'GET':
        user_id = int(request.args.get('userId'))
        number_of_genres = int(request.args.get('n'))
        if number_of_genres > 10 or number_of_genres < 1:
            number_of_genres = 2
        print(user_id)
        res = km.recommend_genre_of_user_by_id(user_id, number_of_genres)
        data = {"data": {}}
        for index, value in enumerate(res):
            data['data'].update({"genre" + str(index): value})
            print(value)
        return jsonify(data)


@app.route("/trainUserKmeans", methods=["GET"])
def train_user_kmeans():
    if request.method == "GET":
        km.train()
        km.predict()
        data = {"data": "trained successfully"}
        return jsonify(data)


@app.route("/remove", methods=["DELETE"])
def remove():
    if request.method == 'DELETE':
        song_name = request.args.get('songName')
        remove_song_features_from_data_csv(song_name)
        os.remove(app.config["UPLOAD_DIR"] + song_name)
        data = {"deleted": song_name}
        return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
