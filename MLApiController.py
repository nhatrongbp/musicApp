from flask import Flask, jsonify, request, send_file
import os

from KnnModel import KnnModel
from RecommenderModel import *
from crudFeatureCsv import *

app = Flask(__name__)
app.config["UPLOAD_DIR"] = "Data/unlabeled/"
km = KnnModel('MLmodels/minMaxScaler.pkl', 'MLmodels/pca_21.pkl', 'MLmodels/knn1.joblib')


@app.route('/hello', methods=['GET'])
def helloworld():
    if request.method == 'GET':
        return send_file('Data\\tempSong\\temp.mp3')
    # data = {"data": "Hello World"}
    # return jsonify(data)


# @app.route("/upload", methods=["GET", "POST"])
# def upload():
# 	if request.method == 'POST':
# 		file = request.files['file']
# 		if file.filename.split('.')[-1] != 'wav' and file.filename.split('.')[-1] != 'mp3':
# 			data = {"error": "the song must be in mp3 or wav format"}
# 			return jsonify(data)
# 		file.save(os.path.join(app.config['UPLOAD_DIR'], file.filename))
# 		data = {"data saved at": os.path.join(app.config['UPLOAD_DIR'], file.filename)}
# 		return jsonify(data)


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
        result = km.predict_wav_file(temp_path)
        data = {"label": result}
        return jsonify(data)


@app.route("/recommend", methods=["GET"])
def recommend():
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
