# -- coding: utf-8
# This is a sample Python script.
from KmeansModel import KmeansModel
from KnnModel import KnnModel
from RecommenderModel import RecommenderModel
from crudFeatureCsv import *


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # HƯỚNG DẪN CÁCH DÙNG MODEL CHO BACKEND

    # 1, GỢI Ý N BÀI HÁT TỪ BÀI pop00019.wav
    # đặc trưng được tính offline từ trước và lưu trong data.csv
    # bây giờ chỉ cần so sánh từng vector đặc trưng
    rm = RecommenderModel('csvData/data.csv')
    # LƯU Ý: chỉ cần truyền vào tên file wav, không cần đường dẫn chính xác
    results = rm.find_similar_songs('pop.00019.wav', min_n=5)
    print("\n*******\nSimilar songs to ", 'pop.00019.wav')
    # Trả về 1 list tên N file wav được gợi ý, dùng vòng for này để xem từng tên
    for r in results:
        print('name: ', r)

    # 2, Dự đoán nhãn (thể loại) của 1 file bài hát (wav hay mp3 đều được)
    # LƯU Ý: lưu file bài hát vào thư mục unlabeled trước rồi mới đoán được nhãn nhé
    knn = KnnModel('MLmodels/minMaxScaler.pkl',
                  'MLmodels/pca_21.pkl',
                  'MLmodels/knn1.joblib')
    # LƯU Ý: cần truyền vào đường dẫn chính xác của file wav
    result = knn.predict_wav_file('Data/unlabeled/newSong00000.mp3')
    print('Data/unlabeled/newSong00000.mp3 is predicted as ', result)
    result = knn.predict_wav_file('Data/unlabeled/newSong00001.mp3')
    print('Data/unlabeled/newSong00001.mp3 is predicted as ', result)
    result = knn.predict_wav_file('Data/unlabeled/newSong00002.mp3')
    print('Data/unlabeled/newSong00002.mp3 is predicted as ', result)

    # 3. Khi xóa 1 file bài hát, ngoài viêc xóa file wav (hoặc mp3) thì nhóc còn phải gọi
    # tới hàm sau để t xóa thông tin bài hát khỏi model nhé
    # remove_song_features_from_data_csv('Data/unlabeled/newSong00002.mp3')

    # 4. Train Kmeans phân cụm người dùng, 1 lần/ngày hoặc 1 lần/tần
    km = KmeansModel('csvData/userFavouriteData.csv')
    km.train()
    km.predict()

    # 5. Dự đoán thể loại mà người dùng thích dựa vào phân cụm người dùng
    # truyền vào id người dùng và số lượng thể loại mong muốn
    # trả về danh sách n thể loại mà cụm người dùng đó nghe nhiều nhất
    res = km.recommend_genre_of_user_by_id(1, 2)
    for r in res:
        print(r)
    res = km.recommend_genre_of_user_by_id(1000, 2)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
