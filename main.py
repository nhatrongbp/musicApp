# This is a sample Python script.
from KnnModel import KnnModel
from RecommenderModel import RecommenderModel


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
    print("\n*******\nCosine similar songs to ", 'pop.00019.wav')
    # Trả về 1 list tên N file wav được gợi ý, dùng vòng for này để xem từng tên
    for r in results:
        print('name: ', r)

    # 2, Dự đoán nhãn (thể loại) của 1 file bài hát (wav hay mp3 đều được)
    km = KnnModel('MLmodels/minMaxScaler.pkl',
                  'MLmodels/pca_21.pkl',
                  'MLmodels/knn1.joblib')
    # LƯU Ý: cần truyền vào đường dẫn chính xác của file wav
    result = km.predict_wav_file('Data/unlabeled/blues.00000.wav')
    print('Data/unlabeled/blues.00000.wav is predicted as ', result)
    result = km.predict_wav_file('Data/unlabeled/blues.00001.wav')
    print('Data/unlabeled/blues.00001.wav is predicted as ', result)
    result = km.predict_wav_file('Data/unlabeled/newSong00000.mp3')
    print('Data/unlabeled/newSong00000.mp3 is predicted as ', result)
    result = km.predict_wav_file('Data/unlabeled/newSong00001.mp3')
    print('Data/unlabeled/newSong00001.mp3 is predicted as ', result)
    result = km.predict_wav_file('Data/unlabeled/newSong00002.mp3')
    print('Data/unlabeled/newSong00002.mp3 is predicted as ', result)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
