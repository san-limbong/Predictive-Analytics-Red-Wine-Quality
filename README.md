# Laporan Proyek Machine Learning - San Antonio Limbong
## Domain Proyek
Domain Poryek : **Ekonomi dan bisnis** , **Pertanian**

Judul : Predictive Analytics: Menentukan Kualitas Anggur

### Latar Belakang
![dataset-cover](https://github.com/user-attachments/assets/964f75c0-a8ca-4de5-8b4a-d5d329ef881c)
Anggur kini menjadi minuman favorit banyak orang. Portugal, negara yang terkenal dengan anggurnya yang enak, terutama anggur vinho verde, semakin gencar mengekspor anggur mereka. Untuk menjaga kualitas anggur yang tinggi, para pembuat anggur menggunakan teknologi terbaru dan melakukan berbagai tes untuk memastikan anggur aman diminum.

Tes-tes ini memeriksa berbagai aspek seperti keasaman, kadar gula, dan kandungan zat-zat tertentu. Berikut adalah variabel input berdasarkan tes fisikokimia:

1. Fixed acidity
2. Volatile acidity
3. Citric acid
4. Residual sugar
5. Chlorides
6. Free sulfur dioxide
7. Total sulfur dioxide
8. Density
9. pH
10. Sulphates
11. Alcohol
12. Quality (skor antara 0 dan 10)

Meskipun kita bisa dengan mudah merasakan enak atau tidaknya suatu anggur, menghubungkan rasa ini dengan hasil tes kimia sangatlah sulit, seperti mencoba menerjemahkan lagu menjadi angka. Di sinilah machine learning berperan. Dengan machine learning, kita dapat menganalisis data hasil tes anggur dalam jumlah besar dan membuat model yang bisa memprediksi rasa anggur berdasarkan hasil tes tersebut. Model prediksi ini sangat berguna bagi lembaga sertifikasi untuk memastikan kualitas anggur lebih akurat, bagi produsen anggur untuk memperbaiki proses pembuatan agar sesuai dengan selera konsumen, dan bagi konsumen untuk memilih anggur yang sesuai dengan selera mereka.

[Publication](https://archive.ics.uci.edu/dataset/186/wine+quality) 


## Business Understanding
### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
-  Atribut mana yang paling berpengaruh terhadap kualitas sebuah anggur? dan atribut mana yang tidak terlalu berpengaruh terhadap kualitas anggur?
-  Bagaimana membuat model machine learning yang dapat menentukan berkualitas atau tidaknya sebuah anggur (Wine) berdasarkan data masukan uji fisikokimia?
-  Model yang seperti apa yang memiliki akurasi paling baik?

### Goals
Tujuan dari proyek ini adalah:
- Membandingkan keseluruhan tes yang digunakan dalam menentukan kualitas anggur.
- Membuat model machine learning yang dapat memprediksi kualitas anggur berdasarkan data masukan uji fisikokimia.
- Membandingkan beberapa algoritma untuk menentukan algoritma mana yang paling bagus.

### Solution Statements
- Melakukan uji korelasi untuk menentukan tes yang paling berpengaruh atau tidaknya terhadap suatu kualitas anggur.
- Mengembangkan model machine learning
- Memilih model machine learning dengan error rate paling rendah.

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | Red Wine Quality |
| Source | [Kaggle](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009) |
| License | [Database: Open Database, Contents: Database Contents](https://opendatacommons.org/licenses/dbcl/1-0/) |
| Visibility | Publik |
| Tags | _Earth and Nature, Education, Beginner, Alcohol_ |
| Usability | 8.82 |

Berikut informasi pada dataset: 

| A_id | Fixed Acidity | Volatile Acidity | Citric Acid | Residual Sugar | Chlorides | Total Sulfur Dioxide | Density | pH | Sulphates | Alcohol | Quality |
|------|---------------|------------------|-------------|----------------|-----------|----------------------|---------|----|-----------|---------|---------|
| 0.0  | 7.4           | 0.70             | 0.00        | 1.9            | 0.076     | 34.0                 | 0.9978  | 3.51 | 0.56      | 9.4     | 5       |
| 1.0  | 7.8           | 0.88             | 0.00        | 2.6            | 0.098     | 67.0                 | 0.9968  | 3.20 | 0.68      | 9.8     | 5       |
| 2.0  | 7.8           | 0.76             | 0.04        | 2.3            | 0.092     | 54.0                 | 0.9970  | 3.26 | 0.65      | 9.8     | 5       |
| 3.0  | 11.2          | 0.28             | 0.56        | 1.9            | 0.075     | 60.0                 | 0.9980  | 3.16 | 0.58      | 9.8     | 6       |


Tabel 1. EDA Deskripsi Variabel

Dilihat dari _Tabel 1. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 1599 sample dengan 12 fitur.
- Dataset memiliki 11 fitur bertipe float64 dan 1 fitur bertipe int.
- Terdapat 240 duplikat dalam dataset.

### Variable - variable pada dataset
1. Fixed acidity : most acids involved with wine or fixed or nonvolatile (do not evaporate readily)
2. Volatile acidity : the amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste
3. Citric acid : found in small quantities, citric acid can add 'freshness' and flavor to wines
4. Residual sugar : the amount of sugar remaining after fermentation stops, it's rare to find wines with less than 1 gram/liter and wines with greater than 45 grams/liter are considered sweet
5. Chlorides : the amount of salt in the wine
6. Free sulfur dioxide : the free form of SO2 exists in equilibrium between molecular SO2 (as a dissolved gas) and bisulfite ion; it prevents microbial growth and the oxidation of wine
7. Total sulfur dioxide : amount of free and bound forms of S02; in low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident in the nose and taste of wine
8. Density : the density of water is close to that of water depending on the percent alcohol and sugar content
9. pH : describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic); most wines are between 3-4 on the pH scale
10. Sulphates : a wine additive which can contribute to sulfur dioxide gas (S02) levels, wich acts as an antimicrobial and antioxidant
11. Alcohol : the percent alcohol content of the wine
12. Quality : output variable (based on sensory data, score between 0 and 10)


### EDA - Univariate Analysis
Gambar. Analisis Univariat
![univariat](https://github.com/user-attachments/assets/b537f1bb-d47f-47b7-8bdd-7f9509967914)

Berdasarkan gambar tersebut:
1. Fixed acidity memiliki mean atau rata rata sebesar 7,2
2. Volatile acidity memiliki mean atau rata rata sebesar 0.500 dan 0,600
3. Citric acid memiliki mean atau rata rata sebesar 0.00
4. Residual sugar memiliki mean atau rata rata sebesar 7,2
5. Chlorides memiliki mean atau rata rata sebesar 0,080
6. Free sulfur dioxide memiliki mean atau rata rata sebesar 6,0
7. Total sulfur dioxide memiliki mean atau rata rata sebesar 28,0
8. Density memiliki mean atau rata rata sebesar 0,99720
9. pH memiliki mean atau rata rata sebesar 3,36
10. Sulphates memiliki mean atau rata rata sebesar 0,54 ddan 0,58
11. Alcohol memiliki mean atau rata rata sebesar 9,5
12. Quality (skor antara 0 dan 10) memiliki mean atau rata rata sebesar 5

### EDA - Multivariate Analysis
- Gambar. Analisis Multivariat
![multivariat](https://github.com/user-attachments/assets/3a6691b7-0e8c-4ff5-aa16-0109d2b96199)

- Gambar. Analisis Matriks Korelasi
![correlation](https://github.com/user-attachments/assets/efc5f85a-a5c5-4323-af5e-f782ca752f8f)

Berdasarkan matriks korelasi :
Volatile acidity memiliki korelasi negatif, yang artinya jika semakin bagus kualitas dari wine tersebut (mendekati 10), maka tingkat volatile aciditynya akan semakin rendah. Dengan kata lain masih memiliki hubungan meskipun tegak lurus. Disisi lain perhatikan korelasi yang dimiliki atribut free sulfur dioxodie. Yang menunjukkan hampir memiliki korelasi yang kecil dengan sebagian besar atribut. Maka dari itu akan kita lakukan feature selection dengan atribut ini



## Data Preparation

Teknik yang digunakan dalam penyiapan data (Data Preparation) yaitu:
- Penanganan Duplicate Values. Pada kasus dataset ini ada beberapa kolom dengan duplicate values dan ditangani dengan melakukan drop untuk menghilangkan redudansi data.
- Mendeteksi outliers. Penanganan dilakukan menggunakan IQR (InterQuartile Range) untuk mendeteksi outliers. IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) selanjutnya nilai apa pun yang berada di luar batas ini dianggap sebagai outlier.
- Melakukan drop terhadap fitur berdasarkan temuan pada matriks korelasi.
- Split Data atau pembagian dataset menjadi data latih dan data uji menggunakan bantuan train_test_split. Pembagian dataset ini bertujuan agar nantinya dapat digunakan untuk melatih dan mengevaluasi kinerja model. Pada proyek ini, 80% dataset digunakan untuk melatih model, dan 20% sisanya digunakan untuk mengevaluasi model.


## Modelling
Berikut adalah keterangan dari algoritma-algoritma yang digunakan:

** 1. K-Nearest Neighbors (KNN)**

``` KNN adalah algoritma yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan mencari sejumlah tetangga terdekat (K) dari suatu titik data baru dan memprediksi nilai berdasarkan mayoritas nilai dari tetangga-tetangga tersebut. Algoritma ini mudah dipahami dan diimplementasikan namun bisa lambat untuk dataset besar karena harus menghitung jarak ke semua titik data.```

Parameter yang digunakan :
- `n_neighbors` = jumlah tetangga terdekat.

** 2. Random Forest (RF)**

``` Random Forest adalah ensemble learning method yang menggunakan banyak pohon keputusan (decision trees) untuk klasifikasi atau regresi. Algoritma ini bekerja dengan membuat banyak pohon keputusan dari subset data dan kemudian menggabungkan hasilnya. Ini membantu meningkatkan akurasi dan mengurangi overfitting.```

Parameter yang digunakan :
- `n_estimators` = Jumlah maksimum estimator di mana boosting dihentikan.
- `max_depth` = Kedalaman maksimum setiap tree.
- `random_state` = Pengambilan sampel secara acak
-  `n_jobs=` jumlah job (pekerjaan) yang digunakan secara paralel

** 3. Adaboost **

``` Adaboost adalah algoritma boosting yang menggabungkan beberapa model lemah (weak learners), seperti pohon keputusan yang pendek (stumps), untuk membentuk model yang kuat. Setiap model lemah dibentuk berdasarkan kesalahan dari model sebelumnya, dengan memberi bobot lebih pada data yang salah diprediksi. Hasil akhirnya adalah model yang lebih akurat.```

Parameter yang digunakan :
- `learning_rate` = Hyperparameter training yang digunakan untuk menghitung nilai koreksi bobot padad waktu proses training. Umumnya nilai learning rate berkisar antara 0 hingga 1
- `random_state` = Pengambilan sampel secara acak

** 4. Gradient Boosting **

``` Gradient Boosting adalah metode ensemble yang mirip dengan Adaboost, tetapi menggunakan pendekatan gradient descent untuk mengurangi kesalahan pada setiap iterasi. Model ini menambahkan model baru untuk memperbaiki residu dari model sebelumnya.```

Parameter yang digunakan :
- `default`

** 5. Support Vector Regression (SVR) **

``` SVR adalah varian dari Support Vector Machines (SVM) yang digunakan untuk tugas regresi. SVR mencoba menemukan hyperplane yang memaksimalkan margin antara data dan hyperplane dengan mempertimbangkan toleransi kesalahan tertentu.```

Parameter yang digunakan :
- `default`

6. Hyperparameter Tuning yang digunakan pada proyek ini :



| Model    | Best Parameters                                                  |
|----------|------------------------------------------------------------------|
| KNN      | `n_neighbors=10`                                                 |
| RF       | `n_estimators=50, max_depth=16, random_state=55, n_jobs=-1`      |
| Adaboost | `learning_rate=0.05, random_state=55`                            |


## Evaluasi
Metrik yang akan kita gunakan pada prediksi ini adalah MSE atau Mean Squared Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut.

![mse](https://github.com/user-attachments/assets/759e96bb-886c-419e-b612-b6047e5d9c61)

```
Keterangan:
N = jumlah dataset
yi = nilai sebenarnya
y_pred = nilai prediksi
```

Berikut model yang digunakan dan perbandingan error rate terendah menggunakan mse secara berurutan.

![evaluasi](https://github.com/user-attachments/assets/730b0be9-3266-4ffa-b0f7-460e2c9a4406)

Berdasarkan gambar tersebut, maka algoritma yang dipilih untuk menjawab tantangan dalam proyek adalah random forest.

## Kesimpulan
1. Atribut yang paling berpengaruh terhadap kualitas sebuah anggur adalah `alkohol` dengan persentase sebesar 0,49. Sementara atribut yang tidak berpengaruh adalah `free sulfur dioxide` dengan persentase -0.003609.
2. Pembangunan model yang baik dilakukan dengan melakukan cleaning terhadap data baik berupa missing value, duplicate, ataupun outlier. Selanjutnya dengan data yang bersih dapat dilanjutkan ketahapan modelling seperti yang tersedia pada notebook.
3. Model yang terbaik ditinjau dengan menggunakan error rate terendah dengan kasus regresi. Berdasarkan pengerjaan yang dilakuakn model terbaik ditemukan pada algoritma random forest. 
