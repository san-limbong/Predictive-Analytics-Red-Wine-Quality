# Laporan Proyek Machine Learning - San Antonio Limbong
## Domain Proyek
Domain Poryek : **Ekonomi dan bisnis** , **Pertanian**
Judul : Predictive Analytics: Menentukan Kualitas Anggur

### Latar Belakang

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
-  Tes apa yang paling menentukan kualitas sebuah anggur? dan apa yang tidak terlalu berpengaruh terhadap kualitas anggur?
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
| 4.0  | 7.4           | 0.70             | 0.00        | 1.9            | 0.076     | 34.0                 | 0.9978  | 3.51 | 0.56      | 9.4     |         |


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
<!-- 

### EDA - Univariate Analysis

![Analisis Univariat (Data Kategori)](https://i.ibb.co/0MRrJCC/jumlah-kualitas-datasets.png)

Gambar 1a. Analisis Univariat (Data Kategori) 

![Univariate Analysis](https://i.ibb.co/V2mQ2dK/EDA-Univariate.png)

Gambar 1b. Analisis Univariat (Data Numerik) 

 Berdasarkan _Gambar 1a_ , dapat dilihat bahwa distribusi data katagorik _Quality_ yang terdiri dari _good_ dan _bad_ kualitas apel, yang mana nilai data **bad** terdiri dari `1928` dan **good** terdiri dari `1862`, yang mana menunjukan perbandingan data yang tidak terlalu jauh. Pada _Gambar 1b,_ untuk data numerik memiliki karakteristik, yaitu:
  - Dilihat dari distribusi data numerik _Size_, ukuran rata-rata buah berkisar dari -2 sampai 2, dan memiliki nilai rata-rata _Mean_ adalah -0.51.
  - Rata-rata berat apel bernilai -0.99 dan nilai _max_ berat apel adalah 3.08.
  - Rata-rata tingkat kemanisan apel -0.48.
  - Tekstur kerenyahan apel berkisar dari 0 hingga 2 yang mana nilai ini menunjukan rata-rata apel itu renyah.
  - Tingkat kesegaran buah dan Kematangan buat berada pada nilai 0.50 dan 0.53.
  - Rata-rata tingkat keasaman buah bernilai 0.06.

 Nilai-nilai ini menunjukkan bahwa data  telah dinormalisasi dengan cara _z-score normalization_ . _z-score normalization_  mengubah data dengan cara:
 - Mengurangi rata-rata (mean) dari setiap data point.
 - Membagi hasil pengurangan tersebut dengan standar deviasi data.
 

Pada kasus ini, rata-rata (mean) data "Size" adalah -0.51 dan standar deviasi data "Size" tidak diketahui. Namun, dengan nilai minimum -2 dan maksimum 2, dapat diasumsikan bahwa data "Size" telah diubah skalanya sehingga memiliki mean 0 dan standar deviasi 1. Data numerik lainnya, seperti _"Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", dan "Acidity"_, juga telah dinormalisasi dengan cara yang sama.


 

### EDA - Multivariate Analysis

![Multivariate Analysis](https://i.ibb.co/yNHmpNZ/EDA-MULTIVARIATE.png)


Gambar 2a. Analisis Multivariat

![Multivariate Analysis](https://i.ibb.co/WBQ5gPy/Matrix-corelasi.png)


Gambar 2b. Analisis Matriks Korelasi

Pada _Gambar 2a. Analisis Multivariat_, dengan menggunakan fungsi _pairplot_ dari _library seaborn_, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada pola sebaran data grafik pairplot, terterlihat bahwa _Size_ dan _Sweetness_ memiliki korelasi negatif menurun, yang mana semakin kecil ukuran buah rasa nya akan semakin manis.
Pada _Gambar 2b. Analisis Matriks Korelasi_, merupakan _Correlation Matrix_ menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur _Juiciness_ memiliki skor korelasi yang cukup besar `0.24` dengan fitur target _Acidity_ . -->


## Data Preparation
Pada proses _Data Preparation_ dilakukan kegiatan seperti _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
- Duplicate data (data yang serupa dengan data lainnya).
- Missing value (data atau informasi yang "hilang" atau tidak tersedia)
- Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Pada proses _Data Cleaning_ yang dilakukan adalah seperti:
- Converting Column Type (Mengubah tipe suatu kolom).
- Train Test Split (membagi data menjadi data latih dan data uji).
- Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding).

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| NaN | NaN | NaN | NaN |NaN | NaN| NaN	| Created_by_Nidula_Elgiriyewithana  | NaN |


Tabel 2. Melihat data missing value

Pada proyek kasus ini tidak ditemukannya data duplikat, tetapi ditemukannya _missing value_. Adapaun metode yang digunakan untuk mengatasi hal ini adalah dengan menerapkan _Dropping_ yaitu menghapus data yang _missing_ digunakannya metode ini dikarenakan jumlah missing value hanya berjumlah `1`. Lihat _Tabel 2. Melihat data missing value_. Adapun untuk _outlier_ juga dilakukan dengan metode _dropping_ menggunakan metode IQR.  IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

$$IQR = Q_3 - Q_1$$

- Q1 adalah kuartil pertama 
- Q3 adalah kuartil ketiga.

Setelah menggunakan metode IQR untuk menghilangkan _outlier_ pada dataset jumlah dataset menjadi `3790` yang awalnya adalah `4000`.
Pada proyek ini digunakan _Train Test Split_ pada library  *sklearn.model_selection* untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 20:80 dan random state sebesar 60. Pada proyek kasus ini digunakan _Normalization_ pada library _sklearn.preprocessing.MinMaxScaler_ untuk menormalisasi dataset. Semua proses ini diperlukan dalam rangka membuat model yang baik.
## Modeling

## Evaluation
