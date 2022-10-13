# Laporan Proyek 2 Machine Learning - Rachma Yuni Andari
## Sistem Rekomendasi Buku CBF


# Domain Proyek
Semakin berkembangnya internet yang menawarkan banyak kemudahan, maka semakin kompleks dan detail juga arus eksekusi teknologi yang diolah di dalamnya. Sistem rekomendasi pada website menjadi bagian dari salah satu dari eksekusi teknologi agar dapat meningkatkan ke dinamisan dan keefektifan website untuk pengguna booklover / pecinta buku. Banyak situs rekomendasi penjualan buku yang tersedia di internet tapi umumnya buku yang direkomendasikan sebagian besar website tidak menarik minat users karena tidak sesuai dengan minat / preferensi pengguna. 


# Business Understanding


## Problem Statements
Dari beberapa algoritma dari sistem rekomendasi, bagaimana penerapan metode ***content based filtering.*** yang digunakan pada proyek Sistem Rekomendasi Buku ini?

## Goals
Mempermudah users dalam menemukan buku yang sesuai preferensi / minatnya. Proyek ini menyajikan bagaimana pendekatan model algoritma machine learning untuk merekomendasikan buku yang relevan pada user.

## Solutions Statements
Goal dari sistem rekomendasi jenis ini adalah menghasilkan preferensi pengguna berdasarkan rating yang telah diberikan sebelumnya. Dari data rating pengguna, model akan mengidentifikasi buku-buku yang mirip untuk direkomendasikan. 


# Data Understanding
![dataset](https://user-images.githubusercontent.com/107310486/195502039-d65d1db0-6366-48aa-b6ac-dde0c9a1cab6.png)


Pertama, import semua library yang dibutuhkan. Data yang digunakan dalam proyek ini adalah **# Book-Crossing: User review ratings** yang diunduh dari [Kaggle](https://www.kaggle.com/datasets/ruchi798/bookcrossing-dataset). Pada proyek kali ini, tidak hanya satu dataset yang digunakan. Ada tambahan dari dataset file yang terpisah. Jadi total dataset yang digunakan di proyek ini ada 2, yaitu *data_book.csv* dan *ratings_book.csv*. Dataset ini memiliki total 1juta 31ribu lebih rows

Di sini kita mengunggah data ke Google Drive, agar tidak berkali-kali upload.
Lalu simpan dataset buku pada variable **data** dan dataset ratings pada variabel **ratings**.

Terdapat 19 kolom pada dataset data_book, yaitu: *Unnamed: 0, user_id, location, age, isbn, rating, book_title, book_author, year_of_publication, publisher, img_s/img_m/img_l, summary, language, category, city, state, country.*

Terdapat 3 kolom pada dataset ratings_book, yaitu: *User-ID, ISBN, Book-Rating*

## Exploratory Data Analysis - Deskripsi Variabel
Tahap ini adalah tahap menganalisis karakteristik, menemukan pola, anomaly, dan memeriksa asumsi data.

### Jenis Variabel dalam Dataset data_book.csv
-   **user_id**: merupakan nomor id dari user
-   **location**: merupakan domisili user
-   **age**: merupakan umur user
-   **isbn**: merupakan international standard book number (unik)
-   **rating**: merupakan nilai yang diberikan user pada buku
-   **book_title**: merupakan judul buku
-   **book_author**: merupakan nama penulis dari buku
-   **year_of_publication**: merupakan tahun penerbitan buku
-   **publisher**: merupakan penerbit buku
-   **img_s**: merupakan gambar cover dari buku
-   **img_m**: merupakan gambar cover dari buku
-   **img_l**: merupakan gambar cover dari buku
-   **Summary**: merupakan garis besar paragraf yang menggambarkan buku
-   **Language**: merupakan bahasa buku
-   **Category**: merupakan  kategori buku
-   **city**: merupakan kota di mana buku tersebut dibeli
-   **state**: merupakan provinsi di mana buku tersebut dibeli
- **country**:merupakan negara di mana buku tersebut dibeli

### Jenis Variabel dalam Dataset ratings_book.csv
-   **user_id**: merupakan nomor id dari user
-   **User-ID**: merupakan nomor id dari user yang memberi rating
-   **ISBN**: merupakan international standard book number (unik)
-   **Book-Rating**: merupakan nilai/rating yang diberikan pada buku


### Fitur Tidak Berguna (Redundant)

Fitur *'Unnamed: 0', 'user_id', 'location', 'age', 'year_of_publication', 'img_s', 'img_l', 'img_m', 'Summary', 'Language'* tidak terlalu berguna karena kolom tersebut tidak memberikan informasi tentang buku dan tidak terlalu berpengaruh terhadap fitur yang lain dalam konteks model yang akan dibuat.

Setelah itu adalah menghapus duplikat kolom judul buku yang ada pada dataset pertama dan isbn pada dataset kedua.
Data ya
| book_title | book_author | publisher | category | 
|---|--------|----------------|-------|
| As Hogan Said . . . : The 389 Best Things Anyone Said about How to Play Golf  |    Randy Voorhees | Simon & Schuster | ['Humor'] | 
| All Elevations Unknown: An Adventure in the Heart of Borneo |    Sam Lightner | Broadway Books |  ['Nature'] | 
|The Are You Being Served? Stories: 'Camping In' and Other Fiascoes |    Jeremy Lloyd | Kqed Books |   ['Fiction'] | 

Category yang telah bersih akan disimpan dalam kolom baru bernama kategori_bersih.

| book_title | book_author | publisher | category | kategori_bersih |
|---|--------|----------------|-------|-------|
| As Hogan Said . . . : The 389 Best Things Anyone Said about How to Play Golf  |    Randy Voorhees | Simon & Schuster | ['Humor'] | Humor
| All Elevations Unknown: An Adventure in the Heart of Borneo |    Sam Lightner | Broadway Books |  ['Nature'] | Nature
|The Are You Being Served? Stories: 'Camping In' and Other Fiascoes |    Jeremy Lloyd | Kqed Books |   ['Fiction'] | Fiction


Dari output terlihat bahwa pada dataset buku:
- Terdapat 14 kolom kategori dengan tipe object
- Terdapat 5 kolom numerik dengan tipe int dan float


## Exploratory Data Analysis - Cek Missing Value
Pada dataset yang pertama, terdapat 3 kolom yang memiliki missing value, yaitu *'country', 'city', 'state'*. Dan bersamaan bahwa ketiga kolom ini tidak relevan / tidak penting dalam pertimbangan variabel proyek ini. Kemudian, pada dataset rating tidak ditemukan missing value.


## Exploratory Data Analysis - Merging Dataset

Pada tahap ini, terjadi penyatuan dataset *data_book.csv* dan *ratings_book.csv* pada kolom isbn.


# Data Preparation

Pada bagian ini, akan dilakukan empat tahap persiapan data, yaitu:
- Menghapus fitur yang tidak perlu
- Menghapus duplikat kolom
- Menggabungkan dataset

**Menghapus fitur yang tidak diperlukan**
Pada tahap ini, kita melakukan penghapusan semua kolom kecuali kolom-kolom yang penting, seperti: judul buku, isbn, author buku, publisher, dan kategori. menggunakan perintah **drop**.

**Menghapus duplikat kolom**
Pada tahap ini, variabel data_bersih2 yang telah diproses dengan **drop_duplicates** sehingga menghasilkan data yang bersih, disimpan di variabel data_bersih3.

**Menggabungkan dataset*
Pada tahap ini, variabel data_bersih2 yang telah diproses dengan **drop_duplicates** sehingga menghasilkan data yang bersih, disimpan di variabel data_bersih3.

## Modeling
Dalam proyek ini digunakan satu algoritma untuk mengembangkan model sistem rekomendasi. Model yang digunakan adalah *content based filtering* memakai **TfidVectorizer**.

**Content Based Filtering**
Content based filtering ini merekomendasikan item yang mirip dengan preferensi pengguna yang telah dibaca sebelumnya berdasarkan isi ringkasan buku dari catatan sejarah masa lalu pembeli.
Model dari metode ini juga membentuk untuk menggabungkan pendapat user yang lain melalui rating yang diberikan untuk membuat prediksi yang akurat / dipersonalisasi. 
Sehingga, metode ini mengarah pada hasil rekomendasi buku yang baik kepada pengguna berdasarkan preferensi mereka.

Digunakan **cosine_similarity** untuk membandingkan kedua dokumen atau menemukan kesamaan di antara kedua komponen matrix.
Konsep kerja dari cosine_similarity adalah menemukan kesamaan kosinus. Nilainya antara -1 dan 1. Jika nilainya 1 atau mendekati 1 maka kedua dokumen tersebut sama dan sebaliknya.
![190915969-92ac61ae-b1ac-44d9-9ec1-92f778c19602](https://user-images.githubusercontent.com/107310486/195525575-f9db3136-3e4f-4c1f-8300-03a3522d767c.png)


Pada pendefinisian fungsi rekomendasi, langkah yang diimplementasikan:

- Dapatkan indeks buku yang diberikan judulnya
- Dapatkan daftar skor cosinus imilarity
- Ubah menjadi menjadi daftar tupel di mana elemen pertama adalah posisi dan elemen kedua adalah skor kesamaan


## Result
Fungsi get_recommendations dibuat untuk menemukan rekomendasi buku menggunakan similarity yang telah sebelumnya didefinisikan.

Hasil dari proyek ini, apabila preferensi kita adalah kategori "Science", dengan kata kunci judul buku "The Dragons of Eden: Speculations on the Evolution of Human Intelligence", maka yang direkomendasikan selanjutnya adalah judul yang ber-kategori "Science" pula.

Sistem telah berhasil merekomendasikan top 9 buku yang mirip dengan 'Their First Time in the Movies (With DVD & VHS)', Jadi, jika pengguna menyukai buku kategori **'Peforming Arts'**, maka sistem dapat merekomendasikan kategori 'Peforming Arts' lainnya.

![peforming arts](https://user-images.githubusercontent.com/107310486/195537330-25600c63-b025-4781-a44b-b3343e5fb6fb.png)


## Evaluation

Selisih nilai sebenarnya dengan nilai prediksi disebut **error.**
Metrik adalah yang mengukur seberapa kacil error tersebut.
Dalam beberapa proyek, beberapa metrik digunakan sebagai ukuran kinerja.
Metrik yang digunakan pada prediksi ini adalah *precision*. 

$$ Precision = {True Positive \over  {TruePositive} + FalsePositive} $$

Dalam konteks ini, kita menggunakan rating untuk penentuan dalam penentuan rekomendasi dengan formula antara lain, sebagai berikut.

$$ Precision@K = {(Rekomendasi Item Relevan) \over  {(Rekomendasi Item)}} $$

Relevan di sana mengacu pada ketentuan sebagai berikut:

- Apabila rating > 5 disebut sebagai relevan
- Apabila rating < 5 disebut sebagai tidak relevan



## References
[1] A. S. Tewari, A. Kumar and A. G. Barman, "Book recommendation system based on combine features of content based filtering, collaborative filtering and association rule mining," 2014 IEEE International Advance Computing Conference (IACC), 2014, pp. 500-503, doi: 10.1109/IAdCC.2014.6779375.

[2] P. Mathew, B. Kuriakose and V. Hegde, "Book Recommendation System through content based and collaborative filtering method," 2016 International Conference on Data Mining and Advanced Computing (SAPIENCE), 2016, pp. 47-52, doi: 10.1109/SAPIENCE.2016.7684166.

[3] [Online]. Available: https://www.dicoding.com/academies/319/tutorials/17116. [Accessed: 11- Oct- 2022]

[4] Naaz, S. (2018). Machine Learning Algorithm to Predict Survivability in Breast Cancer Patients. _International Journal on Computer Science and Engineering_, _10_(4), 97-101.
