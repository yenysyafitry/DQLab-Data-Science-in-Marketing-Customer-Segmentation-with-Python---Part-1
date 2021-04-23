Salah satunya teknik yang dapat digunakan adalah pengaplikasian machine learning menggunakan algoritma K-Prototypes.  Algoritma K-Prototypes merupakan gabungan dari K-Means dan juga K-Modes yang dapat di gunakan untuk melakukan segmentasi dengan data.

### Mempersiapkan Library
Masalah ini akan dapat di selesaikan dengan menggunakan bantuan library - library di bawah ini:
Pandas, di gunakan untuk melakukan pemrosesan analisis data
Matplotlib, di gunakan sebagai dasar untuk melakukan visualisasi data
Seaborn, di gunakan di atas matplotlib untuk melakukan data visualisasi yang lebih menarik
Scikit - Learn, digunakan untuk mempersiapkan data sebelum dilakukan permodelan
kmodes, digunakan untuk melakukan permodelan menggunakan algoritma K-Modes dan K-Prototypes.
Pickle, digunakan untuk melakukan penyimpanan dari model yang akan di buat.

sou :
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import pickle
from pathlib import Path


### Membaca Data Pelanggan
Langkah pertama yang perlu di lakukan adalah membaca data tersebut yang semula adalah textfile menjadi pandas dataframe.

Tugas:
Kamu akan menggunakan fungsi read_csv yang ada di pandas untuk memasukkan data dan kemudian menampilkan 5 data teratas.

sou:
# import dataset
df = pd.read_csv("https://dqlab-dataset.s3-ap-southeast-1.amazonaws.com/customer_segments.txt", sep="\t")
# menampilkan data
print (df.head())

out: 
Customer_ID       Nama Pelanggan Jenis Kelamin  Umur       Profesi  \
0    CUST-001         Budi Anggara          Pria    58    Wiraswasta   
1    CUST-002     Shirley Ratuwati        Wanita    14       Pelajar   
2    CUST-003         Agus Cahyono          Pria    48  Professional   
3    CUST-004     Antonius Winarta          Pria    53  Professional   
4    CUST-005  Ibu Sri Wahyuni, IR        Wanita    41    Wiraswasta   

  Tipe Residen  NilaiBelanjaSetahun  
0       Sector              9497927  
1      Cluster              2722700  
2      Cluster              5286429  
3      Cluster              5204498  
4      Cluster             10615206  


### Melihat Informasi dari Data
Selanjutnya kamu perlu melihat informasi dari data yang ada. Sehingga dengan kamu bisa mengetahui jumlah baris dan kolom, nama kolom, identifikasi null values,  dan juga mengetahui tipe data dengan mudah.

Tugas:

Gunakan fungsi info() dari pandas untuk melihat informasi dari data kita. Jika kamu melakukan dengan benar, kamu akan mendapatkan hasil sebagai berikut:

sou:
# Menampilkan informasi data  
df.info()

out :download (1).png

### Kesimpulan
Setelah melakukan pemanggilan data dan melihat informasi data yang kamu miliki, kamu akhirnya mengetahui bahwa:

Data yang akan digunakan terdiri dari 50 baris dan 7 kolom
Tidak ada nilai Null padat data
Dua kolom memiliki tipe data numeric dan lima data bertipe string
Tips:

Dalam setiap project machine learning, kita harus memahami informasi dasar dari data yang kita miliki sebelum melakukan analisa lebih lanjut. Dengan melakukan hal ini, kita bisa memastikan tipe data dari masing-masing kolom sudah benar, mengetahui apakah ada data null di tiap tiap kolom, dan juga mengetahui nama-nama kolom di dataset yang kita gunakan. Informasi ini nantinya akan menentukan proses apa yang perlu kita lakukan selanjutnya.

### Eksplorasi Data Kategorikal
Selain data numerikal, kamu juga perlu melihat bagaimana persebaran data pada kolom-kolom yang memiliki jenis kategorikal yaitu Jenis Kelamin, Profesi dan Tipe Residen. Kamu dapat melakukan hal ini dengan menggunakan countplot dari library seaborn.

Tugas:
Buatlah countplot dengan menggunakan seaborn untuk kolom-kolom yang berjenis kategorikal.

sou:
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
plt.clf() 
# Menyiapkan kolom kategorikal  
kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']  
# Membuat canvas
fig, axs = plt.subplots(3,1,figsize=(7,10)) 
# Membuat plot untuk setiap kolom kategorikal  
for i, kol in enumerate(kolom_kategorikal):  
    # Membuat Plot  
    sns.countplot(df[kol], order = df[kol].value_counts().index, ax = axs[i])  
    axs[i].set_title('\nCount Plot %s\n'%(kol), fontsize=15)  
      
    # Memberikan anotasi  
    for p in axs[i].patches:  
        axs[i].annotate(format(p.get_height(), '.0f'),  
                        (p.get_x() + p.get_width() / 2., p.get_height()),  
                        ha = 'center',  
                        va = 'center',  
                        xytext = (0, 10),  
                        textcoords = 'offset points')  
    # Setting Plot  
    sns.despine(right=True,top = True, left = True)  
    axs[i].axes.yaxis.set_visible(False) 
    plt.setp(ax)
    plt.tight_layout()
# Tampilkan plot
plt.show()



out: 
download (2).png
download (3).png
download (4).png

### Kesimpulan
Dari hasil explorasi data tersebut kamu dapat mendapatkan informasi:

Rata-rata dari umur pelanggan adalah 37.5 tahun
Rata-rata dari nilai belanja setahun pelanggan adalah 7,069,874.82
Jenis kelamin pelanggan di dominasi oleh wanita sebanyak 41 orang (82%) dan laki-laki sebanyak 9 orang (18%)
Profesi terbanyak adalah Wiraswasta (40%) diikuti dengan Professional (36%) dan lainnya sebanyak (24%)
Dari seluruh pelanggan 64% dari mereka tinggal di cluster dan 36% nya tinggal di sektor
Tips:

Kita dapat mengenal data kita lebih jauh lagi pada tahapan eksplorasi data ini. Proses eksplorasi data bisa berupa univariate maupun multivariate data eksplorasi. Eksplorasi Data Univariate melihat karakteristik tiap-tiap feature, misal nya dengan melihat statistik deskriptif, membuat histogram, kdeplot, count plot maupun boxplot. Sedangkat untuk Eksplorasi Data Multivariate, kita melihat hubungan tiap variabel dengan variabel lainnya, misal kan dengan membuat korelasi matrix, melihat predictive power, cross tabulasi, dan lainnya.


### Standarisasi Kolom Numerik

sou :
from sklearn.preprocessing import StandardScaler    
kolom_numerik  = ['Umur','NilaiBelanjaSetahun']    
# Statistik sebelum Standardisasi  
print('Statistik Sebelum Standardisasi\n')  
print(df[kolom_numerik ].describe().round(1))    
# Standardisasi  
df_std = StandardScaler().fit_transform(df[kolom_numerik])    
# Membuat DataFrame  
df_std = pd.DataFrame(data=df_std, index=df.index, columns=df[kolom_numerik].columns)    
# Menampilkan contoh isi data dan summary statistic  
print('Contoh hasil standardisasi\n')  
print(df_std.head())    
print('Statistik hasil standardisasi\n')  
print(df_std.describe().round(0)) 

out :
Statistik Sebelum Standardisasi

       Umur  NilaiBelanjaSetahun
count  50.0                 50.0
mean   37.5            7069874.8
std    14.7            2590619.0
min    14.0            2722700.0
25%    25.0            5257529.8
50%    35.0            5980077.0
75%    49.8            9739615.0
max    64.0           10884508.0
Contoh hasil standardisasi

       Umur  NilaiBelanjaSetahun
0  1.411245             0.946763
1 -1.617768            -1.695081
2  0.722833            -0.695414
3  1.067039            -0.727361
4  0.240944             1.382421
Statistik hasil standardisasi

       Umur  NilaiBelanjaSetahun
count  50.0                 50.0
mean    0.0                 -0.0
std     1.0                  1.0
min    -2.0                 -2.0
25%    -1.0                 -1.0
50%    -0.0                 -0.0
75%     1.0                  1.0
max     2.0                  1.0


### Konversi Kategorikal Data dengan Label Encoder
Selanjutnya kamu perlu merubah kolom-kolom yang berjenis kategorikal menjadi angka. Kita akan menggunakan salah satu fungsi dari sklearn yaitu LabelEncoder. Pada dasarnya fungsi ini akan melakukan konversi data pelanggan dari teks menjadi numerik.

Sebagai contoh untuk kolom Jenis Kelamin, teks "Pria" akan dirubah menjadi angka 0 dan teks "Wanita" akan di rubah menjadi angka satu. Perubahan ini perlu kita untuk semua teks sebelum di gunakan pada algoritma K-Prototype.

Tugas:

Ubahlah kolom-kolom kategorikal pada data set kamu menjadi numerik menggunakan LabelEncoder dari sklearn. Kemudian tampilkan hasil lima teratas nya.

sou :
from sklearn.preprocessing import LabelEncoder    
# Inisiasi nama kolom kategorikal  
kolom_kategorikal = ['Jenis Kelamin','Profesi','Tipe Residen']    
# Membuat salinan data frame  
df_encode = df[kolom_kategorikal].copy()      
# Melakukan labelEncoder untuk semua kolom kategorikal  
for col in kolom_kategorikal:  
    df_encode[col] = LabelEncoder().fit_transform(df_encode[col])        
# Menampilkan data  
print(df_encode.head()) 


out:
Jenis Kelamin  Profesi  Tipe Residen
0              0        4             1
1              1        2             0
2              0        3             0
3              0        3             0
4              1        4             0



### Menggabungkan Data untuk Permodelan
Setelah menyelesaikan dua tahap sebelumnya, kali ini kamu akan menggabungkan kedua hasil pemrosesan tersebut menjadi satu data frame. Data frame ini yang akan di gunakan untuk permodelan.

Tugas:
Gabungkan kedua data frame df_std dan df_encode yang sudah di buat di tahap sebelumnya menjadi df_model.

sou :
# Menggabungkan data frame
df_model = df_encode.merge(df_std, left_index = True, right_index=True, how = 'left')
print (df_model.head())

out :
    Jenis Kelamin  Profesi  Tipe Residen      Umur      NilaiBelanjaSetahun
0              0        4             1    1.411245             0.946763
1              1        2             0   -1.617768            -1.695081
2              0        3             0   0.722833            -0.695414
3              0        3             0   1.067039            -0.727361
4              1        4             0   0.240944             1.382421
