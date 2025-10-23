import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

def preprocess_data(input_path):
    """
    Fungsi ini memuat data mentah dari input_path,
    melakukan semua langkah preprocessing,
    dan mengembalikan DataFrame yang bersih.
    """
    
    print(f"Memuat data dari {input_path}...")
    # 1. Load Data
    try:
        df = pd.read_excel(input_path, sheet_name='Full_new') 
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return None

    print("Memulai preprocessing data...")

    # 2. Membersihkan spasi ekstra di nama kolom (Langkah ini saya majukan)
    print("Membersihkan nama kolom...")
    df.columns = [col.strip() for col in df.columns]
    
    # 3. Konversi Tipe Data
    print("Mengonversi tipe data ke numerik...")
    df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors='coerce')
    df["II    beta-HCG(mIU/mL)"] = pd.to_numeric(df["II    beta-HCG(mIU/mL)"], errors='coerce')
    
    # 4. Penanganan Missing Values (Imputasi)
    print("Mengisi missing values...")
    
    # Mengisi dengan Median
    df['Marraige Status (Yrs)'] = df['Marraige Status (Yrs)'].fillna(df['Marraige Status (Yrs)'].median())
    df['II    beta-HCG(mIU/mL)'] = df['II    beta-HCG(mIU/mL)'].fillna(df['II    beta-HCG(mIU/mL)'].median())
    df['AMH(ng/mL)'] = df['AMH(ng/mL)'].fillna(df['AMH(ng/mL)'].median())
    
    # Mengisi dengan Modus (karena 'Fast food' adalah kategorikal)
    df['Fast food (Y/N)'] = df['Fast food (Y/N)'].fillna(df['Fast food (Y/N)'].mode()[0])

    # 5. Penanganan Baris Duplikat 
    print("Mengecek dan menghapus baris duplikat...")
    duplicate_rows = df.duplicated().sum()
    print(f"Jumlah baris duplikat ditemukan: {duplicate_rows}")

    # Hapus baris duplikat jika ada
    if duplicate_rows > 0:
        df.drop_duplicates(inplace=True)
        print("Baris duplikat telah dihapus.")
    
    print(f"Jumlah baris setelah penghapusan duplikat: {len(df)}")
    
    print("Preprocessing selesai.")
    
    # 6. Feature Scaling
    print("Melakukan feature scaling (MinMaxScaler)...")
    
    # Pisahkan fitur (X) dan target (y)
    X = df.drop('PCOS (Y/N)', axis=1)
    y = df['PCOS (Y/N)']

    # Identifikasi kolom numerik (tidak termasuk target)
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()

    # Inisialisasi dan terapkan MinMaxScaler
    scaler = MinMaxScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print("Feature scaling selesai.")
    
    # Gabungkan kembali X (yang sudah discaling) dan y
    processed_df = pd.concat([X, y], axis=1)

    # 7. Feature Selection 
    # berdasarkan EDA heatmap korelasi
    print("Melakukan feature selection...")
    selected_features = [
        'PCOS (Y/N)',
        'Follicle No. (R)',
        'Follicle No. (L)',
        'Skin darkening (Y/N)',
        'hair growth(Y/N)',
        'Weight gain(Y/N)',
        'Cycle(R/I)',
        'Fast food (Y/N)',
        'Cycle length(days)', 
        'Age (yrs)', 
        'Marraige Status (Yrs)' 
    ]
    
    df = processed_df[selected_features].copy()

    # 8. Kembalikan data yang sudah bersih
    return df

# Bagian ini akan otomatis berjalan saat Anda eksekusi file .py
if __name__ == "__main__":
    
    # path data mentah
    RAW_DATA_PATH = '../PCOS_raw/PCOS_data_without_infertility.xlsx'
    
    # path data bersih
    CLEANED_DATA_PATH = '../preprocessing/PCOS_preprocessing.csv'
    
    # Jalankan fungsi preprocessing
    cleaned_df = preprocess_data(RAW_DATA_PATH)
    
    if cleaned_df is not None:
        cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
        print(f"\nData bersih berhasil disimpan di: {CLEANED_DATA_PATH}")
    else:
        print("Preprocessing gagal.")