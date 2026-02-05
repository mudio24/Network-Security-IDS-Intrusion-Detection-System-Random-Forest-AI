# 🛡️ CyberGuard AI: Enterprise Edition

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Sistem Deteksi Intrusi Jaringan (Network IDS) Berbasis Machine Learning**

*Network Intrusion Detection System (IDS) Powered by Machine Learning*

[🇮🇩 Bahasa Indonesia](#-tentang-proyek) | [🇬🇧 English](#-about-the-project)

</div>

---

## 📖 Tentang Proyek

**CyberGuard AI: Enterprise Edition** adalah aplikasi Sistem Deteksi Intrusi Jaringan (IDS) yang canggih, dibangun menggunakan Streamlit dan Machine Learning. Aplikasi ini menggunakan dataset NSL-KDD untuk melatih model Random Forest Classifier dalam mendeteksi intrusi jaringan.

Sistem ini dirancang untuk mensimulasikan skenario pemantauan keamanan dunia nyata, khususnya dalam lingkungan yang menggunakan router Mikrotik.

### 🎯 Tujuan

- Mendeteksi anomali dan serangan pada lalu lintas jaringan secara real-time
- Menyediakan dashboard monitoring keamanan yang komprehensif
- Analisis log jaringan secara batch untuk audit keamanan
- Mendukung integrasi dengan Mikrotik Router

---

## 📖 About The Project

**CyberGuard AI: Enterprise Edition** is an advanced Network Intrusion Detection System (IDS) application built using Streamlit and Machine Learning. It uses the NSL-KDD dataset to train a Random Forest classifier for detecting network intrusions.

The system is designed to simulate real-world security monitoring scenarios, particularly in environments using Mikrotik routers.

### 🎯 Purpose

- Detect anomalies and attacks on network traffic in real-time
- Provide a comprehensive security monitoring dashboard
- Batch analysis of network logs for security audits
- Support integration with Mikrotik Routers

---

## ✨ Fitur Utama / Key Features

| Fitur / Feature | Deskripsi / Description |
|-----------------|-------------------------|
| 📊 **Dashboard Interaktif** | Ringkasan eksekutif dengan metrik keamanan real-time |
| 🔍 **Simulasi Langsung** | Analisis paket manual dengan prediksi AI |
| 📁 **Analisis Batch** | Upload file CSV untuk analisis massal |
| 🧠 **Performa AI** | Confusion matrix, feature importance, dan metrik evaluasi |
| 🌐 **Dual Bahasa** | Mendukung Bahasa Indonesia dan English |
| 🎨 **UI Glassmorphism** | Desain modern dengan tema cyber security |

---

## 🛠️ Teknologi / Tech Stack

### Framework & Library

```
├── 🌐 Web Framework
│   └── Streamlit >= 1.28.0
│
├── 📊 Data Processing
│   ├── Pandas >= 2.0.0
│   └── NumPy >= 1.24.0
│
├── 🤖 Machine Learning
│   └── Scikit-learn >= 1.3.0
│
├── 📈 Visualization
│   ├── Matplotlib >= 3.7.0
│   └── Seaborn >= 0.12.0
│
└── 📦 Dataset
    └── NSL-KDD (via local/kagglehub)
```

### Model Machine Learning

- **Algoritma**: Random Forest Classifier
- **Estimators**: 100 trees
- **Max Depth**: 20
- **Akurasi**: ~99% pada dataset NSL-KDD

---

## 📁 Struktur Proyek / Project Structure

```
soc-streamlite/
├── 📄 app.py              # Aplikasi utama Streamlit
├── 📄 languages.py        # Konfigurasi multi-bahasa (ID/EN)
├── 📄 icons.py            # Koleksi ikon SVG untuk UI
├── 📄 mikrotik_api.py     # Integrasi API Mikrotik Router
├── 📄 requirements.txt    # Daftar dependensi Python
├── 📄 README.md           # Dokumentasi proyek
│
├── 📁 dataset/            # Dataset NSL-KDD
│   └── KDDTrain+.txt
│
├── 📁 models/             # Model ML tersimpan
│
└── 📁 venv/               # Virtual environment
```

---

## 🚀 Instalasi / Installation

### Prasyarat / Prerequisites

- Python 3.10 atau lebih baru
- pip (Python package manager)
- Git (opsional)

### Langkah Instalasi / Installation Steps

1. **Clone repository**
   ```bash
   git clone https://github.com/username/soc-streamlite.git
   cd soc-streamlite
   ```

2. **Buat virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependensi**
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```

5. **Buka browser**
   ```
   http://localhost:8501
   ```

---

## 📊 Dataset

### NSL-KDD Dataset

Dataset yang digunakan adalah **NSL-KDD**, yang merupakan versi perbaikan dari dataset KDD Cup 99 untuk penelitian Network Intrusion Detection.

**Fitur yang dipilih (8 kolom):**

| Kolom | Deskripsi | Tipe |
|-------|-----------|------|
| `duration` | Durasi koneksi dalam detik | Integer |
| `protocol_type` | Protokol transport (TCP/UDP/ICMP) | Categorical |
| `service` | Layanan jaringan (http, ftp, dll) | Categorical |
| `flag` | Status koneksi | Categorical |
| `src_bytes` | Bytes dari sumber ke tujuan | Integer |
| `dst_bytes` | Bytes dari tujuan ke sumber | Integer |
| `count` | Jumlah koneksi ke host yang sama | Integer |
| `srv_count` | Jumlah koneksi ke layanan yang sama | Integer |

**Klasifikasi Target:**
- `0` = Normal Traffic
- `1` = Attack / Anomaly

---

## 📸 Screenshots

### Dashboard Overview
> Tampilan utama dengan metrik keamanan real-time, grafik lalu lintas, dan distribusi klasifikasi.

### Live Simulation
> Form input untuk analisis paket individual dengan hasil prediksi AI.

### Batch Analysis
> Upload file CSV log jaringan untuk analisis massal dan download hasil.

### AI Performance
> Confusion matrix, feature importance, dan metrik evaluasi model.

---

## 🔧 Konfigurasi / Configuration

### Mengubah Bahasa Default

Edit file `app.py`, pada fungsi `main()`:

```python
# Default bahasa Indonesia
if 'lang' not in st.session_state:
    st.session_state.lang = 'id'  # Ubah ke 'en' untuk English
```

### Mengubah Port

```bash
streamlit run app.py --server.port 8080
```

### Mode Production

```bash
streamlit run app.py --server.headless true --server.port 80
```

---

## 📚 Referensi Akademik / Academic References

### Dataset Citation

> Tavallaee, M., Bagheri, E., Lu, W., and Ghorbani, A. A. (2009).
> "A Detailed Analysis of the KDD CUP 99 Data Set."
> *Proceedings of the 2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA)*.

### Model Reference

> Breiman, L. (2001). "Random Forests." *Machine Learning*, 45(1), 5-32.

---

## 🤝 Kontribusi / Contributing

Kontribusi sangat diterima! Silakan buat Pull Request atau buka Issue untuk diskusi.

1. Fork repository
2. Buat branch fitur (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buka Pull Request

---

## 📄 Lisensi / License

Didistribusikan di bawah Lisensi MIT. Lihat `LICENSE` untuk informasi lebih lanjut.

---

## 👨‍💻 Pengembang / Developer

**CyberGuard AI Development Team**

---

## 📞 Kontak / Contact

Jika ada pertanyaan atau saran, silakan hubungi melalui:

- 📧 Email: [your-email@example.com]
- 🌐 Website: [your-website.com]
- 💼 LinkedIn: [your-linkedin]

---

<div align="center">

**⭐ Jika proyek ini bermanfaat, berikan bintang di GitHub! ⭐**

*Powered by Machine Learning | © 2024 mudio24*

</div>
