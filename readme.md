# Panduan Instalasi dan Menjalankan Aplikasi

## Prasyarat

- Python 3.7 atau versi lebih baru
- pip (Python package installer)

## Langkah-langkah Instalasi

### 1. Clone Repository

```bash
git clone <repository-url>
cd <nama-project>
```

### 2. Membuat Virtual Environment

```bash
# Membuat virtual environment
python -m venv venv

# Atau menggunakan python3 jika diperlukan
python3 -m venv venv
```

### 3. Mengaktifkan Virtual Environment

#### Untuk Windows:

```bash
# Command Prompt
venv\Scripts\activate

# PowerShell
venv\Scripts\Activate.ps1

# Git Bash
source venv/Scripts/activate
```

#### Untuk Linux/macOS:

```bash
source venv/bin/activate
```

### 4. Menginstall Dependencies

```bash
# Pastikan virtual environment sudah aktif (akan terlihat (venv) di awal command line)
pip install -r requirements.txt
```

### 5. Menjalankan Aplikasi

```bash
# Sesuaikan dengan file utama folder
python jawaban4.py
```

## Menonaktifkan Virtual Environment

```bash
deactivate
```

## Troubleshooting

### Jika pip tidak dikenali:

```bash
python -m pip install -r requirements.txt
```

### Update pip jika diperlukan:

```bash
python -m pip install --upgrade pip
```
