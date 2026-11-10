
import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib

from pathlib import Path


# ==========================================
# KONFIGURASI
# ==========================================

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🌐",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model_anomali.pth"
SCALER_PATH = BASE_DIR / "scaler.pkl"

FEATURES = [
    "SourceIP",
    "DestinationIP",
    "SourcePort",
    "DestinationPort",
    "Protocol",
    "BytesSent",
    "BytesReceived",
    "PacketsSent",
    "PacketsReceived",
    "Duration",
    "TotalBytes",
    "TotalPackets"
]


# ==========================================
# MODEL PYTORCH
# ==========================================

class AnomalyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(12, 64)
        self.bn1 = nn.BatchNorm1d(64, eps=0.001)
        self.dropout1 = nn.Dropout(0.3)

        self.dense2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32, eps=0.001)
        self.dropout2 = nn.Dropout(0.3)

        self.dense3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.dense1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        x = self.dense2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        x = self.dense3(x)
        x = self.sigmoid(x)

        return x


# ==========================================
# LOAD MODEL DAN SCALER
# ==========================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH.name}"
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"File scaler tidak ditemukan: {SCALER_PATH.name}"
        )

    model = AnomalyModel()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)

    model.eval()

    scaler = joblib.load(SCALER_PATH)

    if hasattr(scaler, "feature_names_in_"):

        scaler_features = list(scaler.feature_names_in_)

        if scaler_features != FEATURES:

            raise ValueError(
                "Urutan fitur scaler tidak sesuai dengan aplikasi."
                f"\nFitur scaler: {scaler_features}"
                f"\nFitur aplikasi: {FEATURES}"
            )

    if hasattr(scaler, "n_features_in_"):

        if scaler.n_features_in_ != len(FEATURES):

            raise ValueError(
                "Jumlah fitur scaler tidak sesuai. "
                f"Scaler: {scaler.n_features_in_}, "
                f"aplikasi: {len(FEATURES)}"
            )

    return model, scaler


# ==========================================
# HEADER
# ==========================================

st.title("🌐 Network Anomaly Detection")

st.write(
    "Aplikasi Machine Learning untuk mendeteksi "
    "anomali pada trafik jaringan menggunakan model PyTorch."
)

st.divider()


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.header("Informasi Model")

    st.success("Model: PyTorch")

    st.info("Scaler: StandardScaler")

    st.write("Jumlah fitur: 12")

    st.write("Target: IsAnomaly")

    st.divider()

    st.caption(
        "Machine Learning - Deteksi Anomali Trafik Jaringan"
    )


# ==========================================
# LOAD MODEL
# ==========================================

try:

    model, scaler = load_model()

    st.success("Model dan scaler berhasil dimuat.")

except Exception as e:

    st.error("Model atau scaler gagal dimuat.")

    st.code(str(e))

    st.stop()


# ==========================================
# UPLOAD CSV
# ==========================================

st.subheader("📂 Upload Dataset Trafik")

uploaded_file = st.file_uploader(
    "Upload CSV dataset Kaggle mentah",
    type=["csv"]
)


# ==========================================
# PROSES DATASET
# ==========================================

if uploaded_file is not None:

    try:

        # Baca dataset
        df_raw = pd.read_csv(uploaded_file)

        st.success("Dataset berhasil dibaca!")

        st.subheader("Preview Dataset")

        st.dataframe(
            df_raw.head(10),
            use_container_width=True
        )

        # ======================================
        # VALIDASI KOLOM AWAL
        # ======================================

        required_columns = [
            "SourceIP",
            "DestinationIP",
            "SourcePort",
            "DestinationPort",
            "Protocol",
            "BytesSent",
            "BytesReceived",
            "PacketsSent",
            "PacketsReceived",
            "Duration"
        ]

        missing_columns = [
            column
            for column in required_columns
            if column not in df_raw.columns
        ]

        if missing_columns:

            st.error(
                "Dataset tidak memiliki kolom yang diperlukan."
            )

            st.write("Kolom yang belum tersedia:")

            st.write(missing_columns)

            st.stop()

        # ======================================
        # CLEANING DATA
        # ======================================

        data_clean = df_raw.dropna().copy()

        # ======================================
        # FEATURE ENGINEERING
        # ======================================

        data_clean["TotalBytes"] = (
            data_clean["BytesSent"]
            + data_clean["BytesReceived"]
        )

        data_clean["TotalPackets"] = (
            data_clean["PacketsSent"]
            + data_clean["PacketsReceived"]
        )

        # ======================================
        # VALIDASI FITUR
        # ======================================

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in data_clean.columns
        ]

        if missing_features:

            st.error(
                "Dataset belum memiliki seluruh fitur model."
            )

            st.write(missing_features)

            st.stop()

        # ======================================
        # INFORMASI DATA
        # ======================================

        st.divider()

        col1, col2 = st.columns(2)

        col1.metric(
            "Jumlah Data Awal",
            len(df_raw)
        )

        col2.metric(
            "Data Setelah Cleaning",
            len(data_clean)
        )

        st.subheader("Preview 12 Fitur")

        st.dataframe(
            data_clean[FEATURES].head(10),
            use_container_width=True
        )

        # ======================================
        # DETEKSI ANOMALI
        # ======================================

        st.divider()

        st.subheader("🔍 Deteksi Anomali")

        if st.button(
            "Jalankan Deteksi",
            type="primary",
            use_container_width=True
        ):

            try:

                # Ambil 12 fitur sesuai urutan training
                X = data_clean[FEATURES].copy()

                # Pastikan seluruh fitur numerik
                X = X.apply(
                    pd.to_numeric,
                    errors="raise"
                )

                # Validasi nilai kosong
                if X.isnull().any().any():

                    st.error(
                        "Masih terdapat nilai kosong pada fitur."
                    )

                    st.stop()

                # Validasi nilai tak terhingga
                if not np.isfinite(
                    X.to_numpy(dtype=np.float64)
                ).all():

                    st.error(
                        "Dataset memiliki nilai inf atau -inf."
                    )

                    st.stop()

                # ==================================
                # SCALING
                # ==================================

                # Data masih mentah.
                # Gunakan scaler training satu kali.

                X_scaled = scaler.transform(X)

                X_scaled = np.asarray(
                    X_scaled,
                    dtype=np.float32
                )

                # ==================================
                # MODEL INFERENCE
                # ==================================

                tensor_data = torch.tensor(
                    X_scaled,
                    dtype=torch.float32
                )

                with torch.no_grad():

                    predictions = model(tensor_data)

                probabilities = (
                    predictions.cpu().numpy().flatten()
                )

                # Threshold 0.5
                labels = (
                    probabilities >= 0.5
                ).astype(int)

                # ==================================
                # HASIL
                # ==================================

                result_df = data_clean.copy()

                result_df["AnomalyScore"] = probabilities

                result_df["Prediction"] = np.where(
                    labels == 1,
                    "Anomali",
                    "Normal"
                )

                total_data = len(result_df)

                total_normal = int(
                    (labels == 0).sum()
                )

                total_anomaly = int(
                    (labels == 1).sum()
                )

                # ==================================
                # METRICS
                # ==================================

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Trafik",
                    total_data
                )

                col2.metric(
                    "Trafik Normal",
                    total_normal
                )

                col3.metric(
                    "Trafik Anomali",
                    total_anomaly
                )

                # ==================================
                # HASIL TABEL
                # ==================================

                st.subheader("📊 Hasil Prediksi")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                # ==================================
                # DOWNLOAD HASIL
                # ==================================

                csv_result = result_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Hasil Prediksi",
                    data=csv_result,
                    file_name="hasil_deteksi_anomali.csv",
                    mime="text/csv"
                )

            except Exception as e:

                st.error(
                    "Terjadi kesalahan saat melakukan prediksi."
                )

                st.code(str(e))

    except Exception as e:

        st.error("Dataset gagal dibaca.")

        st.code(str(e))
