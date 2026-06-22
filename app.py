import streamlit as st
import ultralytics
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import hashlib
import torch

YOLO_IMGSZ = 416
YOLO_IOU = 0.7
YOLO_MAX_DET = 300
CONFIDENCE_THRESHOLD = 0.50

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Penyakit Ikan Nila",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling minimalis
st.markdown("""
    <style>
    * { font-family: 'Segoe UI', Tahoma, Geneva, sans-serif; }
    [data-testid="stMetric"] { background: #f0f2f6; padding: 1rem; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🐟 Deteksi Penyakit Ikan Nila")
st.markdown("Aplikasi AI untuk deteksi penyakit ikan menggunakan YOLOv8")


# Database Penyakit
DISEASE_DB = {
    "Healthy-FIsh": {
        "emoji": "✅",
        "deskripsi": "Ikan dalam kondisi sehat",
        "gejala": "Tidak ada gejala penyakit",
        "solusi": [
            "Pertahankan kualitas air optimal (pH 6.5-7.5, Suhu 26-32°C)",
            "Berikan pakan berkualitas 2-3x sehari (3-5% berat badan)",
            "Lakukan pengecekan kesehatan berkala",
            "Jaga kebersihan kolam/tangki"
        ]
    },
    "Bacterial Aeromanas Disease": {
        "emoji": "🦠",
        "deskripsi": "Penyakit akibat infeksi bakteri Aeromonas pada ikan",
        "gejala": "Luka terbuka/ulcer di kulit, warna badan pucat atau kemerahan, insang berwarna pucat/merah, perut membengkak, ikan lesu",
        "solusi": [
            "Isolasi ikan yang terinfeksi segera ke tangki terpisah",
            "Lakukan pergantian air 50-70% setiap hari",
            "Berikan antibiotik sesuai dosis (Tetracycline, Erythromycin, atau Chloramphenicol)",
            "Obati luka dengan antiseptik (Povidone Iodine)",
            "Jaga suhu air 26-28°C dan pH 6.5-7.5 untuk mempercepat pemulihan"
        ]
    },
    "Streptococus": {
        "emoji": "⚪",
        "deskripsi": "Infeksi bakteri Streptococcus pada ikan",
        "gejala": "Mata berdarah/exophthalmia, insang pucat dan berlendir, tubuh berwarna gelap, sirip rusak, perut membengkak, ikan terlihat gelisah",
        "solusi": [
            "Isolasi ikan sakit segera dari kolam utama",
            "Ganti air 70% setiap hari untuk mengurangi beban bakteri",
            "Berikan antibiotik spektrum luas (Oxytetracycline atau Florfenicol) sesuai dosis",
            "Tingkatkan aerasi kolam untuk meningkatkan oksigen terlarut",
            "Hindari stres pada ikan dengan mengurangi kepadatan dan suara bising"
        ]
    },
    "Tilapia Lake Virus": {
        "emoji": "🦠",
        "deskripsi": "Penyakit virus TILV (Tilapia Lake Virus) pada ikan nila",
        "gejala": "Ikan berenang tidak normal/bengkok, mata keruh/berair, lesi di kulit, perut membengkak, ikan terlihat lemas dan tidak mau makan",
        "solusi": [
            "Isolasi ikan yang terinfeksi dan kurangi stres kolam",
            "Lakukan pergantian air besar (80-100%) secara berkala",
            "Tingkatkan kualitas pakan dengan vitamin C tinggi untuk meningkatkan imunitas",
            "Sterilisasi semua alat dan net untuk mencegah penyebaran virus",
            "Konsultasikan dengan ahli karena TILV belum ada obat spesifik (treatment suportif saja)"
        ]
    }
}

CLASS_ALIASES = {
    "Bacterial Aeromonas Disease": "Bacterial Aeromanas Disease",
    "Bacterial Aeromanas Disease": "Bacterial Aeromanas Disease",
    "Healthy Fish": "Healthy-FIsh",
    "Healthy-FIsh": "Healthy-FIsh",
    "Streptococcus": "Streptococus",
    "Streptococus": "Streptococus",
    "TILV": "Tilapia Lake Virus",
    "TLVD": "Tilapia Lake Virus",
    "Tilapia Lake Virus": "Tilapia Lake Virus",
}

def normalize_class_name(class_name):
    """Samakan variasi nama kelas dari model/dataset ke database app."""
    return CLASS_ALIASES.get(class_name, class_name)

def model_checksum(path="best.pt"):
    """Checksum singkat untuk memastikan model lokal sama dengan model di Colab."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()
    except OSError:
        return "-"

@st.cache_resource
def load_model(model_path, checksum):
    """Load YOLO model"""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def run_detection(image, model, conf=0.5):
    """Jalankan deteksi"""
    # Ensure image is numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ultralytics menganggap numpy array sebagai BGR seperti cv2.imread.
    # Streamlit/PIL memberi RGB, jadi perlu disamakan sebelum inferensi.
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results = model(
        image,
        conf=conf,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAX_DET,
        agnostic_nms=False,
        augment=False,
        verbose=False,
    )
    return results

def decode_image_bytes(image_bytes):
    """Decode upload bytes dengan OpenCV agar sama seperti cv2.imread/model(path)."""
    file_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Gambar tidak bisa dibaca oleh OpenCV. Coba file JPG/PNG lain.")
    return image_bgr

def run_detection_bgr(image_bgr, model, conf=0.5):
    """Jalankan deteksi dari gambar BGR seperti output cv2.imread di Colab."""
    results = model(
        image_bgr,
        conf=conf,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAX_DET,
        agnostic_nms=False,
        augment=False,
        verbose=False,
    )
    return results

def run_detection_numpy_as_is(image_array, model, conf=0.5):
    """Debug: jalankan YOLO tanpa konversi channel tambahan."""
    return model(
        image_array,
        conf=conf,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        max_det=YOLO_MAX_DET,
        agnostic_nms=False,
        augment=False,
        verbose=False,
    )

def result_to_table(result, min_conf=CONFIDENCE_THRESHOLD):
    """Ubah hasil YOLO menjadi tabel ringkas."""
    rows = []
    for i, (conf, cls) in enumerate(zip(result.boxes.conf, result.boxes.cls)):
        confidence_val = conf.item()
        if confidence_val < min_conf:
            continue
        rows.append({
            "No": len(rows) + 1,
            "Penyakit": normalize_class_name(result.names[int(cls)]),
            "Raw Class": result.names[int(cls)],
            "Confidence": f"{confidence_val:.1%}"
        })
    return rows

def show_disease_info(class_name, confidence):
    """Tampilkan info penyakit sesuai deteksi"""
    class_name = normalize_class_name(class_name)
    info = DISEASE_DB.get(class_name, {"emoji": "❓", "deskripsi": "Kelas tidak dikenali", "gejala": "", "solusi": []})
    
    if class_name == "Healthy-FIsh":
        st.success(f"{info['emoji']} **IKAN SEHAT**")
        st.info("Ikan Anda dalam kondisi baik. Lanjutkan menjaga kualitas air dan pakan berkualitas.")
    else:
        st.error(f"{info['emoji']} **HASIL: {class_name.upper()}** - Confidence: {confidence:.1%}")
        
        st.divider()
        
        # Gejala
        st.subheader(f"🔍 Gejala {class_name}")
        st.markdown(f"{info['gejala']}")
        
        st.divider()
        
        # Solusi Penanganan
        st.subheader(f"💊 Solusi Penanganan {class_name}")
        for i, sol in enumerate(info['solusi'], 1):
            st.markdown(f"**{i}. {sol}**")

def show_low_confidence_suggestions():
    """Tampilkan saran saat tidak ada prediksi valid di atas 50%."""
    st.warning("⚠️ **Tidak ada deteksi valid di atas confidence 50%.**")
    st.info(
        "Hasil di bawah 50% tidak dimasukkan ke class deteksi karena belum cukup yakin."
    )
    st.markdown("""
    **Kemungkinan penyebab:**
    - Gambar bukan ikan nila atau objek ikan tidak jelas
    - Pencahayaan kurang terang atau terlalu banyak bayangan
    - Gambar buram/blur atau resolusi terlalu rendah
    - Ikan terlalu jauh, terpotong, atau tidak berada di tengah frame
    - Ada pantulan air/kaca yang menutupi bagian tubuh ikan

    **Saran:**
    - Ambil ulang foto dengan cahaya lebih terang
    - Pastikan seluruh tubuh ikan nila terlihat jelas
    - Gunakan gambar JPG/PNG yang tajam dan tidak terkompres berlebihan
    - Hindari background terlalu ramai
    """)

def draw_no_detection_overlay(image_bgr):
    """Beri penanda visual saat tidak ada box valid di atas threshold."""
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]
    color = (0, 165, 255)
    thickness = max(2, width // 220)
    cv2.rectangle(annotated, (4, 4), (width - 5, height - 5), color, thickness)

    label = "Tidak ada deteksi >= 50%"
    font_scale = max(0.45, min(width, height) / 520)
    font_thickness = max(1, width // 360)
    (text_w, text_h), baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )
    pad = 8
    cv2.rectangle(
        annotated,
        (8, 8),
        (min(width - 8, 8 + text_w + pad * 2), 8 + text_h + baseline + pad * 2),
        color,
        -1,
    )
    cv2.putText(
        annotated,
        label,
        (8 + pad, 8 + pad + text_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )
    return annotated

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Panduan Deteksi")
    confidence = CONFIDENCE_THRESHOLD
    
    # Tips Upload
    with st.expander("📷 Tips Upload Gambar", expanded=False):
        st.markdown("""
        **Untuk hasil terbaik:**
        - Ambil foto dengan pencahayaan cukup
        - Posisikan ikan di tengah frame
        - Hindari bayangan atau refleksi
        - Gunakan resolusi minimal 640x640 pixel
        - Foto dari berbagai sudut
        """)
    
    # Penyakit Info
    with st.expander("🦠 Info Penyakit", expanded=False):
        disease_select = st.selectbox(
            "Pilih penyakit untuk info:",
            list(DISEASE_DB.keys())
        )
        
        if disease_select:
            disease_info = DISEASE_DB[disease_select]
            st.markdown(f"### {disease_info['emoji']} {disease_select}")
            st.markdown(f"**Gejala:** {disease_info['gejala']}")
            st.markdown("**Penanganan:**")
            for sol in disease_info['solusi']:
                st.markdown(f"- {sol}")
    
    # Troubleshooting
    with st.expander("⚠️ Masalah & Solusi", expanded=False):
        st.markdown("""
        **Deteksi tidak muncul?**
        - Gambar terlalu buram/blur
        - Pencahayaan kurang cerah
        - Ikan tidak jelas terlihat
        - Coba upload ulang dengan kualitas lebih baik
        
        **Hasil tidak akurat?**
        - Ambil foto dari sudut berbeda
        - Pastikan seluruh tubuh ikan terlihat
        - Hindari posisi ikan terpotong
        """)

    with st.expander("🧪 Debug Model", expanded=False):
        st.caption("Model: best.pt")
        st.caption(f"MD5: {model_checksum()}")
        st.caption(f"Ultralytics: {ultralytics.__version__}")
        st.caption(f"Torch: {torch.__version__}")
        if st.button("Clear model cache"):
            st.cache_resource.clear()
            st.rerun()
    show_debug_predictions = st.checkbox("Tampilkan raw prediction", value=False)

# Load model
MODEL_PATH = "best.pt"
MODEL_MD5 = model_checksum(MODEL_PATH)
model = load_model(MODEL_PATH, MODEL_MD5)

if model:
    tab1, tab2, tab3 = st.tabs(["📸 Upload", "🎥 Kamera", "📊 Info"])
    
    # TAB 1: UPLOAD
    with tab1:
        uploaded = st.file_uploader("Pilih gambar ikan", type=["jpg", "jpeg", "png"])
        
        if uploaded:
            try:
                uploaded_bytes = uploaded.getvalue()
                image_bgr = decode_image_bytes(uploaded_bytes)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_md5 = hashlib.md5(uploaded_bytes).hexdigest()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image_rgb, caption="Gambar Original", use_column_width=True)
                
                with col2:
                    with st.spinner("Mendeteksi..."):
                        results = run_detection_bgr(image_bgr, model, confidence)
                    result = results[0]
                    annotated = result.plot() if len(result.boxes) > 0 else draw_no_detection_overlay(image_bgr)
                    st.image(annotated, caption="Hasil Deteksi", use_column_width=True, channels="BGR")
                
                st.divider()
                
                if len(result.boxes) > 0:
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Jumlah Deteksi", len(result.boxes))
                    with col2:
                        st.metric("Rata-rata Conf", f"{result.boxes.conf.mean():.1%}")
                    with col3:
                        st.metric("Confidence Max", f"{result.boxes.conf.max():.1%}")
                    with col4:
                        st.metric("Model", "YOLOv8")
                    
                    st.divider()
                    
                    # Detail tabel
                    st.dataframe(result_to_table(result), use_container_width=True, hide_index=True)

                    if show_debug_predictions:
                        with st.expander("Raw prediction debug", expanded=True):
                            st.caption(f"Image MD5: {image_md5}")
                            st.caption(f"Model MD5: {MODEL_MD5}")
                            debug_result = run_detection_bgr(image_bgr, model, CONFIDENCE_THRESHOLD)[0]
                            debug_rgb_result = run_detection_numpy_as_is(image_rgb, model, CONFIDENCE_THRESHOLD)[0]
                            st.markdown("**OpenCV/BGR seperti Colab `model(path)`**")
                            debug_rows = result_to_table(debug_result)
                            if debug_rows:
                                st.dataframe(debug_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")
                            st.markdown("**RGB numpy debug**")
                            debug_rgb_rows = result_to_table(debug_rgb_result)
                            if debug_rgb_rows:
                                st.dataframe(debug_rgb_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")
                    
                    st.divider()
                    
                    # Info penyakit
                    best_idx = int(result.boxes.conf.argmax())
                    first_class_id = int(result.boxes.cls[best_idx])
                    first_class = result.names[first_class_id]
                    first_conf = result.boxes.conf[best_idx].item()
                    
                    st.subheader("Analisis Deteksi Utama")
                    show_disease_info(first_class, first_conf)
                
                else:
                    if show_debug_predictions:
                        with st.expander("Raw prediction debug", expanded=True):
                            st.caption(f"Image MD5: {image_md5}")
                            st.caption(f"Model MD5: {MODEL_MD5}")
                            debug_result = run_detection_bgr(image_bgr, model, CONFIDENCE_THRESHOLD)[0]
                            debug_rgb_result = run_detection_numpy_as_is(image_rgb, model, CONFIDENCE_THRESHOLD)[0]
                            st.markdown("**OpenCV/BGR seperti Colab `model(path)`**")
                            debug_rows = result_to_table(debug_result)
                            if debug_rows:
                                st.dataframe(debug_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")
                            st.markdown("**RGB numpy debug**")
                            debug_rgb_rows = result_to_table(debug_rgb_result)
                            if debug_rgb_rows:
                                st.dataframe(debug_rgb_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")

                    st.divider()
                    show_low_confidence_suggestions()
            
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.info("Pastikan gambar jelas dan format yang didukung (JPG, PNG)")
    
    # TAB 2: KAMERA
    with tab2:
        camera = st.camera_input("Ambil foto ikan")
        
        if camera:
            try:
                camera_bytes = camera.getvalue()
                image_bgr = decode_image_bytes(camera_bytes)
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                image_md5 = hashlib.md5(camera_bytes).hexdigest()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image_rgb, caption="Foto Kamera", use_column_width=True)
                
                with col2:
                    with st.spinner("Memproses..."):
                        results = run_detection_bgr(image_bgr, model, confidence)
                    result = results[0]
                    annotated = result.plot() if len(result.boxes) > 0 else draw_no_detection_overlay(image_bgr)
                    st.image(annotated, caption="Hasil Deteksi", use_column_width=True, channels="BGR")
                
                st.divider()
                
                if len(result.boxes) > 0:
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Deteksi", len(result.boxes))
                    with col2:
                        st.metric("Rata-rata", f"{result.boxes.conf.mean():.1%}")
                    with col3:
                        st.metric("Max Conf", f"{result.boxes.conf.max():.1%}")
                    with col4:
                        st.metric("Model", "YOLOv8")
                    
                    st.divider()
                    st.subheader("Hasil Deteksi")

                    if show_debug_predictions:
                        with st.expander("Raw prediction debug", expanded=True):
                            st.caption(f"Image MD5: {image_md5}")
                            st.caption(f"Model MD5: {MODEL_MD5}")
                            debug_result = run_detection_bgr(image_bgr, model, CONFIDENCE_THRESHOLD)[0]
                            debug_rgb_result = run_detection_numpy_as_is(image_rgb, model, CONFIDENCE_THRESHOLD)[0]
                            st.markdown("**OpenCV/BGR seperti Colab `model(path)`**")
                            debug_rows = result_to_table(debug_result)
                            if debug_rows:
                                st.dataframe(debug_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")
                            st.markdown("**RGB numpy debug**")
                            debug_rgb_rows = result_to_table(debug_rgb_result)
                            if debug_rgb_rows:
                                st.dataframe(debug_rgb_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")
                    
                    best_idx = int(result.boxes.conf.argmax())
                    first_class_id = int(result.boxes.cls[best_idx])
                    first_class = result.names[first_class_id]
                    first_conf = result.boxes.conf[best_idx].item()
                    
                    show_disease_info(first_class, first_conf)
                
                else:
                    if show_debug_predictions:
                        with st.expander("Raw prediction debug", expanded=True):
                            st.caption(f"Image MD5: {image_md5}")
                            st.caption(f"Model MD5: {MODEL_MD5}")
                            debug_result = run_detection_bgr(image_bgr, model, CONFIDENCE_THRESHOLD)[0]
                            debug_rgb_result = run_detection_numpy_as_is(image_rgb, model, CONFIDENCE_THRESHOLD)[0]
                            st.markdown("**OpenCV/BGR seperti Colab `model(path)`**")
                            debug_rows = result_to_table(debug_result)
                            if debug_rows:
                                st.dataframe(debug_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")
                            st.markdown("**RGB numpy debug**")
                            debug_rgb_rows = result_to_table(debug_rgb_result)
                            if debug_rgb_rows:
                                st.dataframe(debug_rgb_rows, use_container_width=True, hide_index=True)
                            else:
                                st.write("Tidak ada box dengan confidence minimal 50%.")

                    st.divider()
                    show_low_confidence_suggestions()
            
            except Exception as e:
                st.error(f"Error processing camera image: {str(e)}")
                st.info("Coba ambil foto dengan pencahayaan yang lebih baik")
    
    # TAB 3: INFO
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Model Info")
            st.markdown(f"""
            - **Type**: YOLO v8
            - **Framework**: Ultralytics
            - **Classes**: {len(model.names)}
            - **Input Size**: 640x640
            - **Accuracy**: ~95%
            """)
        
        with col2:
            st.markdown("### 📋 Kelas Deteksi")
            for idx, name in model.names.items():
                normalized_name = normalize_class_name(name)
                emoji = "✅" if normalized_name == "Healthy-FIsh" else "⚠️"
                st.markdown(f"- {emoji} **{normalized_name}**")
        
        st.divider()
        st.markdown("### 💡 Tips Penggunaan")
        st.markdown("""
        1. **Upload Gambar**: Unggah foto berkualitas baik dari berbagai sudut
        2. **Kamera**: Ambil foto langsung dengan pencahayaan optimal
        3. **Confidence**: Sesuaikan threshold untuk hasil akurat
        4. **Konsultasi**: Hubungi ahli untuk kasus serius
        """)
        
        st.divider()
        st.markdown("### 🌡️ Kondisi Air Optimal")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Parameter Air**
            - pH: 6.5-7.5
            - Suhu: 26-32°C
            - Oksigen: >5 mg/L
                - Amoniak: <0.02 mg/L
                """)
        
        with col2:
            st.markdown("""
            **Manajemen Pakan**
            - Frekuensi: 2-3x/hari
            - Jumlah: 3-5% berat badan
            - Kualitas: Premium
            - Variasi: Pellet + alami
            """)

else:
    st.error("❌ Gagal memuat model. Pastikan file best.pt ada di folder aplikasi.")
