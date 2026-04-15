import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")

# ================= UI GLASS =================
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}
.block-container {
    padding-top: 1rem;
}
.glass {
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    padding: 20px;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
h1,h2,h3 {
    color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)

st.title("🌴 SPK Sawit AI (AHP + SAW + Agronomi + Ekonomi)")

# ================= DATABASE =================
bibit_db = {
    "DxP PPKS": {"skor": 90, "produksi": "28–32 ton/ha"},
    "Tenera Socfindo": {"skor": 85, "produksi": "25–30 ton/ha"},
    "Lonsum": {"skor": 80, "produksi": "22–28 ton/ha"}
}

tanah_db = {
    "Latosol": {"skor": 90, "ph": "5.0–6.5"},
    "Podsolik": {"skor": 75, "ph": "4.5–5.5"},
    "Gambut": {"skor": 60, "ph": "3.0–4.0"}
}

biaya_db = {
    "Latosol": 28000000,
    "Podsolik": 32000000,
    "Gambut": 38000000
}

# ================= INPUT =================
mode = st.radio("Input Data:", ["Manual", "Upload Excel"], horizontal=True)

if mode == "Manual":
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("✍️ Input Agronomi")

    jumlah = st.number_input("Jumlah Alternatif", 1, 10, 3)

    data_list = []

    for i in range(jumlah):
        c1, c2, c3, c4 = st.columns(4)

        alt = c1.text_input(f"Alt {i+1}", f"A{i+1}", key=f"a{i}")
        tanah = c2.selectbox(f"Tanah {i+1}", list(tanah_db.keys()), key=f"t{i}")
        bibit = c3.selectbox(f"Bibit {i+1}", list(bibit_db.keys()), key=f"b{i}")
        ekonomi = c4.number_input(f"Ekonomi {i+1}", 0, 100, 70, key=f"e{i}")

        lahan_score = tanah_db[tanah]["skor"]
        bibit_score = bibit_db[bibit]["skor"]

        data_list.append([alt, lahan_score, bibit_score, ekonomi, tanah, bibit])

    df = pd.DataFrame(data_list, columns=[
        "Alternatif","Lahan","Bibit","Ekonomi","Tanah","Jenis Bibit"
    ])
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    file = st.file_uploader("Upload Excel", type=["xlsx"])
    if file:
        df = pd.read_excel(file)
    else:
        st.stop()
    st.markdown('</div>', unsafe_allow_html=True)

# ================= DATA =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("📊 Data")
st.dataframe(df, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ================= AHP =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("⚖️ AHP")

c1,c2,c3 = st.columns(3)
w1 = c1.slider("Lahan vs Bibit",1,9,3)
w2 = c2.slider("Lahan vs Ekonomi",1,9,4)
w3 = c3.slider("Bibit vs Ekonomi",1,9,2)

matrix = np.array([
    [1,w1,w2],
    [1/w1,1,w3],
    [1/w2,1/w3,1]
])

norm = matrix / matrix.sum(axis=0)
weights = norm.mean(axis=1)

eigval = np.max(np.linalg.eigvals(matrix).real)
CR = ((eigval-3)/2)/0.58

st.dataframe(pd.DataFrame({
    "Kriteria":["Lahan","Bibit","Ekonomi"],
    "Bobot":weights
}))
st.info(f"CR = {CR:.3f}")
st.markdown('</div>', unsafe_allow_html=True)

# ================= SAW =================
norm_df = df[["Lahan","Bibit","Ekonomi"]].copy()
for col in norm_df.columns:
    norm_df[col] /= norm_df[col].max()

df["Skor"] = (
    norm_df["Lahan"]*weights[0] +
    norm_df["Bibit"]*weights[1] +
    norm_df["Ekonomi"]*weights[2]
)

df = df.sort_values("Skor", ascending=False)
best = df.iloc[0]

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🏆 Ranking")
st.dataframe(df)
st.success(f"Terbaik: {best['Alternatif']}")
st.markdown('</div>', unsafe_allow_html=True)

# ================= AI PRODUKSI =================
tahun = np.arange(1,26).reshape(-1,1)

produksi_jurnal = np.array([
    0,0,3,8,15,20,25,28,30,30,
    29,28,27,26,25,24,23,22,21,20,
    18,16,14,12,10
])

model = LinearRegression()
model.fit(tahun, produksi_jurnal)

prediksi = model.predict(tahun) * best["Skor"]

st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("📈 Grafik Produksi")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=tahun.flatten(),
    y=prediksi,
    mode='lines+markers',
    line=dict(width=4)
))

peak = np.argmax(prediksi)

fig.add_trace(go.Scatter(
    x=[tahun.flatten()[peak]],
    y=[prediksi[peak]],
    mode='markers+text',
    text=["Peak"],
    textposition="top center"
))

fig.update_layout(template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ================= EKONOMI =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("💰 Analisis Ekonomi")

harga = st.slider("Harga TBS (Rp/kg)",1500,3500,2500)

tanah = best["Tanah"]
biaya = biaya_db[tanah]

if best["Skor"] >= 0.8:
    produksi_ton = 30
elif best["Skor"] >= 0.6:
    produksi_ton = 25
else:
    produksi_ton = 20

pendapatan = produksi_ton*1000*harga
keuntungan = pendapatan - biaya
roi = (keuntungan/biaya)*100

col1,col2,col3,col4 = st.columns(4)
col1.metric("Biaya", f"Rp {biaya:,}")
col2.metric("Produksi", f"{produksi_ton} ton")
col3.metric("Pendapatan", f"Rp {pendapatan:,.0f}")
col4.metric("ROI", f"{roi:.2f}%")

st.markdown('</div>', unsafe_allow_html=True)

# ================= REKOMENDASI =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🌱 Rekomendasi")

if tanah == "Gambut":
    pupuk = "Dolomit + NPK + Organik"
elif tanah == "Podsolik":
    pupuk = "NPK + Urea + KCl"
else:
    pupuk = "NPK + Kompos"

st.write(f"""
- Pupuk: {pupuk}  
- Panen: 30–36 bulan  
- Produksi: {produksi_ton} ton/ha  
- ROI: {roi:.2f}%  
""")

st.markdown('</div>', unsafe_allow_html=True)

# ================= INSIGHT =================
st.markdown('<div class="glass">', unsafe_allow_html=True)
st.subheader("🧠 Insight")

st.write(f"""
Alternatif {best['Alternatif']} terbaik berdasarkan kombinasi agronomi dan ekonomi.
Sistem menunjukkan bahwa kualitas lahan dan bibit sangat mempengaruhi hasil produksi dan keuntungan.
""")

st.markdown('</div>', unsafe_allow_html=True)
