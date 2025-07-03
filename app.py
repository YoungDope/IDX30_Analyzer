import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
from tradingview_screener import Query
from datetime import datetime
import pytz  # Untuk zona waktu Indonesia
import io  # Untuk membuat file Excel di memori

# Page Config
st.set_page_config(
    page_title="IDX30 Fundamental Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
STOCK_LIST = [
    'ADRO', 'AKRA', 'AMRT', 'ANTM', 'ARTO', 'ASII',
    'BBCA', 'BBNI', 'BBRI', 'BMRI', 'BMTR', 'BRPT',
    'CPIN', 'EXCL', 'GOTO', 'ICBP', 'INCO', 'INDF',
    'INKP', 'ISAT', 'KLBF', 'MAPI', 'MBMA', 'MDKA',
    'MEDC', 'PGAS', 'PTBA', 'SMGR', 'TLKM', 'UNTR', 'UNVR'
]
COLUMN_NAMES = [
    'Name',
    'Price Earnings TTM',
    'Price Book Ratio',
    'Debt to Equity',
    'Return on Equity',
    'Return on Assets',
    'After Tax Margin',
    'Current Ratio',
    'Operating Margin'
]
CRITERIA_TYPES = {
    'Price Earnings TTM': 'cost',
    'Price Book Ratio': 'cost',
    'Debt to Equity': 'cost',
    'Return on Equity': 'benefit',
    'Return on Assets': 'benefit',
    'After Tax Margin': 'benefit',
    'Current Ratio': 'benefit',
    'Operating Margin': 'benefit'
}
DEFAULT_AHP_MATRIX = np.array([
    [1,     3,       5,     0.5, 6,   4,   7,       2],
    [1/3,   1,       4,     1/3, 5,   3,   6,       0.5],
    [1/5,   1/4,     1,     1/5, 4,   2,   5,       1/3],
    [2,     3,       5,     1,   7,   4,   8,       3],
    [1/6,   1/5,     1/4,   1/7, 1,   1/3, 2,       1/5],
    [1/4,   1/3,     1/2,   1/4, 3,   1,   4,       1/2],
    [1/7,   1/6,     1/5,   1/8, 1/2, 1/4, 1,       1/6],
    [1/2,   2,       3,     1/3, 5,   2,   6,       1]
])

# Fungsi Hitung Bobot AHP dengan Geometric Mean
def calculate_ahp_weights(matrix):
    n = matrix.shape[0]
    weights = np.zeros(n)
    for i in range(n):
        product = np.prod(matrix[i, :], axis=0)
        weights[i] = product ** (1 / n)
    weights /= weights.sum()
    return dict(zip(COLUMN_NAMES[1:], weights))

# Validasi Konsistensi CR
def ahp_consistency(matrix):
    n = matrix.shape[0]
    eigenvalues, _ = np.linalg.eig(matrix)
    max_lambda = max(eigenvalues.real)
    ci = (max_lambda - n) / (n - 1)
    ri_values = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41}
    ri = ri_values.get(n, 1.45)
    cr = ci / ri
    return {"CI": ci, "RI": ri, "CR": cr}

# Caching data fetch
@st.cache_data(ttl=3600, show_spinner="Fetching IDX30 data...")
def get_fundamental_data():
    q = Query()
    q.limit(300)
    q.set_markets('indonesia')
    q.select(
        'name',
        'price_earnings_ttm',
        'price_book_ratio',
        'debt_to_equity',
        'return_on_equity',
        'return_on_assets',
        'after_tax_margin',
        'current_ratio',
        'operating_margin'
    )
    total, df = q.get_scanner_data()
    
    # Simpan timestamp saat data diambil
    fetch_time = datetime.now(pytz.timezone('Asia/Jakarta')).strftime('%d %b %Y, %H:%M:%S')

    if df is None or df.empty:
        return None, fetch_time
    
    df_idx30 = df[df['name'].isin(STOCK_LIST)][[
        'name',
        'price_earnings_ttm',
        'price_book_ratio',
        'debt_to_equity',
        'return_on_equity',
        'return_on_assets',
        'after_tax_margin',
        'current_ratio',
        'operating_margin'
    ]]
    df_idx30.columns = COLUMN_NAMES
    df_idx30.set_index('Name', inplace=True)

    return df_idx30, fetch_time

# Load custom data
def load_data_from_excel(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
        missing_cols = [col for col in COLUMN_NAMES if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
            return None
        df.set_index('Name', inplace=True)
        return df
    except Exception as e:
        st.error(f"Upload Error: {str(e)}")
        return None

# Fungsi AHP-TOPSIS Manual
def ahptopsis(df, weights):
    criteria = list(weights.keys())
    weight_values = np.array([weights[c] for c in criteria])
    # Normalisasi
    normalized = df[criteria].apply(lambda x: x / np.linalg.norm(x), axis=0)
    # Pembobotan
    weighted_matrix = normalized * weight_values
    # Tentukan A+ dan A-
    ideal_positive = []
    ideal_negative = []
    for col in criteria:
        if CRITERIA_TYPES[col] == 'benefit':
            ideal_positive.append(weighted_matrix[col].max())
            ideal_negative.append(weighted_matrix[col].min())
        else:
            ideal_positive.append(weighted_matrix[col].min())
            ideal_negative.append(weighted_matrix[col].max())
    ideal_positive = pd.Series(ideal_positive, index=criteria)
    ideal_negative = pd.Series(ideal_negative, index=criteria)
    # Jarak ke A+ dan A-
    dist_positive = np.sqrt(((weighted_matrix - ideal_positive) ** 2).sum(axis=1))
    dist_negative = np.sqrt(((weighted_matrix - ideal_negative) ** 2).sum(axis=1))
    # Skor Ci
    scores = dist_negative / (dist_positive + dist_negative)
    ranking_df = pd.DataFrame({
        'Score': scores,
        'Rank': scores.rank(ascending=False, method='dense').astype(int)
    }, index=df.index).sort_values(by='Score', ascending=False)
    return ranking_df

# CSS Styling with Glassmorphism
def inject_custom_css():
    st.markdown("""
    <style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
    }
    .main .block-container {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .stButton button {
        background-color: #3b82f6;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2563eb;
    }
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        animation: spin 1s linear infinite;
        margin: auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
    """, unsafe_allow_html=True)

# Generate Template Excel
def generate_excel_template():
    example_data = [
        ['BBRI', 12.34, 1.56, 0.89, 18.75, 5.67, 20.45, 2.3, 25.1],
        ['TLKM', 14.56, 2.10, 0.65, 16.40, 4.80, 18.90, 2.1, 22.0]
    ]
    df_example = pd.DataFrame(example_data, columns=COLUMN_NAMES)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_example.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return output

# Main Application
def main():
    inject_custom_css()
    st.title("📊 Sistem Pendukung Keputusan Investasi Saham Metode AHP-TOPSIS")
    
    with st.sidebar:
        st.title("📈 SPK AHP-TOPSIS")
        analysis_type = st.radio("Pilih Mode Input:", ["Data Pasar Realtime", "Upload Excel"], index=0)
        if st.button("🔄 Segarkan Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        # Tombol Unduh Template
        st.markdown("📥 Unduh Template Excel:")
        template_data = generate_excel_template()
        st.download_button(
            label="📄 Unduh Template Kosong",
            data=template_data,
            file_name="template_idx30.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    df = None
    if analysis_type == "Data Pasar Realtime":
        with st.spinner("🔍 Mengambil data saham dari pasar..."):
            df, fetch_time = get_fundamental_data()
            st.caption(f"🕒 Data diambil dari pasar pada: {fetch_time} WIB")
    else:
        uploaded_file = st.file_uploader("Unggah File Excel", type=['xlsx', 'xls'])
        if uploaded_file:
            df = load_data_from_excel(uploaded_file)

    if df is not None:
        clean_df = df.dropna()
        removed_stocks = df[~df.index.isin(clean_df.index)]
        st.subheader("📈 Ringkasan Informasi Finansial")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Rata-rata P/E", f"{clean_df['Price Earnings TTM'].mean():.2f}")
        metric_cols[1].metric("Median Debt/Equity", f"{clean_df['Debt to Equity'].median():.2f}")
        metric_cols[2].metric("Rata-rata ROE", f"{clean_df['Return on Equity'].mean():.2f}%")
        metric_cols[3].metric("Jumlah Saham", f"{len(clean_df)}")

        if not removed_stocks.empty:
            st.info(f"ℹ️ {len(removed_stocks)} saham tidak diproses karena mengandung nilai kosong: {', '.join(removed_stocks.index)}")
        if len(clean_df) == 0:
            st.error("❌ Tidak ada saham yang dapat diproses karena semua baris mengandung nilai kosong.")
            return

        st.divider()
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏆 Hasil Ranking", "⚖️ Pengaturan Bobot AHP", "📊 Visualisasi", "📋 Tabel Data", "ℹ️ Tentang"])

        # Tab 1: Hasil Ranking
        with tab1:
            st.markdown("### 🥇 Top 5 Saham Terbaik")
            if 'weights' not in st.session_state:
                st.session_state.weights = calculate_ahp_weights(DEFAULT_AHP_MATRIX)
            ranking = ahptopsis(clean_df, st.session_state.weights)
            top_5 = ranking.head(5)
            for i, (index, row) in enumerate(top_5.iterrows()):
                st.markdown(f"{i+1}. **{index}** - Skor: `{row['Score']:.4f}`")
            st.dataframe(ranking.style.background_gradient(cmap='Greens_r', subset=['Score']), use_container_width=True)

        # Tab 2: Pengaturan Bobot AHP
        with tab2:
            st.markdown("### ⚖️ Atur Matriks AHP Berpasangan")
            st.markdown("> 📘 *Catatan untuk Investor Pemula:* Geser nilai antar kriteria untuk menyesuaikan bobot.")

            # Penjelasan Skala Saaty
            st.markdown("""
            #### 📐 **Skala Perbandingan Saaty (1–9)**  
            Gunakan skala berikut saat membandingkan dua kriteria:
            
            | Nilai | Arti |
            |-------|------|
            | 1     | Sama penting |
            | 3     | Cukup dominan |
            | 5     | Dominan |
            | 7     | Sangat dominan |
            | 9     | Mutlak dominan |
            | 2,4,6,8 | Nilai antara |
            
            _Contoh: Jika ROE lebih penting daripada Debt to Equity, pilih nilai 5 atau lebih._
            """)

            criteria_names = COLUMN_NAMES[1:]
            if 'user_matrix' not in st.session_state:
                st.session_state.user_matrix = DEFAULT_AHP_MATRIX.tolist()

            def reset_matrix():
                st.session_state.user_matrix = DEFAULT_AHP_MATRIX.tolist()

            # Tombol Reset
            if st.button("🔁 Reset ke Default", use_container_width=True):
                reset_matrix()

            # Mapping label ke angka
            scale_options = {
                "1 - Sama Penting": 1.0,
                "2 - Sedikit Lebih Penting": 2.0,
                "3 - Cukup Dominan": 3.0,
                "4 - Lebih Penting": 4.0,
                "5 - Dominan": 5.0,
                "6 - Sangat Dominan (Antara)": 6.0,
                "7 - Sangat Dominan": 7.0,
                "8 - Hampir Mutlak": 8.0,
                "9 - Mutlak Dominan": 9.0
            }

            # Loop hanya sampai len(criteria_names) - 1 karena kita hindari Operating Margin terakhir
            for i in range(len(criteria_names) - 1):
                cols = st.columns(2)
                idx = 0
                for j in range(i + 1, len(criteria_names)):
                    col_label = cols[idx % 2]
                    col_select = cols[idx % 2]

                    with col_label:
                        st.markdown(f"**{criteria_names[i]} vs {criteria_names[j]}**")

                    with col_select:
                        default_value = float(st.session_state.user_matrix[i][j])
                        
                        # Cari key berdasarkan value
                        closest_key = min(scale_options.keys(), key=lambda k: abs(scale_options[k] - default_value))
                        
                        choice = st.selectbox(
                            label="",
                            options=list(scale_options.keys()),
                            index=list(scale_options.keys()).index(closest_key),
                            key=f"select_{i}_{j}",
                            help=f"Pilih tingkat dominasi {criteria_names[i]} terhadap {criteria_names[j]}"
                        )
                        val = scale_options[choice]
                        st.session_state.user_matrix[i][j] = val
                        st.session_state.user_matrix[j][i] = 1.0 / (val if val != 0 else 1)

                    idx += 1

            user_ahp_matrix = np.array(st.session_state.user_matrix)
            weights = calculate_ahp_weights(user_ahp_matrix)
            consistency = ahp_consistency(user_ahp_matrix)

            if consistency['CR'] > 0.1:
                st.warning(f"⚠️ Matriks tidak konsisten! CR = {consistency['CR']:.2f}. Harus kurang dari 0.1.")
            else:
                st.success(f"✅ Matriks konsisten! CR = {consistency['CR']:.2f}")

            st.session_state.weights = weights
            weight_df = pd.DataFrame({"Bobot": list(weights.values())}, index=criteria_names)
            fig = px.bar(weight_df, y='Bobot', title='Distribusi Bobot Kriteria (AHP)', color_discrete_sequence=['#3b82f6'])
            st.plotly_chart(fig, use_container_width=True)

        # Tab 3: Visualisasi
        with tab3:
            st.subheader("📉 Heatmap Korelasi Antar Indikator")
            corr = clean_df.corr().reset_index().melt('index')
            corr.columns = ['X', 'Y', 'Correlation']
            fig = px.imshow(corr.pivot(index='Y', columns='X', values='Correlation'),
                            text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🏆 Top Performers")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                undervalued = clean_df[clean_df['Price Earnings TTM'] > 0].nsmallest(5, 'Price Earnings TTM').reset_index()
                fig = px.bar(undervalued, x='Name', y='Price Earnings TTM', title='🔽 P/E Terendah (Undervalued)', color='Price Earnings TTM', color_continuous_scale='Tealgrn')
                st.plotly_chart(fig, use_container_width=True)
                
            with col2:
                high_roe = clean_df.nlargest(5, 'Return on Equity').reset_index()
                fig = px.bar(high_roe, x='Name', y='Return on Equity', title='💰 ROE Tertinggi', color='Return on Equity', color_continuous_scale='Emrld')
                st.plotly_chart(fig, use_container_width=True)
                
            with col3:
                low_debt = clean_df[clean_df['Debt to Equity'] > 0].nsmallest(5, 'Debt to Equity').reset_index()
                fig = px.bar(low_debt, x='Name', y='Debt to Equity', title='📉 Rasio Utang Terendah', color='Debt to Equity', color_continuous_scale='PurpOr')
                st.plotly_chart(fig, use_container_width=True)

        # Tab 4: Tabel Data
        with tab4:
            st.markdown("### 📘 Penjelasan Kolom Kriteria Saham")
            st.markdown("""
                | Kolom | Penjelasan |
                |--------|------------|
                | **Price Earnings TTM** | Rasio harga saham terhadap laba per saham (semakin rendah semakin bagus) |
                | **Price Book Ratio** | Harga saham dibanding nilai buku perusahaan |
                | **Debt to Equity** | Rasio utang terhadap ekuitas (rendah = stabil) |
                | **Return on Equity (ROE)** | Laba bersih terhadap ekuitas (tinggi = untung banyak) |
                | **Return on Assets (ROA)** | Efisiensi aset perusahaan |
                | **After Tax Margin** | Laba setelah pajak dalam persen |
                | **Current Ratio** | Kemampuan bayar utang jangka pendek |
                | **Operating Margin** | Laba operasional terhadap pendapatan |
            """)
            st.dataframe(
                clean_df.style.format({
                    'Price Earnings TTM': '{:.2f}',
                    'Price Book Ratio': '{:.2f}',
                    'Debt to Equity': '{:.2f}',
                    'Return on Equity': '{:.2f}%',
                    'Return on Assets': '{:.2f}%',
                    'After Tax Margin': '{:.2f}%',
                    'Current Ratio': '{:.2f}',
                    'Operating Margin': '{:.2f}%'
                }).background_gradient(cmap='Blues'),
                use_container_width=True,
                height=600
            )
            st.download_button(
                label="💾 Unduh CSV",
                data=clean_df.to_csv().encode('utf-8'),
                file_name="idx30_ranked.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Tab 5: Tentang
        with tab5:
            st.markdown("## ℹ️ Tentang Aplikasi Ini")
            st.markdown("""
                Web ini adalah **Sistem Pendukung Keputusan (SPK)** berbasis web yang dirancang untuk membantu investor menganalisis dan memilih saham terbaik di indeks **IDX30** menggunakan metode **AHP-TOPSIS**.
                ### 🎯 Tujuan
                - Membantu investor membuat keputusan investasi berdasarkan data fundamental.
                - Menghitung skor dan ranking saham berdasarkan bobot kriteria finansial.
                ### 🧩 Fitur
                - Pengambilan data pasar secara *real-time*
                - Upload file Excel untuk analisis offline
                - Penyesuaian bobot kriteria manual via AHP
                - Visualisasi interaktif dan hasil ranking
                - Download hasil analisis dalam format CSV
            """)

if __name__ == "__main__":
    main()