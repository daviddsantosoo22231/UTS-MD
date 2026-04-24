import os
import pickle
import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    with open("best_cls_model.pkl", "rb") as f:
        cls_model = pickle.load(f)
    with open("best_reg_model.pkl", "rb") as f:
        reg_model = pickle.load(f)
    return cls_model, reg_model

cls_model, reg_model = load_models()

# ── Title ─────────────────────────────────────────────────────
st.title("🎓 Student Placement Predictor")
st.markdown("Prediksi **status penempatan kerja** dan **estimasi gaji** berdasarkan profil mahasiswa.")
st.divider()

# ── Input Form ────────────────────────────────────────────────
st.subheader("📋 Data Akademik")

col1, col2, col3 = st.columns(3)
with col1:
    cgpa = st.number_input("CGPA", 0.0, 10.0, 7.5, 0.1)
    tenth = st.number_input("Nilai SMA (10th %)", 0.0, 100.0, 75.0, 0.1)
    twelfth = st.number_input("Nilai SMA (12th %)", 0.0, 100.0, 75.0, 0.1)
    backlogs = st.number_input("Backlogs", 0, 20, 0)
with col2:
    study_hours = st.number_input("Jam Belajar/Hari", 0.0, 24.0, 4.0, 0.5)
    attendance = st.number_input("Kehadiran (%)", 0.0, 100.0, 80.0, 0.1)
    projects = st.number_input("Proyek Selesai", 0, 20, 2)
    internships = st.number_input("Magang", 0, 10, 1)
with col3:
    sleep_hours = st.number_input("Jam Tidur/Hari", 0.0, 12.0, 7.0, 0.5)
    stress = st.number_input("Tingkat Stres (1-10)", 1, 10, 5)
    hackathons = st.number_input("Hackathon Diikuti", 0, 20, 1)
    certifications = st.number_input("Sertifikasi", 0, 20, 2)

st.subheader("🛠️ Skill Rating")
col4, col5, col6 = st.columns(3)
with col4:
    coding = st.slider("Coding Skill (1-10)", 1, 10, 6)
with col5:
    communication = st.slider("Communication Skill (1-10)", 1, 10, 6)
with col6:
    aptitude = st.slider("Aptitude Skill (1-10)", 1, 10, 6)

st.subheader("👤 Data Personal")
col7, col8 = st.columns(2)
with col7:
    gender = st.selectbox("Gender", ["Male", "Female"])
    branch = st.selectbox("Jurusan", ["CSE", "ECE", "ME", "CE", "EE", "IT", "Other"])
    part_time = st.selectbox("Part-Time Job", ["Yes", "No"])
    family_income = st.selectbox("Tingkat Pendapatan Keluarga", ["Low", "Medium", "High"])
with col8:
    city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
    internet = st.selectbox("Akses Internet", ["Yes", "No"])
    extracurricular = st.selectbox("Kegiatan Ekstrakurikuler", ["Yes", "No"])

# ── Predict ───────────────────────────────────────────────────
st.divider()
if st.button("🔮 Prediksi Sekarang", type="primary", use_container_width=True):

    # Feature engineering (sama seperti pipeline.py)
    skill_composite = (coding + communication + aptitude) / 3
    academic_score  = cgpa * 10 * 0.5 + tenth * 0.25 + twelfth * 0.25

    input_data = pd.DataFrame([{
        # Numerical
        "cgpa": cgpa,
        "tenth_percentage": tenth,
        "twelfth_percentage": twelfth,
        "backlogs": backlogs,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance,
        "projects_completed": projects,
        "internships_completed": internships,
        "coding_skill_rating": coding,
        "communication_skill_rating": communication,
        "aptitude_skill_rating": aptitude,
        "hackathons_participated": hackathons,
        "certifications_count": certifications,
        "sleep_hours": sleep_hours,
        "stress_level": stress,
        "skill_composite": skill_composite,
        "academic_score": academic_score,
        # Categorical
        "gender": gender,
        "branch": branch,
        "part_time_job": part_time,
        "family_income_level": family_income,
        "city_tier": city_tier,
        "internet_access": internet,
        "extracurricular_involvement": extracurricular,
    }])

    placement_pred = cls_model.predict(input_data)[0]
    placement_prob = cls_model.predict_proba(input_data)[0][1]
    salary_pred    = reg_model.predict(input_data)[0]

    # ── Results ──────────────────────────────────────────────
    st.subheader("📊 Hasil Prediksi")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if placement_pred == 1:
            st.success("✅ **PLACED**")
            st.metric("Probabilitas Penempatan", f"{placement_prob*100:.1f}%")
        else:
            st.error("❌ **NOT PLACED**")
            st.metric("Probabilitas Penempatan", f"{placement_prob*100:.1f}%")
    with col_r2:
        if placement_pred == 1:
            st.metric("💰 Estimasi Gaji", f"₹ {salary_pred:.2f} LPA")
        else:
            st.metric("💰 Estimasi Gaji", "N/A")
            st.caption("Prediksi gaji hanya tersedia jika status Placed.")

    # ── Profile Summary ───────────────────────────────────────
    with st.expander("📈 Ringkasan Profil"):
        summary = pd.DataFrame({
            "Metrik": ["Skill Composite", "Academic Score", "CGPA", "Kehadiran", "Jam Belajar/Hari"],
            "Nilai": [
                f"{skill_composite:.2f} / 10",
                f"{academic_score:.2f} / 100",
                f"{cgpa:.1f} / 10",
                f"{attendance:.1f}%",
                f"{study_hours:.1f} jam"
            ]
        })
        st.table(summary)

st.caption("NIM: 2802421381 | DTSC6007001 – Deep Learning | UTS")
