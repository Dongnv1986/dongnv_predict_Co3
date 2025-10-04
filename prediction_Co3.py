import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("Ứng dụng dự đoán CO3")

# ===========================
# 1. Upload file Excel
# ===========================
uploaded_file = st.file_uploader("📂 Chọn file Excel", type=["xlsx"])

# ===== Chỉ chạy nếu có file =====
if uploaded_file is not None:
    # Đọc dữ liệu
    df = pd.read_excel(uploaded_file)
    st.write("📂 Dữ liệu đã tải lên:")
    st.dataframe(df.head())

    # ===== Chuẩn bị dữ liệu =====
    features = ["Protein", "Salt", "Cacium"]  # đổi theo cột thực tế
    target = "ion_CO3"

    if all(f in df.columns for f in features + [target]):
        X = df[features].values
        y = df[target].values

        # Tách train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Tạo mô hình
        models = {
            "Linear": LinearRegression(),
            "Linear + StandardScaler": make_pipeline(StandardScaler(), LinearRegression()),
            "Linear + MinMaxScaler": make_pipeline(MinMaxScaler(), LinearRegression()),
            "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # Huấn luyện và đánh giá mô hình
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            results.append({
                "Model": name,
                "R2_train": r2_score(y_train, y_train_pred),
                "R2_test": r2_score(y_test, y_test_pred),
                "MSE_train": mean_squared_error(y_train, y_train_pred),
                "MSE_test": mean_squared_error(y_test, y_test_pred)
            })

        results_df = pd.DataFrame(results)

        # ===== Giao diện dự đoán =====
        st.subheader("Chọn mô hình dự đoán")
        model_choice = st.selectbox("Mô hình", results_df["Model"].tolist())

        st.subheader("Nhập thông số đầu vào")
        protein = st.number_input("Protein", value=0.0)
        salt = st.number_input("Salt", value=0.0)
        cacium = st.number_input("Cacium", value=0.0)

        if st.button("Dự đoán"):
            X_new = pd.DataFrame([[protein, salt, cacium]], columns=features)
            selected_model = models[model_choice]
            y_pred = selected_model.predict(X_new)[0]
            st.success(f"🔮 Dự đoán ion_CO3: {y_pred:.4f}")

        # Hiển thị bảng đánh giá mô hình
        st.subheader("📊 Hiệu quả các mô hình trên train/test")
        st.dataframe(results_df)

    else:
        st.error(f"⚠️ File Excel phải có các cột: {features + [target]}")
else:
    st.warning("👉 Vui lòng upload file Excel để tiếp tục")

# Hiển thị bảng đánh giá tất cả mô hình
st.subheader("Hiệu quả các mô hình trên train/test")
st.dataframe(results_df)
