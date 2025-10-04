import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pickle

st.title("Ứng dụng dự đoán CO3")

uploaded_file = st.file_uploader("📂 Chọn file Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.write("📂 Dữ liệu đã tải lên:")
    st.dataframe(df.head())

    features = ["Protein", "Salt", "Cacium"]
    target = "ion_CO3"

    if all(f in df.columns for f in features + [target]):
        X = df[features].values
        y = df[target].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Tạo các mô hình
        models = {
            "Linear": LinearRegression(),
            "Linear + StandardScaler": make_pipeline(StandardScaler(), LinearRegression()),
            "Linear + MinMaxScaler": make_pipeline(MinMaxScaler(), LinearRegression()),
            "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
            "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
        }

        # Huấn luyện và đánh giá, lưu model tốt nhất
        results = []
        best_r2 = -float('inf')
        best_model = None
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_test_pred)

            results.append({
                "Model": name,
                "R2_train": r2_score(y_train, model.predict(X_train)),
                "R2_test": r2,
                "MSE_train": mean_squared_error(y_train, model.predict(X_train)),
                "MSE_test": mean_squared_error(y_test, y_test_pred)
            })

            if r2 > best_r2:
                best_r2 = r2
                best_model = model

        results_df = pd.DataFrame(results)

        # Giao diện dự đoán
        st.subheader("Nhập thông số đầu vào")
        protein = st.number_input("Protein", value=0.0)
        salt = st.number_input("Salt", value=0.0)
        cacium = st.number_input("Cacium", value=0.0)

        if st.button("Dự đoán"):
            X_new = pd.DataFrame([[protein, salt, cacium]], columns=features)
            y_pred = best_model.predict(X_new)[0]
            st.success(f"🔮 Dự đoán ion_CO3: {y_pred:.4f}")

        st.subheader("📊 Hiệu quả các mô hình trên train/test")
        st.dataframe(results_df)

    else:
        st.error(f"⚠️ File Excel phải có các cột: {features + [target]}")
else:
    st.warning("👉 Vui lòng upload file Excel để tiếp tục")

# Hiển thị bảng đánh giá tất cả mô hình
st.subheader("Hiệu quả các mô hình trên train/test")
st.dataframe(results_df)
