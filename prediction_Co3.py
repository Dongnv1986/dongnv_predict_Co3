import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ===========================
import streamlit as st
import pandas as pd

st.title("Ứng dụng dự đoán CO3")

# Bước 1: Upload file
uploaded_file = st.file_uploader("📂 Chọn file Excel", type=["xlsx"])

if uploaded_file is not None:
    # Bước 2: Đọc dữ liệu
    df = pd.read_excel(uploaded_file)
    st.write("Dữ liệu đã tải lên:")
    st.write(df.head())

    # Bước 3: Xác định features (thay tên cột thật trong file bạn)
    features = ["Protein", "Salt", "Cacium"]   # <--- bạn phải đổi chỗ này
    if all(f in df.columns for f in features):
        X = df[features].values
        st.write("✅ Ma trận X (5 dòng đầu):")
        st.write(X[:5])
    else:
        st.error("⚠️ File Excel không có đủ các cột: " + str(features))

    # Bước 4: thêm code train/predict tại đây
    # model.fit(X, y) ...
else:
    st.warning("👉 Vui lòng upload file Excel để tiếp tục")

# 1. Chuẩn bị dữ liệu và mô hình
# ===========================
features = ["Protein", "Salt", "Cacium"]
target = "ion_CO3"

# Giả sử df là DataFrame của bạn
X = df[features].values
y = df[target].values

# Tách train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tạo các mô hình
models = {
    "Linear": LinearRegression(),
    "Linear + StandardScaler": make_pipeline(StandardScaler(), LinearRegression()),
    "Linear + MinMaxScaler": make_pipeline(MinMaxScaler(), LinearRegression()),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

# Huấn luyện tất cả mô hình
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test  = r2_score(y_test, y_test_pred)
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test  = mean_squared_error(y_test, y_test_pred)
    
    results.append({
        "Model": name,
        "R2_train": r2_train,
        "R2_test": r2_test,
        "MSE_train": mse_train,
        "MSE_test": mse_test
    })

results_df = pd.DataFrame(results)

# ===========================
# 2. Giao diện Streamlit
# ===========================
st.title("Dự đoán ion_CO3 từ Protein, Salt, Cacium")

st.subheader("Chọn mô hình dự đoán")
model_choice = st.selectbox("Mô hình", results_df["Model"].tolist())

st.subheader("Nhập thông số đầu vào")
protein = st.number_input("Protein", value=0.0)
salt    = st.number_input("Salt", value=0.0)
cacium  = st.number_input("Cacium", value=0.0)

# Khi nhấn nút dự đoán
if st.button("Dự đoán"):
    X_new = pd.DataFrame([[protein, salt, cacium]], columns=features)
    
    # Lấy mô hình đã train
    selected_model = models[model_choice]
    y_pred = selected_model.predict(X_new)[0]
    
    st.success(f"Dự đoán ion_CO3: {y_pred:.4f}")

# Hiển thị bảng đánh giá tất cả mô hình
st.subheader("Hiệu quả các mô hình trên train/test")
st.dataframe(results_df)
