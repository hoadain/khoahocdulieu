# Students Performance Prediction - Full Script

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Đọc dữ liệu
df = pd.read_csv(r"D:\dulieu\khoahocdulieu\bt\StudentsPerformance.csv")
df.columns = df.columns.str.replace(' ', '_')  # Đổi tên cột cho dễ xử lý

# 2. Khám phá dữ liệu
print("Thông tin dữ liệu:")
print(df.info())
print("\nMô tả dữ liệu:")
print(df.describe())

# 3. Vẽ phân phối điểm từng môn
subjects = ["math_score", "reading_score", "writing_score"]
for subject in subjects:
    sns.histplot(df[subject], kde=True)
    plt.title(f"Phân phối điểm {subject}")
    plt.xlabel("Điểm")
    plt.ylabel("Số lượng học sinh")
    plt.show()

# 4. Tiền xử lý dữ liệu
df_encoded = pd.get_dummies(df, drop_first=True)

# 5A. REGRESSION - Dự đoán điểm Toán
print("\n--- REGRESSION: Dự đoán điểm Toán ---")
X = df_encoded.drop("math_score", axis=1)
y = df_encoded["math_score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 5B. CLASSIFICATION - Phân loại mức điểm Toán
print("\n--- CLASSIFICATION: Phân loại điểm Toán ---")

# Tạo nhãn phân loại
def grade_label(score):
    if score < 60:
        return "Low"
    elif score < 80:
        return "Medium"
    else:
        return "High"

df["math_grade"] = df["math_score"].apply(grade_label)

# Vẽ biểu đồ mức điểm
sns.countplot(x="math_grade", data=df, order=["Low", "Medium", "High"])
plt.title("Phân bố mức điểm Toán")
plt.xlabel("Mức điểm")
plt.ylabel("Số học sinh")
plt.show()

# Encode dữ liệu
X_cls = pd.get_dummies(df.drop(["math_score", "math_grade", "reading_score", "writing_score"], axis=1), drop_first=True)
le = LabelEncoder()
y_cls = le.fit_transform(df["math_grade"])

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred_cls = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_cls))
print(classification_report(y_test, y_pred_cls, target_names=le.classes_))
