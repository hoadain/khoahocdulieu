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

# 2. Vẽ biểu đồ phân phối điểm từng môn
subjects = ["math_score", "reading_score", "writing_score"]
for subject in subjects:
    sns.histplot(df[subject], kde=True, bins=20)
    plt.title(f"Phân phối điểm {subject}")
    plt.xlabel("Điểm")
    plt.ylabel("Số lượng học sinh")
    plt.show()

# 3. Tiền xử lý cho Regression
df_encoded = pd.get_dummies(df, drop_first=True)

# 4. Hồi quy (Regression) cho từng môn
for subject in subjects:
    print(f"\n--- REGRESSION: Dự đoán {subject} ---")
    X = df_encoded.drop(subject, axis=1)
    y = df_encoded[subject]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    y_pred = reg_model.predict(X_test)

    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))


# 5. Tạo nhãn phân loại theo phần vị (Low, Medium, High)
def dynamic_grade_label(score, q1, q3):
    if score < q1:
        return "Low"
    elif score < q3:
        return "Medium"
    else:
        return "High"


for subject in subjects:
    q1 = df[subject].quantile(0.25)
    q3 = df[subject].quantile(0.75)
    grade_col = subject.replace("score", "grade")
    df[grade_col] = df[subject].apply(lambda x: dynamic_grade_label(x, q1, q3))

# 6. Vẽ biểu đồ phân loại theo từng môn
for grade_col in ["math_grade", "reading_grade", "writing_grade"]:
    sns.countplot(x=grade_col, data=df, order=["Low", "Medium", "High"])
    plt.title(f"Phân bố mức điểm {grade_col}")
    plt.xlabel("Mức điểm")
    plt.ylabel("Số học sinh")
    plt.show()

# 7. Phân loại (Classification) cho từng môn
for subject in subjects:
    grade_col = subject.replace("score", "grade")
    print(f"\n--- CLASSIFICATION: Phân loại {grade_col} ---")

    # Chuẩn bị dữ liệu
    X_cls = pd.get_dummies(df.drop([subject, grade_col] + [s for s in subjects if s != subject], axis=1),
                           drop_first=True)
    le = LabelEncoder()
    y_cls = le.fit_transform(df[grade_col])

    X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_cls = clf.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred_cls))
    print(classification_report(y_test, y_pred_cls, target_names=le.classes_))
