# ============================================================
#  Titanic — Stacking Ensemble ML
#  Base Models : Random Forest, SVM, XGBoost
#  Meta Model  : Logistic Regression
# ============================================================

# ── 1. ติดตั้ง library (รันครั้งแรกครั้งเดียว) ──────────────
# pip install xgboost scikit-learn pandas numpy matplotlib seaborn joblib

# ── 2. Import ───────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


# ── 3. โหลด Dataset ─────────────────────────────────────────
print("=" * 50)
print("📦 โหลด Dataset...")
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print(f"Shape: {df.shape}")
print("\n🔍 Missing Values ก่อนเตรียมข้อมูล:")
print(df.isnull().sum()[df.isnull().sum() > 0])


# ── 4. Data Preparation ─────────────────────────────────────
print("\n" + "=" * 50)
print("🔧 เตรียมข้อมูล...")

# Drop คอลัมน์ที่ไม่จำเป็น
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# จัดการ Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# Cabin → ดึงตัวอักษรแรก หรือ 'Unknown'
df['Cabin'] = df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'Unknown')

# Encode Categorical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
df['Cabin'] = df['Cabin'].astype('category').cat.codes

# Feature Engineering
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print("✅ Missing Values หลังเตรียมข้อมูล:")
print(df.isnull().sum()[df.isnull().sum() > 0] if df.isnull().sum().any() else "  ไม่มี missing values แล้ว!")
print(f"Shape หลังเตรียมข้อมูล: {df.shape}")


# ── 5. Visualize ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Titanic — Data Exploration', fontsize=14)

df.groupby('Sex')['Survived'].mean().plot(
    kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'])
axes[0].set_title('อัตรารอดชีวิตตามเพศ')
axes[0].set_xticklabels(['Male', 'Female'], rotation=0)
axes[0].set_ylabel('Survival Rate')

df.groupby('Pclass')['Survived'].mean().plot(
    kind='bar', ax=axes[1], color=['gold', 'silver', '#cd7f32'])
axes[1].set_title('อัตรารอดชีวิตตาม Class')
axes[1].set_xticklabels(['1st', '2nd', '3rd'], rotation=0)

df['Age'].hist(ax=axes[2], bins=20, color='#2ecc71', edgecolor='white')
axes[2].set_title('การกระจายของอายุ')
axes[2].set_xlabel('Age')

plt.tight_layout()
plt.savefig('titanic_eda.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → titanic_eda.png")


# ── 6. แบ่ง Train / Test ─────────────────────────────────────
print("\n" + "=" * 50)
print("✂️  แบ่ง Train/Test...")

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"Train size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")


# ── 7. สร้าง Stacking Ensemble ───────────────────────────────
print("\n" + "=" * 50)
print("🤖 สร้าง Stacking Ensemble Model...")

base_models = [
    ('random_forest', RandomForestClassifier(
        n_estimators=100, max_depth=6, random_state=42)),
    ('svm', SVC(
        kernel='rbf', C=1.0, probability=True, random_state=42)),
    ('xgboost', XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        random_state=42, eval_metric='logloss'))
]

meta_model = LogisticRegression(random_state=42)

stacking_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    passthrough=False
)

stacking_model.fit(X_train_scaled, y_train)
print("✅ เทรนโมเดลเสร็จแล้ว!")


# ── 8. ประเมินผล ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("📊 ผลการประเมิน Stacking Ensemble:")

y_pred = stacking_model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy : {acc:.4f} ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['ไม่รอดชีวิต', 'รอดชีวิต']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['ไม่รอด', 'รอดชีวิต'],
            yticklabels=['ไม่รอด', 'รอดชีวิต'])
plt.title('Confusion Matrix — Stacking Ensemble')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('titanic_confusion_matrix.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → titanic_confusion_matrix.png")


# ── 9. เปรียบเทียบแต่ละโมเดล ────────────────────────────────
print("\n" + "=" * 50)
print("📈 เปรียบเทียบ Accuracy แต่ละโมเดล:")

individual_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'          : SVC(kernel='rbf', probability=True, random_state=42),
    'XGBoost'      : XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
}

results = {}
for name, model in individual_models.items():
    model.fit(X_train_scaled, y_train)
    score = accuracy_score(y_test, model.predict(X_test_scaled))
    results[name] = score
    print(f"  {name:<20}: {score:.4f} ({score*100:.2f}%)")

results['Stacking Ensemble'] = acc
print(f"  {'Stacking Ensemble':<20}: {acc:.4f} ({acc*100:.2f}%)")

# Plot
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
plt.figure(figsize=(8, 4))
bars = plt.bar(results.keys(), results.values(), color=colors)
plt.ylim(0.7, 1.0)
plt.title('เปรียบเทียบ Accuracy แต่ละโมเดล')
plt.ylabel('Accuracy')
plt.xticks(rotation=15)
for bar, val in zip(bars, results.values()):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.005,
             f'{val:.4f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('titanic_model_comparison.png', dpi=150)
plt.show()
print("💾 บันทึกกราฟ → titanic_model_comparison.png")


# ── 10. บันทึกโมเดล ──────────────────────────────────────────
print("\n" + "=" * 50)
print("💾 บันทึกโมเดล...")

joblib.dump(stacking_model, 'titanic_stacking_model.pkl')
joblib.dump(scaler,         'titanic_scaler.pkl')

print("✅ บันทึกเสร็จแล้ว!")
print("   → titanic_stacking_model.pkl")
print("   → titanic_scaler.pkl")
print("\n🎉 เสร็จสิ้นทุกขั้นตอน!")
