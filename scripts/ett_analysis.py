#%%===================================================
# １ データの理解とEDA
# ====================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%------------------------------------------
# 1-0 画像保存関数の作成
#--------------------------------------------
def save_plot_from_title(folder='../outputs'):
    """グラフタイトルをそのままファイル名にして保存"""
    os.makedirs(folder, exist_ok=True)
    
    # 現在のグラフタイトルを取得
    title = plt.gca().get_title()
    if not title:
        title = 'untitled_plot'
    
    # ファイルパスを組み立てて保存
    filename = f"{title}.png"
    path = os.path.join(folder, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {path}")

#%%------------------------------------------
# 1-1 データの読み込み
#--------------------------------------------
data = pd.read_csv("../data/ett.csv", parse_dates=["date"], encoding="utf-8")
data = data.sort_values('date')
num_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
print(data.head(10))
print(data.shape)

#%%------------------------------------------
# 1-2 基本統計量の確認と可視化
#--------------------------------------------

#基本統計量の確認
print(data.describe())        # 基本統計量
print(data.info())            # データ型や欠損情報
print(data.isnull().sum())    # 欠損値の合計

#基本統計量の可視化
plt.figure(figsize=(12, 6))
sns.boxplot(data=data[num_cols])
plt.title('Boxplot of Numeric Columns', fontsize=14)
plt.ylabel('Value')
plt.xticks(rotation=30) 
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
save_plot_from_title()
plt.show()

#%%------------------------------------------
# 1-3 目的変数の可視化
#--------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(data['date'], data['OT'], color='steelblue')
plt.title('Overview of OT (Oil Temperature)')
plt.xlabel('Date')
plt.ylabel('Oil Temperature (OT)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
save_plot_from_title()
plt.show()

#%%------------------------------------------
# 1-4 自己相関の可視化
#--------------------------------------------
plt.figure(figsize=(12, 8))

for col in num_cols:
    autocorr = [data[col].autocorr(lag=i) for i in range(1, 500)]
    plt.plot(range(1, 500), autocorr, label=col)

plt.title('Autocorrelation of All Variables')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
save_plot_from_title()
plt.show()

#%%------------------------------------------
# 1-5 変数の分布の可視化
#--------------------------------------------
sns.set_theme(style="whitegrid")

cols = 3
rows = -(-len(num_cols) // cols) 

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    if col not in data.columns:
        continue

    sns.histplot(data[col].dropna(), bins='fd', kde=True,
                 ax=axes[i], color='skyblue', edgecolor='black')

    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequency')

# 余ったグラフ枠を非表示
for ax in axes[len(num_cols):]:
    ax.set_visible(False)

plt.title('Distributions of Numeric Variables', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
save_plot_from_title()
plt.show()

#%%------------------------------------------
# 1-6 目的変数と説明変数の相関
#--------------------------------------------

#相関係数を求める
target_col = 'OT'
numeric_features = [col for col in num_cols if col != target_col]

corrs = data[numeric_features].corrwith(data[target_col], method='pearson')
corrs = corrs.sort_values(ascending=False)
print("Correlation with OT:\n", corrs)

#相関行列の作成
numeric_data = data.select_dtypes(include=['number'])
corr_matrix = numeric_data.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(
    corr_matrix,
    annot=True,             
    fmt='.2f',               
    cmap='coolwarm',
    vmin=-1, vmax=1,         
    square=True,           
    linewidths=0.5,     
    annot_kws={"size": 8}
)

plt.title('Correlation Matrix of Numeric Features', fontsize=16)
plt.tight_layout()
save_plot_from_title()
plt.show()

#%%------------------------------------------
# 1-7 周期性の可視化
#--------------------------------------------
from statsmodels.tsa.seasonal import seasonal_decompose

# 時系列分解
result = seasonal_decompose(data['OT'], model='additive', period=24*30)

# 分解された成分を取得
observed = result.observed
trend = result.trend
seasonal = result.seasonal
resid = result.resid

# 手動で4段プロット
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1.5, 1, 1.5]})

axes[0].plot(observed, color='steelblue')
axes[0].set_title('Original (Observed Data)', fontsize=13)
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(trend, color='orange')
axes[1].set_title('Trend (Long-term Movement)', fontsize=13)
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].plot(seasonal, color='green')
axes[2].set_title('Seasonal (Periodic Pattern)', fontsize=13)
axes[2].grid(True, linestyle='--', alpha=0.5)

axes[3].plot(resid, color='gray')
axes[3].set_title('Residual (Irregular / Random)', fontsize=13)
axes[3].grid(True, linestyle='--', alpha=0.5)

plt.xlabel('Time (hour index)')
plt.tight_layout(pad=2)
plt.show()

#%%------------------------------------------
# 1-8 日周期の可視化
#--------------------------------------------
from statsmodels.tsa.seasonal import seasonal_decompose

# 日周期で分解
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date').asfreq('H')
result = seasonal_decompose(data['OT'], model='additive', period=24)
trend = result.trend.dropna().reset_index()

# トレンド成分を「日ごと」に平均化
trend['hour'] = trend['date'].dt.hour
daily_trend = trend.groupby('hour')['trend'].mean()

# 可視化
plt.figure(figsize=(10, 5))
plt.plot(daily_trend.index, daily_trend.values, marker='o', color='darkorange')
plt.title('Average Hourly Trend Component (from Decomposition)', fontsize=14)
plt.xlabel('Hour of Day')
plt.ylabel('Average Trend Value')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
save_plot_from_title()
plt.show()

#%%===================================================
# 2 データの前処理と特徴量エンジニアリング
# ====================================================

#欠損値は無し
#フーリエ変数を入れた後に異常値処理を行うため、データの前処理は後述する
#異常値処理として「標準化」「クリップ」を行った


#%%===================================================
# 3 モデル設定とトレーニング
# ====================================================

#%%------------------------------------------
# 3-1 フーリエ変数の追加
#--------------------------------------------
import statsmodels.api as sm

#AICとBICを用いて、最適なフーリエ次数を求める

# 時間軸
t = np.arange(len(data))
y = data['OT']

# パラメータ設定
max_k = 10
periods = {
    "day": 24,
    "week":24*7,
    "month":24*30,
    "year": 24*365
}

# 結果保存用
results = []

# 各周期ごとに最適kを探索
for name, period in periods.items():
    aic_list = []
    bic_list = []
    
    for k in range(1, max_k + 1):
        # フーリエ変数生成
        X = pd.DataFrame({
            f'{name}_sin{k}': np.sin(2 * np.pi * k * t / period),
            f'{name}_cos{k}': np.cos(2 * np.pi * k * t / period)
        })
        X.index = y.index
        X = sm.add_constant(X)

        # 回帰
        model = sm.OLS(y, X).fit()

        # 結果保存
        aic_list.append(model.aic)
        bic_list.append(model.bic)

    # DataFrame化
    tmp_df = pd.DataFrame({
        'k': range(1, max_k + 1),
        'AIC': aic_list,
        'BIC': bic_list
    })
    
    # 最小AIC/BIC
    best_k_aic = tmp_df.loc[tmp_df['AIC'].idxmin(), 'k']
    best_k_bic = tmp_df.loc[tmp_df['BIC'].idxmin(), 'k']

    results.append({
        'Period': name,
        'Best_k_AIC': best_k_aic,
        'Best_k_BIC': best_k_bic
    })

    # 可視化
    plt.figure(figsize=(7,4))
    plt.plot(tmp_df['k'], tmp_df['AIC'], marker='o', label='AIC')
    plt.plot(tmp_df['k'], tmp_df['BIC'], marker='s', label='BIC')
    plt.title(f'AIC/BIC by Fourier Order (Period = {name})')
    plt.xlabel('Fourier Order k')
    plt.ylabel('Information Criterion')
    plt.legend()
    plt.grid(True)
    plt.show()

# 全周期の結果まとめ
result_df = pd.DataFrame(results)

print("=== 日周期と年周期おける最適フーリエ次数 ===")
print(result_df)

#最適なkの設定
best_k = {
    "day": 2, #1で最小だったが、1の時に自己相関に日周期が残っていたため2で設定
    "week":1, #7で最小だったが、多重共線性への懸念から1に設定
    "month":2,
    "year": 1
}

#期間の設定
periods = {
    "day": 24,
    "week":24*7,
    "month":24*30,
    "year": 24*365
}

# data_fourier_ts の作成-
data_fourier_ts = pd.DataFrame()
data_fourier_ts.index = data.index

# 時間軸
t = np.arange(len(data))

# フーリエ変数を data_fourier_ts に追加
for name, period in periods.items():
    k_opt = best_k[name]
    for k in range(1, k_opt + 1):
        data_fourier_ts[f'{name}_sin{k}'] = np.sin(2 * np.pi * k * t / period)
        data_fourier_ts[f'{name}_cos{k}'] = np.cos(2 * np.pi * k * t / period)

print("✅ data_fourier_ts にフーリエ変数を追加しました")
print([col for col in data_fourier_ts.columns if any(x in col for x in ['sin', 'cos'])])

#%%------------------------------------------
# 3-2 時系列構造化
#--------------------------------------------

# 時系列特徴量の作成
ts_features = pd.DataFrame(index=data.index)
ts_features['OT_lag1'] = data['OT'].shift(1)
ts_features['OT_lag24'] = data['OT'].shift(24)
ts_features['OT_ma24'] = data['OT'].shift(1).rolling(window=24).mean()

#data_fourier_tsに統合する
data_fourier_ts = pd.concat([data_fourier_ts, ts_features], axis=1)

data_fourier_ts = data_fourier_ts.loc[:, ~data_fourier_ts.columns.duplicated()]


#確認
print(data_fourier_ts.columns)

#%%------------------------------------------
# 3-3 データの標準化
#--------------------------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 標準化対象の列を定義
cols_from_data = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
cols_from_ts_features = ['OT_lag1', 'OT_lag24','OT_ma24',]

# 外生変数を標準化
scaled_data_part = pd.DataFrame(
    scaler.fit_transform(data[cols_from_data]),
    columns=[f"{c}_scaled" for c in cols_from_data],
    index=data.index
)

# 時系列データを標準化 ---
scaled_ts_part = pd.DataFrame(
    scaler.fit_transform(data_fourier_ts[cols_from_ts_features]),
    columns=[f"{c}_scaled" for c in cols_from_ts_features],
    index=data_fourier_ts.index
)

# --- 標準化後の外れ値を ±2 にクリップ ---
scaled_data_part = scaled_data_part.clip(lower=-2, upper=2)
scaled_ts_part = scaled_ts_part.clip(lower=-2, upper=2)

# 結果を結合
data_scaled = scaled_data_part.join(scaled_ts_part, how='left')
data_fourier_ts = data_fourier_ts.join(data_scaled, how='left')

# 欠損を含む行を削除
data_fourier_ts = data_fourier_ts.dropna(subset=ts_features.columns)

#確認
print("✅ 欠損を削除後のデータ形状:", data_fourier_ts.shape)

# 確認
print(data_fourier_ts.columns)
print(data_fourier_ts.head())

#%%------------------------------------------
# 3-4 フーリエ変数と時系列データのみで回帰分析
#--------------------------------------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 特徴量と目的変数
X = data_fourier_ts[[
    'day_sin1','day_cos1',
    'year_sin1','year_cos1',
    'OT_ma24_scaled'
]]
y = data_fourier_ts['OT_scaled']

# データ分割（時系列順に8:2）
split_point = int(len(data_fourier_ts) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# モデル学習（trainのみ）
model = LinearRegression()
model.fit(X_train, y_train)

# 予測
data_fourier_ts.loc[X_train.index, 'OT_pred_linear'] = model.predict(X_train)
data_fourier_ts.loc[X_test.index, 'OT_pred_linear'] = model.predict(X_test)

# 学習データでの性能
r2_train = r2_score(y_train, data_fourier_ts.loc[X_train.index, 'OT_pred_linear'])
rmse_train = np.sqrt(mean_squared_error(y_train, data_fourier_ts.loc[X_train.index, 'OT_pred_linear']))

# 検証データでの性能
r2_test = r2_score(y_test, data_fourier_ts.loc[X_test.index, 'OT_pred_linear'])
rmse_test = np.sqrt(mean_squared_error(y_test, data_fourier_ts.loc[X_test.index, 'OT_pred_linear']))

print("✅  時系列8:2分割")
print(f"🔹 R²(学習): {r2_train:.4f}")
print(f"🔹 RMSE(学習): {rmse_train:.4f}")
print(f"🔹 R²(検証): {r2_test:.4f}")
print(f"🔹 RMSE(検証): {rmse_test:.4f}")


#%%------------------------------------------
# 3-5 Ridge回帰分析
#--------------------------------------------
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 残差を作成（一次モデルからの誤差）
data_fourier_ts['residual_linear'] = data_fourier_ts['OT_scaled'] - data_fourier_ts['OT_pred_linear']

# 特徴量（_scaledで終わる列だけを使用）
scaled_cols = [col for col in data_fourier_ts.columns if col.endswith('_scaled')]
X_ridge = data_fourier_ts[['HUFL_scaled', 'HULL_scaled', 'MUFL_scaled', 'MULL_scaled', 'LUFL_scaled', 'LULL_scaled']]
y_ridge = data_fourier_ts['residual_linear']

# 時系列分割（前8割で学習、後2割で検証）
split_point = int(len(data_fourier_ts) * 0.8)
X_train, X_test = X_ridge.iloc[:split_point], X_ridge.iloc[split_point:]
y_train, y_test = y_ridge.iloc[:split_point], y_ridge.iloc[split_point:]

# Ridge回帰モデルを定義・学習
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# train/test それぞれで予測
data_fourier_ts.loc[X_train.index, 'OT_pred_residual'] = ridge_model.predict(X_train)
data_fourier_ts.loc[X_test.index, 'OT_pred_residual'] = ridge_model.predict(X_test)

# 評価（参考）
r2_train = r2_score(y_train, data_fourier_ts.loc[X_train.index, 'OT_pred_residual'])
r2_test = r2_score(y_test, data_fourier_ts.loc[X_test.index, 'OT_pred_residual'])
rmse_train = np.sqrt(mean_squared_error(y_train, data_fourier_ts.loc[X_train.index, 'OT_pred_residual']))
rmse_test = np.sqrt(mean_squared_error(y_test, data_fourier_ts.loc[X_test.index, 'OT_pred_residual']))

print("✅ Ridge回帰分析完了")
print(f"🔹 R²(学習): {r2_train:.4f}")
print(f"🔹 R²(検証): {r2_test:.4f}")
print(f"🔹 RMSE(学習): {rmse_train:.4f}")
print(f"🔹 RMSE(検証): {rmse_test:.4f}")

#%%===================================================
# 4 モデルの評価と結果の分析
# ====================================================

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
#%%------------------------------------------
# 4-1 最終予測値の算出
#--------------------------------------------

# 前半8割を学習用データ、後半2割を予測用データとして利用する
# データ分割
split_point = int(len(data_fourier_ts) * 0.8)
train = data_fourier_ts.iloc[:split_point].copy()
test = data_fourier_ts.iloc[split_point:].copy()
test['OT_pred_final'] = test['OT_pred_linear'] + test['OT_pred_residual']

#　モデル評価
r2_final = r2_score(test['OT_scaled'], test['OT_pred_final'])
rmse_final = np.sqrt(mean_squared_error(test['OT_scaled'], test['OT_pred_final']))
mae_final = mean_absolute_error(test['OT_scaled'], test['OT_pred_final'])

print("✅ 最終モデル評価(検証データ2割)完了")
print(f"決定係数 R²(検証): {r2_final:.4f}")
print(f"RMSE(検証): {rmse_final:.4f}")
print(f"MAE(検証): {mae_final:.4f}")

#%%------------------------------------------
# 4-2 最終予測値の可視化
#--------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(test.index, test['OT_scaled'], label='Actual', linewidth=1)
plt.plot(test.index, test['OT_pred_final'], label='Predicted (Final)', linestyle='--', linewidth=1.2)
plt.title('Actual vs Predicted (Final Model)')
plt.xlabel('Date')
plt.ylabel('OT_scaled')
plt.legend()
plt.grid(True)
plt.show()

#%%------------------------------------------
# 4-3 残差の推移の可視化
#--------------------------------------------

plt.figure(figsize=(10, 3))
plt.plot(test.index, test['residual_linear'], label='Residual (Before Ridge)', color='gray', alpha=0.7)
plt.plot(test.index, test['OT_pred_residual'], label='Residual (Predicted by Ridge)', color='blue', alpha=0.7)
plt.title('Residuals Before and After Ridge Correction')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.show()


#%%------------------------------------------
# 4-4 残差の自己相関・偏自己相関の可視化
#--------------------------------------------
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ridge補正後の残差を使用
residuals = data_fourier_ts['OT_pred_residual']  # 自分の列名に合わせて変更！

# ACF（自己相関関数）
plt.figure(figsize=(10, 4))
plot_acf(residuals, lags=50)
plt.title("Autocorrelation Function (ACF) of Residuals")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.grid(True)
plt.tight_layout()
plt.show()

# PACF（偏自己相関関数）
plt.figure(figsize=(10, 4))
plot_pacf(residuals, lags=50, method='ywm')
plt.title("Partial Autocorrelation Function (PACF) of Residuals")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.grid(True)
plt.tight_layout()
plt.show()

#DW検定の実行
from statsmodels.stats.stattools import durbin_watson
dw_stat = durbin_watson(data_fourier_ts['residual_linear'])
print('Durbin-Watson 検定統計量:', dw_stat)


#%%------------------------------------------
# 4-5 モデルの特徴量重要度の可視化
#--------------------------------------------

# フーリエ変数と時系列データのみで回帰分析の重要度
coef_linear = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
}).sort_values('coefficient', ascending=False)

print("🔹【線形回帰モデルの特徴量重要度】")
print(coef_linear)

plt.figure(figsize=(6,4))
plt.barh(coef_linear['feature'], coef_linear['coefficient'])
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficient')
plt.gca().invert_yaxis()
plt.show()


# Ridge回帰モデルの重要度
coef_ridge = pd.DataFrame({
    'feature': X_ridge.columns,
    'coefficient': ridge_model.coef_
}).sort_values('coefficient', ascending=False)

print("\n🔹【Ridge回帰モデルの特徴量重要度】")
print(coef_ridge.head(15))  # 上位15件のみ表示

plt.figure(figsize=(8,6))
plt.barh(coef_ridge['feature'].head(15), coef_ridge['coefficient'].head(15))
plt.title('Ridge Regression Coefficients (Top 15)')
plt.xlabel('Coefficient')
plt.gca().invert_yaxis()
plt.show()

#%%------------------------------------------
# 4-6 回帰モデルのp値・有意性の確認
#--------------------------------------------
import statsmodels.api as sm
import pandas as pd

# === ① 線形回帰モデル（フーリエ＋時系列変数） ===
X_train_sm = sm.add_constant(X_train)  # 切片を追加
ols_model = sm.OLS(y_train, X_train_sm).fit()

print("🔹【線形回帰モデル：統計的要約】")
print(ols_model.summary())

# 主要な指標だけ抜き出す表
summary_linear = pd.DataFrame({
    "feature": ['const'] + list(X_train.columns),
    "coef": ols_model.params.values,
    "std_err": ols_model.bse.values,
    "t_value": ols_model.tvalues.values,
    "p_value": ols_model.pvalues.values
})
print("\n【線形回帰モデル 係数・p値一覧】")
print(summary_linear.sort_values("p_value").head(10))



# Ridge回帰モデル

X_ridge_sm = sm.add_constant(X_train)
ols_ridge_like = sm.OLS(y_train, X_ridge_sm).fit()

print("\n🔹【Ridge回帰モデル相当の統計要約】")
print(ols_ridge_like.summary())

summary_ridge = pd.DataFrame({
    "feature": ['const'] + list(X_train.columns),
    "coef": ols_ridge_like.params.values,
    "std_err": ols_ridge_like.bse.values,
    "t_value": ols_ridge_like.tvalues.values,
    "p_value": ols_ridge_like.pvalues.values
})
print("\n【Ridge相当モデル 係数・p値一覧】")
print(summary_ridge.sort_values("p_value").head(10))

#%% ===================================================
# 5 改善策の検討とモデルの再トレーニング
# =====================================================

#①週周期と月周期も変数に設定する
#②一日前、一週間前、過去一週間の移動平均も変数に設定
#多重共線性が懸念されるので、有意性が低い変数は除外

#%%------------------------------------------
# 5-1 変数の再設定とフーリエ変数＋時系列データの回帰分析
#--------------------------------------------
X2 = data_fourier_ts[[
    'day_sin1', 'day_cos1','day_sin2','day_cos2',
    'week_sin1', 'week_cos1',
    'month_sin1', 'month_cos1',
    'year_sin1', 'year_cos1',
    'OT_ma24_scaled'
    ]]

y2 = data_fourier_ts['OT_scaled']

#相関係数を調べる
r1 = data_fourier_ts['OT_lag1_scaled'].corr(data_fourier_ts['OT_lag24_scaled'])
r2 = data_fourier_ts['OT_lag1_scaled'].corr(data_fourier_ts['OT_ma24_scaled'])
r3 = data_fourier_ts['OT_lag24_scaled'].corr(data_fourier_ts['OT_ma24_scaled'])

print(f"OT_lag1_scaled と OT_lag24_scaled の相関係数: {r1:.3f}")
print(f"OT_lag1_scaled と OT_ma24_scaled の相関係数: {r2:.3f}")
print(f"OT_lag24_scaled と OT_ma24_scaled の相関係数: {r3:.3f}")

# データ分割（時系列順に8:2）
split_point = int(len(data_fourier_ts) * 0.8)
X2_train, X2_test = X2.iloc[:split_point], X2.iloc[split_point:]
y2_train, y2_test = y2.iloc[:split_point], y2.iloc[split_point:]

# モデル学習（trainのみ）
model2 = LinearRegression()
model2.fit(X2_train, y2_train)

# 予測
data_fourier_ts.loc[X2_train.index, 'OT_pred_linear'] = model2.predict(X2_train)
data_fourier_ts.loc[X2_test.index, 'OT_pred_linear'] = model2.predict(X2_test)

# 学習データでの性能
r2_train2 = r2_score(y2_train, data_fourier_ts.loc[X2_train.index, 'OT_pred_linear'])
rmse_train2 = np.sqrt(mean_squared_error(y2_train, data_fourier_ts.loc[X2_train.index, 'OT_pred_linear']))

# 検証データでの性能
r2_test2 = r2_score(y2_test, data_fourier_ts.loc[X2_test.index, 'OT_pred_linear'])
rmse_test2 = np.sqrt(mean_squared_error(y2_test, data_fourier_ts.loc[X2_test.index, 'OT_pred_linear']))

print("✅  時系列8:2分割")
print(f"🔹 R²(訓練): {r2_train2:.4f}")
print(f"🔹 RMSE(訓練): {rmse_train2:.4f}")
print(f"🔹 R²(検証): {r2_test2:.4f}")
print(f"🔹 RMSE(検証): {rmse_test2:.4f}")

#%%------------------------------------------
# 5-2 Ridge回帰分析
#--------------------------------------------

# 残差を作成（一次モデルからの誤差）
data_fourier_ts['residual_linear'] = data_fourier_ts['OT_scaled'] - data_fourier_ts['OT_pred_linear']

# 特徴量
X2_ridge = data_fourier_ts[['HUFL_scaled', 'HULL_scaled','MUFL_scaled','LUFL_scaled']] #MULL_scaledとLULL_scaledを除外
y2_ridge = data_fourier_ts[['residual_linear']]

# 時系列分割
split_point = int(len(data_fourier_ts) * 0.8)
X2_train, X2_test = X2_ridge.iloc[:split_point], X2_ridge.iloc[split_point:]
y2_train, y2_test = y2_ridge.iloc[:split_point], y2_ridge.iloc[split_point:]

# Ridge回帰モデルを定義・学習
ridge_model2 = Ridge(alpha=1.0)
ridge_model2.fit(X2_train, y2_train)

# train/test それぞれで予測
data_fourier_ts.loc[X2_train.index, 'OT_pred_residual'] = ridge_model2.predict(X2_train)
data_fourier_ts.loc[X2_test.index, 'OT_pred_residual'] = ridge_model2.predict(X2_test)

# 評価（参考）
r2_train2   = r2_score(y2_train, data_fourier_ts.loc[X2_train.index, 'OT_pred_residual'])
r2_test2    = r2_score(y2_test, data_fourier_ts.loc[X2_test.index, 'OT_pred_residual'])
rmse_train2 = np.sqrt(mean_squared_error(y2_train, data_fourier_ts.loc[X2_train.index, 'OT_pred_residual']))
rmse_test2  = np.sqrt(mean_squared_error(y2_test, data_fourier_ts.loc[X2_test.index, 'OT_pred_residual']))

print("✅ Ridge回帰分析2完了")
print(f"🔹 R²(訓練): {r2_train2:.4f}")
print(f"🔹 R²(検証): {r2_test2:.4f}")
print(f"🔹 RMSE(訓練): {rmse_train2:.4f}")
print(f"🔹 RMSE(検証): {rmse_test:.4f}")

#%%------------------------------------------
# 5-3 最終予測値の算出
#--------------------------------------------

# 前半8割を学習用データ、後半2割を予測用データとして利用する
# データ分割
split_point2 = int(len(data_fourier_ts) * 0.8)
train2 = data_fourier_ts.iloc[:split_point2].copy()
test2 = data_fourier_ts.iloc[split_point2:].copy()
test2['OT_pred_final'] = test2['OT_pred_linear'] + test2['OT_pred_residual']

#　モデル評価
r2_final2 = r2_score(test2['OT_scaled'], test2['OT_pred_final'])
rmse_final2 = np.sqrt(mean_squared_error(test2['OT_scaled'], test2['OT_pred_final']))
mae_final2 = mean_absolute_error(test2['OT_scaled'], test2['OT_pred_final'])

print("✅ 最終モデル評価(検証データ2割)完了")
print(f"決定係数 R²(検証): {r2_final2:.4f}")
print(f"RMSE(検証): {rmse_final2:.4f}")
print(f"MAE(検証): {mae_final2:.4f}")

#%%------------------------------------------
# 5-4 最終予測値の可視化
#--------------------------------------------

plt.figure(figsize=(10, 4))
plt.plot(test2.index, test2['OT_scaled'], label='Actual', linewidth=1)
plt.plot(test2.index, test2['OT_pred_final'], label='Predicted (Final)', linestyle='--', linewidth=1.2)
plt.title('Actual vs Predicted (Final Model)')
plt.xlabel('Date')
plt.ylabel('OT_scaled')
plt.legend()
plt.grid(True)
plt.show()

#%%------------------------------------------
# 5-5 残差の推移の可視化
#--------------------------------------------

plt.figure(figsize=(10, 3))
plt.plot(test2.index, test2['residual_linear'], label='Residual (Before Ridge)', color='gray', alpha=0.7)
plt.plot(test2.index, test2['OT_pred_residual'], label='Residual (Predicted by Ridge)', color='blue', alpha=0.7)
plt.title('Residuals Before and After Ridge Correction')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.legend()
plt.grid(True)
plt.show()


#%%------------------------------------------
# 5-6 残差の自己相関・偏自己相関の可視化
#--------------------------------------------

# Ridge補正後の残差を使用
residuals2 = data_fourier_ts['OT_pred_residual']  # 自分の列名に合わせて変更！

# ACF（自己相関関数）
plt.figure(figsize=(10, 4))
plot_acf(residuals2, lags=50)
plt.title("Autocorrelation Function (ACF) of Residuals")
plt.xlabel("Lag")
plt.ylabel("ACF")
plt.grid(True)
plt.tight_layout()
plt.show()

# PACF（偏自己相関関数）
plt.figure(figsize=(10, 4))
plot_pacf(residuals2, lags=50, method='ywm')
plt.title("Partial Autocorrelation Function (PACF) of Residuals")
plt.xlabel("Lag")
plt.ylabel("PACF")
plt.grid(True)
plt.tight_layout()
plt.show()

#DW検定の実行
from statsmodels.stats.stattools import durbin_watson
dw_stat2 = durbin_watson(data_fourier_ts['residual_linear'])
print('Durbin-Watson 検定統計量:', dw_stat2)


#%%------------------------------------------
# 5-7 モデルの特徴量重要度の可視化
#--------------------------------------------

# フーリエ変数と時系列データのみで回帰分析の重要度
coef_linear2 = pd.DataFrame({
    'feature': X2.columns,
    'coefficient': model2.coef_
}).sort_values('coefficient', ascending=False)

print("🔹【線形回帰モデルの特徴量重要度】")
print(coef_linear2)

plt.figure(figsize=(6,4))
plt.barh(coef_linear2['feature'], coef_linear2['coefficient'])
plt.title('Linear Regression Coefficients')
plt.xlabel('Coefficient')
plt.gca().invert_yaxis()
plt.show()


# Ridge回帰モデルの重要度
coef_ridge2 = pd.DataFrame({
    'feature': X2_ridge.columns,
    'coefficient': ridge_model2.coef_
}).sort_values('coefficient', ascending=False)

print("\n🔹【Ridge回帰モデルの特徴量重要度】")
print(coef_ridge2.head()) 

plt.figure(figsize=(8,6))
plt.barh(coef_ridge2['feature'].head(), coef_ridge2['coefficient'].head(15))
plt.title('Ridge Regression Coefficients')
plt.xlabel('Coefficient')
plt.gca().invert_yaxis()
plt.show()

#%%------------------------------------------
# 5-8 回帰モデルのp値・有意性の確認
#--------------------------------------------
import statsmodels.api as sm
import pandas as pd

# === ① 線形回帰モデル（フーリエ＋時系列変数） ===
X2_train_sm = sm.add_constant(X2_train)  # 切片を追加
ols_model2 = sm.OLS(y2_train, X2_train_sm).fit()

print("🔹【線形回帰モデル：統計的要約】")
print(ols_model2.summary())

# 主要な指標だけ抜き出す表
summary_linear2 = pd.DataFrame({
    "feature": ['const'] + list(X2_train.columns),
    "coef": ols_model2.params.values,
    "std_err": ols_model2.bse.values,
    "t_value": ols_model2.tvalues.values,
    "p_value": ols_model2.pvalues.values
})
print("\n【線形回帰モデル 係数・p値一覧】")
print(summary_linear2.sort_values("p_value").head(10))



# Ridge回帰モデル
X2_ridge_sm = sm.add_constant(X2_train)
ols_ridge_like2 = sm.OLS(y2_train, X2_ridge_sm).fit()

print("\n🔹【Ridge回帰モデル相当の統計要約】")
print(ols_ridge_like2.summary())

summary_ridge2 = pd.DataFrame({
    "feature": ['const'] + list(X2_train.columns),
    "coef": ols_ridge_like2.params.values,
    "std_err": ols_ridge_like2.bse.values,
    "t_value": ols_ridge_like2.tvalues.values,
    "p_value": ols_ridge_like2.pvalues.values
})
print("\n【Ridge相当モデル 係数・p値一覧】")
print(summary_ridge2.sort_values("p_value").head())
#%%
