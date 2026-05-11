"""МКР з Python for Data Science — наскрізний кейс «Метеослужба».

ШАБЛОН ДЛЯ СТУДЕНТА. Заповніть кожен пункт у блоках 1–4 та допишіть
ВИСНОВКИ у docstring наприкінці файлу.

Перед запуском скрипта підніміть СВІЙ Docker-контейнер з MySQL:

    docker pull <DOCKER_USER>/pfds-mkr-g<N>-<NN>
    docker run -d -p 3306:3306 --name mkr <DOCKER_USER>/pfds-mkr-g<N>-<NN>

(g<N>-<NN> — ваші група і номер у журналі, видається викладачем)

Потім чекайте ~30 секунд на ініціалізацію MySQL і запускайте:

    python solution.py

Графіки зберігаються в підпапку `plots/` поряд зі скриптом.
"""

# ====================================================================
# Прізвище, ім'я, по батькові: Анохіна Ксенія Вячеславівна
# Група:                       ЗК-31 1 варіант
# Дата виконання:              06.05.2026
# ====================================================================

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "meteo"

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_observations(retries: int = 12, delay: float = 2.5) -> pd.DataFrame:
    """Підключитися до MySQL і завантажити таблицю observations.

    MySQL-контейнер на старті виконує LOAD DATA INFILE, що займає
    ~20–30 секунд. Тому робимо retry-цикл — перші спроби очікувано
    падають з OperationalError (server not ready).
    """
    url = (
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
        f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    engine = create_engine(url)
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_sql("SELECT * FROM observations", engine)
            print(f"Підключено до MySQL з {attempt}-ї спроби. Рядків: {len(df)}")
            return df
        except OperationalError:
            if attempt == retries:
                raise
            print(f"  MySQL ще не готова (спроба {attempt}/{retries})...")
            time.sleep(delay)
    raise RuntimeError("Unreachable")


# ====================================================================
# БЛОК 1. NumPy (15 балів)
# ====================================================================

def block_1_numpy(df_raw: pd.DataFrame) -> None:
    section("БЛОК 1. NumPy")

    # Конвертація в numpy array
    t = df_raw['temperature_c'].to_numpy()
    rh = df_raw['humidity_pct'].to_numpy()
    wind = df_raw['wind_speed_ms'].to_numpy()

    # 1) Побудувати np.array apparent temperature
    apparent = t - (100 - rh) / 5
    print(f"1) T_app: len={len(apparent)}, min={np.nanmin(apparent):.2f}, max={np.nanmax(apparent):.2f}")

    # 2) Замінити викидні значення:
    temperature_clean = np.where((t > 60) | (t < -60), np.nan, t)
    wind_clean = np.where(wind > 100, np.nan, wind)

    t_outliers = np.sum((t > 60) | (t < -60))
    w_outliers = np.sum(~np.isnan(wind) & (wind > 100))
    print(f"2) Викидів температури замінено: {t_outliers}")
    print(f"   Викидів вітру замінено:       {w_outliers}")

    # 3) Порахувати mean / median / std температури ВРУЧНУ
    valid_t = temperature_clean[~np.isnan(temperature_clean)]
    mean_t = np.nansum(valid_t) / len(valid_t)
    median_t = np.nanmedian(valid_t)
    std_t = np.sqrt(np.nansum((valid_t - mean_t) ** 2) / len(valid_t))
    print(f"3) mean={mean_t:.3f}  median={median_t:.3f}  std={std_t:.3f}")

    # 4) Маска: скільки спостережень "морозних" (T<0) і "жарких" (T>30).
    n_frost = np.sum(valid_t < 0)
    n_hot = np.sum(valid_t > 30)
    print(f"4) морозних: {n_frost}    жарких: {n_hot}")

    # 5) argmax / argmin температури
    max_idx = np.nanargmax(temperature_clean)
    min_idx = np.nanargmin(temperature_clean)
    print(f"5) Max T: obs_id={df_raw['obs_id'].iloc[max_idx]}, datetime={df_raw['datetime'].iloc[max_idx]}")
    print(f"   Min T: obs_id={df_raw['obs_id'].iloc[min_idx]}, datetime={df_raw['datetime'].iloc[min_idx]}")


# ====================================================================
# БЛОК 2. Pandas — очищення (20 балів)
# ====================================================================

def block_2_cleaning(df_raw: pd.DataFrame) -> pd.DataFrame:
    section("БЛОК 2. Pandas — очищення")

    rows_before = len(df_raw)
    df = df_raw.copy()

    # 1) Перевірте типи (info), статистику (describe).
    print("1) Інформація про DataFrame:")
    df.info()
    print("\nОписова статистика:")
    print(df.describe())

    # 2) Перевести datetime у тип datetime та зробити індексом.
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    # 3) Видалити повні дублі рядків.
    n_before_dups = len(df)
    df.drop_duplicates(inplace=True)
    n_dups = n_before_dups - len(df)
    print(f"\n2) drop_duplicates: видалено {n_dups}")

    # 4) Заповнити NaN у humidity_pct МЕДІАНОЮ ПО МІСЯЦЮ В МЕЖАХ МІСТА.
    df['month'] = df.index.month
    n_nan_before = df['humidity_pct'].isna().sum()
    df['humidity_pct'] = df.groupby(['city', 'month'])['humidity_pct'].transform(lambda s: s.fillna(s.median()))
    n_filled = n_nan_before - df['humidity_pct'].isna().sum()
    print(f"3) Заповнено NaN humidity_pct: {n_filled}")

    # 5) Прибрати фізичні викиди
    n_before_outliers = len(df)
    df = df[
        (-60 <= df['temperature_c']) & (df['temperature_c'] <= 60) &
        (df['wind_speed_ms'].isna() | ((df['wind_speed_ms'] >= 0) & (df['wind_speed_ms'] <= 60)))
        ]
    n_outliers = n_before_outliers - len(df)
    print(f"4) Видалено фізичних викидів: {n_outliers}")

    # 6) Звіт очищення.
    print(f"\n   Звіт: {rows_before} → {len(df)} рядків")

    return df


# ====================================================================
# БЛОК 3. Pandas — аналітика (30 балів)
# ====================================================================

def block_3_analytics(df: pd.DataFrame) -> dict:
    section("БЛОК 3. Pandas — аналітика")

    # 1) Середня температура по містах
    by_city_temp = df.groupby('city')['temperature_c'].mean().sort_values()
    print("1) Середня T по містах:")
    print(by_city_temp.round(2).to_string())

    # 2) Сумарні опади по містах. Хто найвологіше?
    by_city_precip = df.groupby('city')['precipitation_mm'].sum().sort_values(ascending=False)
    print("\n2) Сумарні опади по містах:")
    print(by_city_precip.round(1).to_string())

    # 3) Місячна середня температура: resample('ME').mean()
    monthly_mean = df['temperature_c'].resample('ME').mean()
    print(f"\n3) Місячна середня T ({len(monthly_mean)} точок):")
    print(monthly_mean.round(2).to_string())

    # 4) Pivot: місто × місяць, значення = середня T.
    pivot = df.pivot_table(index='city', columns='month', values='temperature_c', aggfunc='mean')
    print("\n4) Pivot місто × місяць:")
    print(pivot.round(1).to_string())

    # 5) Кількість днів з опадами > 5 мм по містах.
    daily_precip = df.groupby(['city', df.index.date])['precipitation_mm'].sum().reset_index()
    rainy_days = daily_precip[daily_precip['precipitation_mm'] > 5].groupby('city').size()
    print("\n5) Дні з опадами > 5 мм:")
    print(rainy_days.to_string())

    # 6) Знайти аномальний місяць.
    monthly_norm = df.groupby('month')['temperature_c'].mean()
    actual_monthly = df.groupby([df.index.year, df.index.month])['temperature_c'].mean()

    # Відхилення від норми
    normals_mapped = actual_monthly.index.get_level_values(1).map(monthly_norm)
    deviations = actual_monthly - normals_mapped

    anomaly_idx = deviations.abs().idxmax()
    anomaly_month = f"{anomaly_idx[0]}-{anomaly_idx[1]:02d}"
    anomaly_dev = deviations.loc[anomaly_idx]

    print(f"\n6) Аномальний місяць: {anomaly_month}  відхилення = {anomaly_dev:+.2f}°C")

    return {
        "by_city_temp": by_city_temp,
        "by_city_precip": by_city_precip,
        "monthly_mean": monthly_mean,
        "pivot": pivot,
    }


# ====================================================================
# БЛОК 4. Matplotlib + інтерпретація (35 балів)
# ====================================================================

def block_4_plots(df: pd.DataFrame, analytics: dict) -> None:
    section("БЛОК 4. Matplotlib")

    # Графік 1: line — місячна динаміка температури по 3 обраних містах.
    fig, ax = plt.subplots(figsize=(11, 5))
    top_cities = df['city'].unique()[:3]
    for city in top_cities:
        city_data = df[df['city'] == city]['temperature_c'].resample('ME').mean()
        ax.plot(city_data.index, city_data.values, marker='o', label=city)
    ax.set_title("Місячна динаміка температури (3 міста)")
    ax.set_xlabel("Дата")
    ax.set_ylabel("Температура (°C)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.savefig(PLOTS_DIR / "01_monthly_temperature_lines.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 2: bar — сумарні опади по містах.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(analytics["by_city_precip"].index, analytics["by_city_precip"].values, color='skyblue', edgecolor='black')
    ax.set_title("Сумарні опади по містах за весь період")
    ax.set_xlabel("Місто")
    ax.set_ylabel("Опади (мм)")
    plt.xticks(rotation=45)
    fig.savefig(PLOTS_DIR / "02_precipitation_by_city.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 3: hist — розподіл температур з вертикальними лініями
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df['temperature_c'].dropna(), bins=40, color='lightgreen', edgecolor='black', alpha=0.7)
    mean_val = df['temperature_c'].mean()
    median_val = df['temperature_c'].median()
    ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}°C')
    ax.axvline(median_val, color='blue', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.1f}°C')
    ax.set_title("Гістограма розподілу температур")
    ax.set_xlabel("Температура (°C)")
    ax.set_ylabel("Кількість спостережень")
    ax.legend()
    fig.savefig(PLOTS_DIR / "03_temperature_histogram.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Графік 4: heatmap pivot місто × місяць
    fig, ax = plt.subplots(figsize=(11, 5))
    cax = ax.imshow(analytics["pivot"], cmap='coolwarm', aspect='auto')
    fig.colorbar(cax, label="Середня температура (°C)")
    ax.set_xticks(np.arange(len(analytics["pivot"].columns)))
    ax.set_yticks(np.arange(len(analytics["pivot"].index)))
    ax.set_xticklabels(analytics["pivot"].columns)
    ax.set_yticklabels(analytics["pivot"].index)
    ax.set_title("Теплова карта температур (Місто × Місяць)")
    ax.set_xlabel("Місяць")
    ax.set_ylabel("Місто")
    fig.savefig(PLOTS_DIR / "04_city_month_heatmap.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"4 графіки збережені в {PLOTS_DIR}/")


# ====================================================================

def main() -> None:
    df_raw = load_observations()
    print(f"Завантажено: shape={df_raw.shape}")

    block_1_numpy(df_raw)
    df_clean = block_2_cleaning(df_raw)
    analytics = block_3_analytics(df_clean)
    block_4_plots(df_clean, analytics)


if __name__ == "__main__":
    main()

"""
ВИСНОВКИ (5–8 речень).

За результатами аналізу, найхолоднішим містом є Львів (7.29°C), а найтеплішим — Київ (11.90°C). 
Температурна динаміка має чітку сезонність (максимуми у червні-липні), але виявлено екстремально холодну 
аномалію у жовтні 2023 року (відхилення -4.32°C). Найвологішим містом є Львів (624.4 мм опадів), 
найсухішим — Одеса (213.4 мм).
"""