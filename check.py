import pandas as pd
DATA_PATH = "D:/ICUDATASET"
file_path = f"{DATA_PATH}/data/features/preproc_chart_icu.csv.gz"
chart_data = pd.read_csv(file_path, compression='gzip', nrows=100)  # Đọc 100 hàng đầu để kiểm tra
print("Columns:", chart_data.columns.tolist())
print("Shape (first 100 rows):", chart_data.shape)
print("Head:\n", chart_data.head())
print("Missing values in key columns:")
for col in ['stay_id', 'itemid', 'valuenum', 'event_time_from_admit']:
    if col in chart_data.columns:
        print(f"{col}: {chart_data[col].isnull().sum()} missing")
    else:
        print(f"{col}: Missing column!")