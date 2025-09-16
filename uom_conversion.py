import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import glob

def drop_wrong_uom(data, cut_off, chunksize=50000, temp_dir="D:/ICUDATASET/temp_uom"):
    """
    Drop wrong UOM using chunking and save incrementally to avoid MemoryError.
    - data: Input file path to CSV or DataFrame.
    - cut_off: Threshold for dropping UOM (e.g., 0.95).
    - chunksize: Number of rows per chunk (default 50k for 16GB RAM).
    - temp_dir: Directory for temporary files.
    - Returns: Path to the processed file instead of DataFrame.
    """
    if isinstance(data, str):  # If data is a file path
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        chunk_count = 0
        # Tạo file đầu ra ngay từ đầu để ghi từng phần
        output_path = data.replace('.csv.gz', '_processed.csv.gz')
        first_chunk = True
        for chunk in pd.read_csv(data, compression='gzip', chunksize=chunksize):
            processed_chunk = _drop_wrong_uom_chunk(chunk, cut_off)
            temp_file = os.path.join(temp_dir, f"chunk_{chunk_count}.csv.gz")
            try:
                processed_chunk.to_csv(temp_file, compression='gzip', index=False)
            except Exception as e:
                print(f"Error writing chunk {chunk_count} to {temp_file}: {e}")
                raise
            chunk_count += 1
            # Ghi trực tiếp vào file cuối thay vì load toàn bộ
            if first_chunk:
                processed_chunk.to_csv(output_path, compression='gzip', index=False, mode='w')
                first_chunk = False
            else:
                processed_chunk.to_csv(output_path, compression='gzip', index=False, mode='a', header=False)
        
        # Xóa file tạm (giữ thư mục)
        temp_files = glob.glob(os.path.join(temp_dir, "chunk_*.csv.gz"))
        for f in temp_files:
            os.remove(f)
        
        # Kiểm tra file tồn tại trước khi trả về
        if not os.path.exists(output_path):
            raise FileNotFoundError(f"Processed file {output_path} was not created")
        return output_path
    else:  # If data is a DataFrame, chunk it manually
        if len(data) > chunksize:
            print(f"DataFrame large ({len(data)} rows), chunking with size {chunksize}")
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
            chunk_count = 0
            # Tạo file đầu ra ngay từ đầu để ghi từng phần
            output_path = f"{temp_dir}/final_processed.csv.gz"
            first_chunk = True
            for i in tqdm(range(0, len(data), chunksize), desc="Processing DataFrame chunks"):
                chunk = data.iloc[i:i + chunksize].copy(deep=True)
                processed_chunk = _drop_wrong_uom_chunk(chunk, cut_off)
                temp_file = os.path.join(temp_dir, f"chunk_{chunk_count}.csv.gz")
                try:
                    processed_chunk.to_csv(temp_file, compression='gzip', index=False)
                except Exception as e:
                    print(f"Error writing chunk {chunk_count} to {temp_file}: {e}")
                    raise
                chunk_count += 1
                # Ghi trực tiếp vào file cuối thay vì load toàn bộ
                if first_chunk:
                    processed_chunk.to_csv(output_path, compression='gzip', index=False, mode='w')
                    first_chunk = False
                else:
                    processed_chunk.to_csv(output_path, compression='gzip', index=False, mode='a', header=False)
            
            # Xóa file tạm (giữ thư mục)
            temp_files = glob.glob(os.path.join(temp_dir, "chunk_*.csv.gz"))
            for f in temp_files:
                os.remove(f)
            
            # Kiểm tra file tồn tại trước khi trả về
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Processed file {output_path} was not created")
            return output_path
        else:
            processed_df = _drop_wrong_uom_chunk(data, cut_off)
            output_path = f"{temp_dir}/final_processed.csv.gz"
            try:
                processed_df.to_csv(output_path, compression='gzip', index=False)
            except Exception as e:
                print(f"Error writing final file to {output_path}: {e}")
                raise
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Processed file {output_path} was not created")
            return output_path

def _drop_wrong_uom_chunk(chunk, cut_off):
    """Internal function to process a single chunk."""
    def process_group(group):
        value_counts = group['valueuom'].value_counts()
        num_observations = len(group)
        if value_counts.size > 1:
            most_frequent_measurement = value_counts.index[0]
            frequency = value_counts[0]
            if frequency / num_observations > cut_off:
                return group[group['valueuom'] == most_frequent_measurement]
        return group
    
    chunk = chunk.groupby('itemid').apply(process_group).reset_index(drop=True)
    return chunk