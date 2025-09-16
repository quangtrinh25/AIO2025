import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import importlib
import sys
import numpy as np
import evaluation
from sklearn.model_selection import KFold, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample

DATA_PATH = "D:/ICUDATASET"
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')

importlib.reload(evaluation)
import evaluation

class ML_models():
    def __init__(self, data_icu, k_fold, model_type, concat, oversampling):
        self.data_icu = data_icu
        self.k_fold = k_fold
        self.model_type = model_type
        self.concat = concat
        self.oversampling = oversampling
        self.loss = evaluation.Loss('cpu', True, True, True, True, True, True, True, True, True, True, True)
        
        # Khởi tạo encoders một lần duy nhất
        self.encoders = {}
        
        self.ml_train()

    def create_kfolds(self):
        """Tạo k-fold splits cho cross validation"""
        labels = pd.read_csv(f"{DATA_PATH}/data/csv/labels.csv", header=0)
        
        if (self.k_fold == 0):
            k_fold = 5
            self.k_fold = 1
        else:
            k_fold = self.k_fold
        hids = labels.iloc[:, 0]
        y = labels.iloc[:, 1]
        print("Total Samples", len(hids))
        print("Positive Samples", y.sum())
        
        if self.oversampling:
            print("=============OVERSAMPLING===============")
            oversample = RandomOverSampler(sampling_strategy='minority')
            hids = np.asarray(hids).reshape(-1, 1)
            hids, y = oversample.fit_resample(hids, y)
            hids = hids[:, 0]
            print("Total Samples", len(hids))
            print("Positive Samples", y.sum())
        
        ids = range(0, len(hids))
        batch_size = int(len(ids) / k_fold)
        k_hids = []
        for i in range(0, k_fold):
            rids = random.sample(list(ids), batch_size)
            ids = list(set(ids) - set(rids))
            k_hids.append(hids[rids])
        
        return k_hids

    def safe_convert_categorical(self, series):
        """
        Chuyển đổi categorical column một cách an toàn
        """
        # Bước 1: Fill NaN trước
        series = series.fillna('UNKNOWN')
        
        # Bước 2: Chuyển TẤT CẢ thành string
        series = series.astype(str)
        
        # Bước 3: Xử lý các string đặc biệt
        series = series.replace({
            'nan': 'UNKNOWN',
            'None': 'UNKNOWN', 
            'NaN': 'UNKNOWN',
            'null': 'UNKNOWN',
            '': 'UNKNOWN',
            ' ': 'UNKNOWN'
        })
        
        # Bước 4: Strip whitespace và uppercase để chuẩn hóa
        series = series.str.strip().str.upper()
        
        return series

    def robust_categorical_preprocessing(self, df, fit_encoders=False):
        """
        Xử lý categorical data một cách robust và consistent
        
        Args:
            df: DataFrame cần xử lý
            fit_encoders: True nếu cần fit encoders, False nếu chỉ transform
        
        Returns:
            df: DataFrame đã được xử lý
        """
        categorical_columns = ['gender', 'ethnicity', 'insurance']
        
        for col in categorical_columns:
            if col in df.columns:
                # Sử dụng safe conversion
                df[col] = self.safe_convert_categorical(df[col])
                
                # Fit hoặc transform encoders
                if fit_encoders:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(df[col])
                    print(f"{col} unique values: {sorted(self.encoders[col].classes_)}")
                else:
                    # Transform, xử lý unseen categories
                    try:
                        df[col] = self.encoders[col].transform(df[col])
                    except ValueError as e:
                        print(f"Warning: Unseen categories in {col}. Handling...")
                        # Tìm unseen categories và thay thế bằng 'UNKNOWN'
                        unseen_mask = ~df[col].isin(self.encoders[col].classes_)
                        if unseen_mask.any():
                            df.loc[unseen_mask, col] = 'UNKNOWN'
                        df[col] = self.encoders[col].transform(df[col])
        
        return df

    def handle_all_categorical_columns(self, df, fit_encoders=False):
        """
        Xử lý tất cả các cột có thể là categorical (bao gồm cả object columns)
        """
        # Xử lý các cột categorical đã biết
        df = self.robust_categorical_preprocessing(df, fit_encoders)
        
        # Tìm và xử lý tất cả object columns khác
        object_columns = df.select_dtypes(include=['object']).columns.tolist()
        predefined_cats = ['gender', 'ethnicity', 'insurance']
        other_object_cols = [col for col in object_columns if col not in predefined_cats]
        
        for col in other_object_cols:
            # Convert object columns thành numeric nếu có thể, nếu không thì encode
            try:
                # Thử convert thành numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Fill NaN bằng median, nếu tất cả NaN thì fill bằng 0
                if df[col].isna().all():
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna(df[col].median())
            except:
                # Nếu không convert được, encode như categorical
                df[col] = self.safe_convert_categorical(df[col])
                
                if fit_encoders:
                    if col not in self.encoders:
                        self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(df[col])
                else:
                    if col in self.encoders:
                        # Handle unseen categories
                        try:
                            df[col] = self.encoders[col].transform(df[col])
                        except ValueError:
                            unseen_mask = ~df[col].isin(self.encoders[col].classes_)
                            if unseen_mask.any():
                                df.loc[unseen_mask, col] = 'UNKNOWN'
                            df[col] = self.encoders[col].transform(df[col])
                    else:
                        # Nếu không có encoder, chuyển thành numeric hash
                        df[col] = pd.factorize(df[col])[0]
        
        return df

    def ml_train(self):
        k_hids = self.create_kfolds()
        labels = pd.read_csv(f"{DATA_PATH}/data/csv/labels.csv", header=0)
        
        # Get all data để fit encoders
        all_hids = []
        for fold_hids in k_hids:
            all_hids.extend(fold_hids)
        
        concat_cols = []
        if self.concat:
            dyn = pd.read_csv(f"{DATA_PATH}/data/csv/{str(all_hids[0])}/dynamic.csv", header=[0, 1])
            dyn.columns = dyn.columns.droplevel(0)
            cols = dyn.columns
            time = dyn.shape[0]
            for t in range(time):
                cols_t = [x + "_" + str(t) for x in cols]
                concat_cols.extend(cols_t)
        
        print("Fitting encoders on complete dataset...")
        X_all, _ = self.getXY(all_hids, labels, concat_cols)
        
        # FIT ENCODERS trên toàn bộ data
        X_all = self.handle_all_categorical_columns(X_all, fit_encoders=True)
        
        for i in range(self.k_fold):
            print("==================={0:2d} FOLD=====================".format(i))
            test_hids = k_hids[i]
            train_ids = list(set([0, 1, 2, 3, 4]) - set([i]))
            train_hids = []
            for j in train_ids:
                train_hids.extend(k_hids[j])
            
            # Training data
            print('train_hids', len(train_hids))
            X_train, Y_train = self.getXY(train_hids, labels, concat_cols)
            X_train = self.handle_all_categorical_columns(X_train, fit_encoders=False)
            
            # Test data
            print('test_hids', len(test_hids))
            X_test, Y_test = self.getXY(test_hids, labels, concat_cols)
            self.test_data = X_test.copy(deep=True)
            X_test = self.handle_all_categorical_columns(X_test, fit_encoders=False)
            
            print(f"Training shape: {X_train.shape}, Test shape: {X_test.shape}")
            print(f"Training dtypes: {X_train.dtypes.value_counts().to_dict()}")
            
            self.train_model(X_train, Y_train, X_test, Y_test)

    def train_model(self, X_train, Y_train, X_test, Y_test):
        """Simplified train_model - không cần xử lý categorical nữa"""
        print("===============MODEL TRAINING===============")
        
        # Ensure all data is numeric
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test[X_train.columns]  # Ensure same columns
        
        Y_train = np.asarray(Y_train).ravel()
        Y_test = np.asarray(Y_test).ravel()
        
        if self.model_type == 'Gradient Boosting':
            model = HistGradientBoostingClassifier().fit(X_train, Y_train)
            prob = model.predict_proba(X_test)
            logits = np.log2(prob[:, 1] / (prob[:, 0] + 1e-10))
            self.loss(prob[:, 1], Y_test, logits, False, True)
            self.save_output(Y_test, prob[:, 1], logits)
            self.save_model(model, f"{DATA_PATH}/data/output/{self.model_type}_model.pkl")
        
        elif self.model_type == 'Logistic Regression':
            model = LogisticRegression(max_iter=1000).fit(X_train, Y_train)
            prob = model.predict_proba(X_test)
            logits = model.predict_log_proba(X_test)
            self.loss(prob[:, 1], Y_test, logits[:, 1], False, True)
            self.save_outputImp(Y_test, prob[:, 1], logits[:, 1], model.coef_[0], X_train.columns)
            self.save_model(model, f"{DATA_PATH}/data/output/{self.model_type}_model.pkl")
        
        elif self.model_type == 'Random Forest':
            model = RandomForestClassifier(random_state=42).fit(X_train, Y_train)
            prob = model.predict_proba(X_test)
            logits = model.predict_log_proba(X_test)
            self.loss(prob[:, 1], Y_test, logits[:, 1], False, True)
            self.save_outputImp(Y_test, prob[:, 1], logits[:, 1], model.feature_importances_, X_train.columns)
            self.save_model(model, f"{DATA_PATH}/data/output/{self.model_type}_model.pkl")
        
        elif self.model_type == 'Xgboost':
            # Calculate scale_pos_weight
            scale_pos_weight_val = (Y_train == 0).sum() / (Y_train == 1).sum()
            
            base_model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric='logloss',
                use_label_encoder=False,
                tree_method='hist',
                enable_categorical=False,
                scale_pos_weight=scale_pos_weight_val,
                random_state=42
            )
            
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            grid_search_f1 = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring='f1',
                cv=5,
                verbose=1,
                n_jobs=-1
            )
            grid_search_f1.fit(X_train, Y_train)
            model = grid_search_f1.best_estimator_
            
            print(f"Best params for F1: {grid_search_f1.best_params_}")
            print(f"Best F1 score: {grid_search_f1.best_score_:.4f}")
            
            # CV metrics
            scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            cv_results = cross_validate(model, X_train, Y_train, cv=5, scoring=scoring_metrics, n_jobs=-1)
            print("Average CV Metrics:")
            for metric in scoring_metrics:
                mean_score = cv_results[f'test_{metric}'].mean()
                std_score = cv_results[f'test_{metric}'].std()
                print(f"  {metric.capitalize()}: {mean_score:.4f} (+/- {std_score:.4f})")
            
            prob = model.predict_proba(X_test)
            logits = np.log2(prob[:, 1] / (prob[:, 0] + 1e-10))
            self.loss(prob[:, 1], Y_test, logits, False, True)
            self.save_outputImp(Y_test, prob[:, 1], logits, model.feature_importances_, X_train.columns)
            self.save_model(model, f"{DATA_PATH}/data/output/{self.model_type}_model.pkl")

    def evaluate_model_bootstrapped(self, model, X_train_full, y_train_full, X_test_data, y_test_data, n_bootstraps=1000):
        """Bootstrapped evaluation từ Model 2, in kết quả."""
        model.fit(X_train_full, y_train_full)  # Train on full train

        scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        test_metrics = {metric: [] for metric in scoring_metrics}

        for i in range(n_bootstraps):
            bootstrap_indices = np.random.choice(len(X_test_data), len(X_test_data), replace=True)
            X_test_bootstrap = X_test_data.iloc[bootstrap_indices]
            y_test_bootstrap = y_test_data.iloc[bootstrap_indices]

            if X_test_bootstrap.empty:
                continue

            y_pred = model.predict(X_test_bootstrap)
            y_proba = model.predict_proba(X_test_bootstrap)[:, 1]

            test_metrics['accuracy'].append(accuracy_score(y_test_bootstrap, y_pred))
            test_metrics['precision'].append(precision_score(y_test_bootstrap, y_pred, zero_division=0))
            test_metrics['recall'].append(recall_score(y_test_bootstrap, y_pred, zero_division=0))
            test_metrics['f1'].append(f1_score(y_test_bootstrap, y_pred, zero_division=0))
            if len(np.unique(y_test_bootstrap)) > 1:
                test_metrics['roc_auc'].append(roc_auc_score(y_test_bootstrap, y_proba))
            else:
                test_metrics['roc_auc'].append(np.nan)

        # In kết quả bootstrapped
        print("Bootstrapped Test Metrics (1000 iterations):")
        for metric, scores in test_metrics.items():
            scores = [s for s in scores if not np.isnan(s)]
            if len(scores) > 0:
                mean_score = np.mean(scores)
                lower_bound = np.percentile(scores, 2.5)
                upper_bound = np.percentile(scores, 97.5)
                print(f"  {metric.capitalize()}: {mean_score:.4f} (95% CI: {lower_bound:.4f}-{upper_bound:.4f})")
            else:
                print(f"  {metric.capitalize()}: N/A")

    def getXY(self, ids, labels, concat_cols):
        """Lấy X và y data từ các file CSV"""
        X_df = pd.DataFrame()   
        y_df = pd.DataFrame()   
        features = []
        
        for sample in ids:
            if self.data_icu:
                y = labels[labels['stay_id'] == sample]['label']
            else:
                y = labels[labels['hadm_id'] == sample]['label']
            
            dyn = pd.read_csv(f"{DATA_PATH}/data/csv/{str(sample)}/dynamic.csv", header=[0, 1])
            
            if self.concat:
                dyn.columns = dyn.columns.droplevel(0)
                dyn = dyn.to_numpy()
                dyn = dyn.reshape(1, -1)
                dyn_df = pd.DataFrame(data=dyn, columns=concat_cols)
                features = concat_cols
            else:
                dyn_df = pd.DataFrame()
                for key in dyn.columns.levels[0]:
                    dyn_temp = dyn[key]
                    if self.data_icu:
                        if ((key == "CHART") or (key == "MEDS")):
                            agg = dyn_temp.aggregate("mean")
                            agg = agg.reset_index()
                        else:
                            agg = dyn_temp.aggregate("max")
                            agg = agg.reset_index()
                    else:
                        if ((key == "LAB") or (key == "MEDS")):
                            agg = dyn_temp.aggregate("mean")
                            agg = agg.reset_index()
                        else:
                            agg = dyn_temp.aggregate("max")
                            agg = agg.reset_index()
                    if dyn_df.empty:
                        dyn_df = agg
                    else:
                        dyn_df = pd.concat([dyn_df, agg], axis=0)
                dyn_df = dyn_df.T
                dyn_df.columns = dyn_df.iloc[0]
                dyn_df = dyn_df.iloc[1:, :]
                        
            stat = pd.read_csv(f"{DATA_PATH}/data/csv/{str(sample)}/static.csv", header=[0, 1])
            stat = stat['COND']
            demo = pd.read_csv(f"{DATA_PATH}/data/csv/{str(sample)}/demo.csv", header=0)
            
            if X_df.empty:
                X_df = pd.concat([dyn_df, stat], axis=1)
                X_df = pd.concat([X_df, demo], axis=1)
            else:
                X_df = pd.concat([X_df, pd.concat([pd.concat([dyn_df, stat], axis=1), demo], axis=1)], axis=0)
            if y_df.empty:
                y_df = y
            else:
                y_df = pd.concat([y_df, y], axis=0)
                
        print("X_df", X_df.shape)
        print("y_df", y_df.shape)
        return X_df, y_df
    
    def save_output(self, labels, prob, logits):
        """Save output cho gradient boosting"""
        # Tạo thư mục output nếu chưa tồn tại
        output_dir = f"{DATA_PATH}/data/output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_df = pd.DataFrame()
        output_df['Labels'] = labels
        output_df['Prob'] = prob
        output_df['Logits'] = np.asarray(logits)
        output_df['ethnicity'] = list(self.test_data['ethnicity'])
        output_df['gender'] = list(self.test_data['gender'])
        output_df['age'] = list(self.test_data['Age'])
        output_df['insurance'] = list(self.test_data['insurance'])
        
        with open(f"{output_dir}/outputDict", 'wb') as fp:
            pickle.dump(output_df, fp)
    
    def save_outputImp(self, labels, prob, logits, importance, features):
        """Save output với feature importance"""
        # Tạo thư mục output nếu chưa tồn tại
        output_dir = f"{DATA_PATH}/data/output"
        os.makedirs(output_dir, exist_ok=True)
        
        output_df = pd.DataFrame()
        output_df['Labels'] = labels
        output_df['Prob'] = prob
        output_df['Logits'] = np.asarray(logits)
        output_df['ethnicity'] = list(self.test_data['ethnicity'])
        output_df['gender'] = list(self.test_data['gender'])
        output_df['age'] = list(self.test_data['Age'])
        output_df['insurance'] = list(self.test_data['insurance'])
        
        with open(f"{output_dir}/outputDict", 'wb') as fp:
            pickle.dump(output_df, fp)
        
        imp_df = pd.DataFrame()
        imp_df['imp'] = importance
        imp_df['feature'] = features
        imp_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    def save_model(self, model, filepath):
        """Lưu mô hình bằng pickle"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Tạo thư mục nếu chưa có
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {filepath}")