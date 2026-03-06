#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
茶色素智能检测系统 - 数据导入和模型训练脚本
Tea Pigment Detection System - Data Import and Model Training Script
"""

import os
import sys
import pandas as pd
import numpy as np
import cv2
import joblib
from pathlib import Path
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 添加项目目录到路径
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# 导入核心模块
from core.image_processor import ImageProcessor


class DataImporter:
    """数据导入器 - 从CSV和图片加载数据"""
    
    def __init__(self):
        """初始化数据导入器"""
        self.processor = ImageProcessor()
        self.data = []
        self.features = []
        self.labels = {
            'tf': [],
            'tr': [],
            'tb': []
        }
    
    def load_from_csv(self, csv_path: str, img_dir: str = None) -> bool:
        """
        从CSV文件加载图片路径和标签数据
        
        Args:
            csv_path: CSV文件路径
            img_dir: 图片目录（如果CSV中是相对路径）
            
        Returns:
            加载成功返回True
        """
        try:
            print(f"正在读取CSV文件: {csv_path}")
            
            # 尝试用不同编码读取CSV文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    print(f"✓ 使用 {encoding} 编码成功读取CSV文件")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                print(f"✗ 无法用常见编码读取CSV文件，请检查文件编码")
                return False
            print(f"✓ CSV文件读取成功，共 {len(df)} 条记录")
            print(f"  列名: {list(df.columns)}")
            print(f"  前几行数据:\n{df.head()}")
            
            # 检查必需的列
            required_cols = self._check_columns(df)
            if not required_cols:
                print("✗ CSV文件缺少必需的列")
                return False
            
            # 处理每一行数据
            success_count = 0
            failed_count = 0
            
            for idx, row in df.iterrows():
                try:
                    # 获取图片路径
                    img_path = self._get_image_path(row, img_dir, csv_path)
                    
                    if not os.path.exists(img_path):
                        print(f"⚠ 图片不存在，跳过: {img_path}")
                        failed_count += 1
                        continue
                    
                    # 加载图片
                    img = self.processor.load_image(img_path)
                    if img is None:
                        print(f"⚠ 图片加载失败，跳过: {img_path}")
                        failed_count += 1
                        continue
                    
                    # 提取特征
                    features = self.processor.extract_color_features(img)
                    
                    # 提取标签
                    tf_value = self._get_value(row, 'tf')
                    tr_value = self._get_value(row, 'tr')
                    tb_value = self._get_value(row, 'tb')
                    
                    # 存储数据
                    self.data.append({
                        'image_path': img_path,
                        'features': features,
                        'tf': tf_value,
                        'tr': tr_value,
                        'tb': tb_value
                    })
                    
                    success_count += 1
                    
                    # 显示进度
                    if (success_count + failed_count) % 10 == 0:
                        print(f"  进度: {success_count + failed_count}/{len(df)} (成功: {success_count}, 失败: {failed_count})")
                
                except Exception as e:
                    print(f"⚠ 处理第 {idx+1} 行时出错: {e}")
                    failed_count += 1
            
            print(f"\n✓ 数据加载完成")
            print(f"  成功: {success_count} 条")
            print(f"  失败: {failed_count} 条")
            
            return success_count > 0
            
        except Exception as e:
            print(f"✗ 读取CSV文件失败: {e}")
            return False
    
    def _check_columns(self, df: pd.DataFrame) -> bool:
        """
        检查CSV是否包含必需的列
        
        Args:
            df: DataFrame
            
        Returns:
            是否包含必需的列
        """
        # 可能的列名映射
        column_mapping = {
            'image': ['image_path', 'image', 'Image', 'Image_Path', 'path', 'Path', 'file', 'File', '文件', '图片'],
            'tf': ['tf', 'TF', 'tf_value', '茶黄素', '茶黄素(%)', 'Theaflavins', 'tea_pigment_tf'],
            'tr': ['tr', 'TR', 'tr_value', '茶红素', '茶红素(%)', 'Thearubigins', 'tea_pigment_tr'],
            'tb': ['tb', 'TB', 'tb_value', '茶褐素', '茶褐素(%)', 'Theabrownins', 'tea_pigment_tb']
        }
        
        self.col_mapping = {}
        
        for key, possible_names in column_mapping.items():
            found = False
            for name in possible_names:
                if name in df.columns:
                    self.col_mapping[key] = name
                    found = True
                    print(f"✓ 找到 {key} 列: {name}")
                    break
            
            if not found and key != 'image':
                print(f"✗ 未找到 {key} 列")
                return False
        
        return True
    
    def _get_image_path(self, row: pd.Series, img_dir: str, csv_path: str) -> str:
        """
        获取图片的完整路径
        
        Args:
            row: 数据行
            img_dir: 图片目录
            csv_path: CSV文件路径
            
        Returns:
            图片完整路径
        """
        # 获取图片路径（使用列映射）
        img_col = self.col_mapping.get('image')
        img_path = str(row[img_col])
        
        # 如果是相对路径，需要拼接
        if not os.path.isabs(img_path):
            if img_dir:
                img_path = os.path.join(img_dir, img_path)
            else:
                # 使用CSV文件所在目录
                csv_dir = os.path.dirname(csv_path)
                img_path = os.path.join(csv_dir, img_path)
        
        return img_path
    
    def _get_value(self, row: pd.Series, pigment_type: str) -> float:
        """
        获取色素含量值
        
        Args:
            row: 数据行
            pigment_type: 色素类型 (tf/tr/tb)
            
        Returns:
            色素含量值
        """
        col_name = self.col_mapping.get(pigment_type)
        if col_name is None:
            return 0.0
        
        value = float(row[col_name])
        return value
    
    def prepare_dataset(self):
        """准备训练数据集"""
        if not self.data:
            print("✗ 没有可用数据")
            return None
        
        print("\n正在准备训练数据集...")
        
        # 提取所有特征
        feature_dict_list = [item['features'] for item in self.data]
        
        # 展平特征为向量
        self.features = self._flatten_features(feature_dict_list)
        
        # 提取标签
        self.labels['tf'] = np.array([item['tf'] for item in self.data])
        self.labels['tr'] = np.array([item['tr'] for item in self.data])
        self.labels['tb'] = np.array([item['tb'] for item in self.data])
        
        print(f"✓ 数据集准备完成")
        print(f"  特征维度: {self.features.shape}")
        print(f"  样本数量: {len(self.data)}")
        print(f"  茶黄素(TF) 范围: {self.labels['tf'].min():.3f} - {self.labels['tf'].max():.3f}")
        print(f"  茶红素(TR) 范围: {self.labels['tr'].min():.3f} - {self.labels['tr'].max():.3f}")
        print(f"  茶褐素(TB) 范围: {self.labels['tb'].min():.3f} - {self.labels['tb'].max():.3f}")
        
        return self.features, self.labels
    
    def _flatten_features(self, feature_dict_list: list) -> np.ndarray:
        """
        将特征字典列表展平为矩阵
        
        Args:
            feature_dict_list: 特征字典列表
            
        Returns:
            特征矩阵
        """
        # 获取所有特征名称
        all_feature_names = set()
        for features in feature_dict_list:
            for space in features.keys():
                for name in features[space].keys():
                    all_feature_names.add(f"{space}_{name}")
        
        self.feature_names = sorted(list(all_feature_names))
        print(f"  特征数量: {len(self.feature_names)}")
        
        # 构建特征矩阵
        feature_matrix = []
        for features in feature_dict_list:
            feature_vector = []
            for feature_name in self.feature_names:
                space, name = feature_name.split('_', 1)
                value = features.get(space, {}).get(name, 0.0)
                feature_vector.append(value)
            feature_matrix.append(feature_vector)
        
        return np.array(feature_matrix, dtype=np.float64)


class ModelTrainer:
    """模型训练器 - 训练茶色素预测模型"""
    
    def __init__(self, features: np.ndarray, labels: dict, feature_names: list):
        """
        初始化模型训练器
        
        Args:
            features: 特征矩阵
            labels: 标签字典
            feature_names: 特征名称列表
        """
        self.features = features
        self.labels = labels
        self.feature_names = feature_names
        self.models = {}
        self.scalers = {}
        self.pcas = {}
        
    def train(self, test_size: float = 0.2, random_state: int = 42):
        """
        训练模型
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        """
        print("\n开始训练模型...")
        
        # 对每种色素类型训练模型
        for pigment_type in ['tf', 'tr', 'tb']:
            print(f"\n正在训练 {pigment_type.upper()} 模型...")
            
            # 准备数据
            X = self.features
            y = self.labels[pigment_type]
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # 数据标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[pigment_type] = scaler
            
            # PCA降维（保留95%方差）
            pca = PCA(n_components=0.95)
            X_train_pca = pca.fit_transform(X_train_scaled)
            X_test_pca = pca.transform(X_test_scaled)
            self.pcas[pigment_type] = pca
            
            print(f"  特征维度: {X_train.shape[1]} -> {X_train_pca.shape[1]} (PCA降维后)")
            
            # 训练SVR模型
            model = SVR(kernel='rbf', C=10, gamma='scale')
            model.fit(X_train_pca, y_train)
            self.models[pigment_type] = model
            
            # 评估模型
            y_pred = model.predict(X_test_pca)
            
            # 计算指标
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            
            # 计算RPD
            std_dev = np.std(y_test)
            rpd = std_dev / rmse if rmse > 0 else 0
            
            print(f"  训练集大小: {len(X_train)}")
            print(f"  测试集大小: {len(X_test)}")
            print(f"  均方误差(MSE): {mse:.6f}")
            print(f"  均方根误差(RMSE): {rmse:.6f}")
            print(f"  决定系数(R²): {r2:.6f}")
            print(f"  相对预测偏差(RPD): {rpd:.3f}")
            
            if rpd > 3.0:
                print(f"  ✓ 模型质量优秀 (RPD > 3.0)")
            elif rpd > 2.0:
                print(f"  △ 模型质量良好 (RPD > 2.0)")
            else:
                print(f"  ✗ 模型质量需要改进 (RPD < 2.0)")
    
    def save_models(self, output_dir: str = "models"):
        """
        保存训练好的模型
        
        Args:
            output_dir: 输出目录
        """
        print(f"\n正在保存模型到 {output_dir}...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每种色素的模型
        for pigment_type in ['tf', 'tr', 'tb']:
            # 保存模型
            model_path = os.path.join(output_dir, f"{pigment_type}_model.pkl")
            joblib.dump(self.models[pigment_type], model_path)
            print(f"✓ {pigment_type}_model.pkl")
            
            # 保存PCA
            pca_path = os.path.join(output_dir, f"{pigment_type}_pca.pkl")
            joblib.dump(self.pcas[pigment_type], pca_path)
            print(f"✓ {pigment_type}_pca.pkl")
        
        # 保存统一的标准化器（保存第一个色素的标准化器）
        if 'tf' in self.scalers:
            scaler_path = os.path.join(output_dir, "scaler.pkl")
            joblib.dump(self.scalers['tf'], scaler_path)
            print(f"✓ scaler.pkl")
        
        # 保存特征名称
        feature_names_path = os.path.join(output_dir, "feature_names.pkl")
        joblib.dump(self.feature_names, feature_names_path)
        print(f"✓ feature_names.pkl")
        
        print(f"\n✓ 所有模型保存完成！")


def main():
    """主函数"""
    print("=" * 60)
    print("茶色素智能检测系统 - 数据导入和模型训练")
    print("=" * 60)
    
    # 第1步：用户输入
    print("\n请输入数据信息：")
    
    csv_path = input("CSV文件路径（例如: data/tea_data.csv）: ").strip()
    img_dir = input("图片目录（如果CSV中是相对路径，请输入图片目录，直接回车跳过）: ").strip()
    
    if not csv_path:
        print("✗ 必须提供CSV文件路径")
        return
    
    if not img_dir:
        img_dir = None
    
    # 第2步：导入数据
    print("\n" + "-" * 60)
    print("步骤1: 导入数据")
    print("-" * 60)
    
    importer = DataImporter()
    success = importer.load_from_csv(csv_path, img_dir)
    
    if not success:
        print("\n✗ 数据导入失败，请检查CSV文件格式和图片路径")
        return
    
    # 第3步：准备数据集
    print("\n" + "-" * 60)
    print("步骤2: 准备数据集")
    print("-" * 60)
    
    features, labels = importer.prepare_dataset()
    
    if features is None:
        print("\n✗ 数据集准备失败")
        return
    
    # 第4步：训练模型
    print("\n" + "-" * 60)
    print("步骤3: 训练模型")
    print("-" * 60)
    
    trainer = ModelTrainer(features, labels, importer.feature_names)
    trainer.train()
    
    # 第5步：保存模型
    print("\n" + "-" * 60)
    print("步骤4: 保存模型")
    print("-" * 60)
    
    output_dir = input("模型保存目录（默认: models）: ").strip()
    if not output_dir:
        output_dir = "models"
    
    trainer.save_models(output_dir)
    
    print("\n" + "=" * 60)
    print("✓ 训练完成！")
    print("=" * 60)
    print(f"\n模型已保存到: {output_dir}")
    print("现在可以运行主程序使用这些模型:")
    print("  python tea_pigment_detector/main_enhanced.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练已取消")
    except Exception as e:
        print(f"\n\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()