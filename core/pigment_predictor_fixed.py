# -*- coding: utf- -*-
"""
茶色素智能检测系统 - 预测模型模块（增强版）
Tea Pigment Intelligent Detection System - Prediction Model Module (Enhanced)
增加了默认预测模式，无需训练模型即可测试系统功能
"""

import os
import joblib
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin


class PigmentPredictor:
    """
    茶色素预测器 - 基于机器学习模型预测茶色素含量
    支持默认预测模式和机器学习模型预测模式
    """
    
    def __init__(self):
        """初始化茶色素预测器"""
        self.models = {
            'tf': None,  # 茶黄素模型
            'tr': None,  # 茶红素模型
            'tb': None   # 茶褐素模型
        }
        
        self.scaler = None
        self.feature_names = None
        self.model_path = None
        self.models_loaded = False
        
        # 预测结果范围（用于验证）
        self.tf_range = (0.0, 2.0)    # 茶黄素含量范围 (%)
        self.tr_range = (0.0, 15.0)   # 茶红素含量范围 (%)
        self.tb_range = (0.0, 20.0)   # 茶褐素含量范围 (%)
        
        # 默认预测模式参数（基于颜色特征的简单映射）
        self.default_params = {
            'tf': {
                'r_weight': 0.5,
                'g_weight': 0.3,
                'b_weight': 0.2,
                'base_value': 0.1,
                'scale_factor': 1.5
            },
            'tr': {
                'r_weight': 0.2,
                'g_weight': 0.5,
                'b_weight': 0.3,
                'base_value': 2.0,
                'scale_factor': 8.0
            },
            'tb': {
                'r_weight': 0.3,
                'g_weight': 0.2,
                'b_weight': 0.5,
                'base_value': 3.0,
                'scale_factor': 6.0
            }
        }
    
    def load_models(self, model_dir: str) -> bool:
        """
        加载训练好的模型
        
        Args:
            model_dir: 模型文件目录
            
        Returns:
            加载成功返回True，失败返回False
        """
        try:
            # 检查模型目录是否存在
            if not os.path.exists(model_dir):
                print(f"模型目录不存在: {model_dir}")
                print("将使用默认预测模式")
                self.models_loaded = False
                return False
            
            # 加载茶黄素预测模型
            tf_model_path = os.path.join(model_dir, 'tf_model.pkl')
            if os.path.exists(tf_model_path):
                self.models['tf'] = joblib.load(tf_model_path)
            else:
                print(f"茶黄素模型文件不存在: {tf_model_path}")
                print("将使用默认预测模式")
                self.models_loaded = False
                return False
            
            # 加载茶红素预测模型
            tr_model_path = os.path.join(model_dir, 'tr_model.pkl')
            if os.path.exists(tr_model_path):
                self.models['tr'] = joblib.load(tr_model_path)
            else:
                print(f"茶红素模型文件不存在: {tr_model_path}")
                print("将使用默认预测模式")
                self.models_loaded = False
                return False
            
            # 加载茶褐素预测模型
            tb_model_path = os.path.join(model_dir, 'tb_model.pkl')
            if os.path.exists(tb_model_path):
                self.models['tb'] = joblib.load(tb_model_path)
            else:
                print(f"茶褐素模型文件不存在: {tb_model_path}")
                print("将使用默认预测模式")
                self.models_loaded = False
                return False
            
            # 加载特征归一化器
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                print(f"特征归一化器文件不存在: {scaler_path}")
                self.models_loaded = False
                return False
            
            # 加载特征名称
            feature_names_path = os.path.join(model_dir, 'feature_names.pkl')
            if os.path.exists(feature_names_path):
                self.feature_names = joblib.load(feature_names_path)
            else:
                print(f"特征名称文件不存在: {feature_names_path}")
                self.models_loaded = False
                return False
            
            self.model_path = model_dir
            self.models_loaded = True
            print("所有模型文件加载成功！")
            return True
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("将使用默认预测模式")
            self.models_loaded = False
            return False
    
    def predict(self, features: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """
        预测茶色素含量
        如果模型未加载，将使用默认预测模式
        
        Args:
            features: 图像颜色特征
            
        Returns:
            包含预测结果的字典，格式：{'tf': (预测值, 置信度), ...}
        """
        # 检查是否使用机器学习模型
        if self.models_loaded and all(model is not None for model in self.models.values()):
            return self._predict_with_model(features)
        else:
            return self._predict_with_default(features)
    
    def _predict_with_default(self, features: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """
        使用默认预测模式进行预测
        基于RGB颜色特征的简单线性映射
        
        Args:
            features: 图像颜色特征
            
        Returns:
            包含预测结果的字典
        """
        print("使用默认预测模式进行预测...")
        
        try:
            # 提取RGB均值
            r_mean = features.get('RGB', {}).get('R_mean', 128.0)
            g_mean = features.get('RGB', {}).get('G_mean', 128.0)
            b_mean = features.get('RGB', {}).get('B_mean', 128.0)
            
            # 归一化到 [0, 1]
            r_norm = r_mean / 255.0
            g_norm = g_mean / 255.0
            b_norm = b_mean / 255.0
            
            # 计算亮度（用于调整置信度）
            brightness = (r_mean + g_mean + b_mean) / 3.0 / 255.0
            
            # 预测茶黄素（TF）
            tf_params = self.default_params['tf']
            tf_pred = tf_params['base_value'] + \
                     (r_norm * tf_params['r_weight'] + 
                      g_norm * tf_params['g_weight'] + 
                      b_norm * tf_params['b_weight']) * tf_params['scale_factor']
            
            # 预测茶红素（TR）
            tr_params = self.default_params['tr']
            tr_pred = tr_params['base_value'] + \
                     (r_norm * tr_params['r_weight'] + 
                      g_norm * tr_params['g_weight'] + 
                      b_norm * tr_params['b_weight']) * tr_params['scale_factor']
            
            # 预测茶褐素（TB）
            tb_params = self.default_params['tb']
            tb_pred = tb_params['base_value'] + \
                     (r_norm * tb_params['r_weight'] + 
                      g_norm * tb_params['g_weight'] + 
                      b_norm * tb_params['b_weight']) * tb_params['scale_factor']
            
            # 确保预测值在合理范围内
            tf_pred = self._clip_value('tf', tf_pred)
            tr_pred = self._clip_value('tr', tr_pred)
            tb_pred = self._clip_value('tb', tb_pred)
            
            # 计算置信度（基于亮度和颜色平衡度）
            # 亮度越接近 0.5，颜色越平衡，置信度越高
            brightness_confidence = 1.0 - abs(brightness - 0.5) * 2.0
            brightness_confidence = max(0.3, min(0.8, brightness_confidence))
            
            # 添加一些随机波动，使结果更真实
            np.random.seed(int(r_mean + g_mean + b_mean))
            noise = np.random.normal(0, 0.05)
            
            tf_confidence = brightness_confidence + noise
            tr_confidence = brightness_confidence + noise
            tb_confidence = brightness_confidence + noise
            
            # 确保置信度在合理范围内
            tf_confidence = max(0.3, min(0.95, tf_confidence))
            tr_confidence = max(0.3, min(0.95, tr_confidence))
            tb_confidence = max(0.3, min(0.95, tb_confidence))
            
            return {
                'tf': (round(tf_pred, 3), round(tf_confidence, 3)),
                'tr': (round(tr_pred, 3), round(tr_confidence, 3)),
                'tb': (round(tb_pred, 3), round(tb_confidence, 3))
            }
            
        except Exception as e:
            print(f"默认预测失败: {e}")
            return {
                'tf': (0.0, 0.0),
                'tr': (0.0, 0.0),
                'tb': (0.0, 0.0)
            }
    
    def _predict_with_model(self, features: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """
        使用机器学习模型进行预测
        
        Args:
            features: 图像颜色特征
            
        Returns:
            包含预测结果的字典
        """
        if self.scaler is None or self.feature_names is None:
            raise ValueError("模型相关文件未加载完成")
        
        try:
            # 提取特征向量
            feature_vector = self._feature_dict_to_vector(features)
            
            # 特征归一化
            if self.scaler is not None:
                feature_vector_scaled = self.scaler.transform([feature_vector])
            else:
                feature_vector_scaled = [feature_vector]
            
            # 预测茶黄素含量
            tf_pred, tf_confidence = self._predict_single('tf', feature_vector_scaled)
            
            # 预测茶红素含量
            tr_pred, tr_confidence = self._predict_single('tr', feature_vector_scaled)
            
            # 预测茶褐素含量
            tb_pred, tb_confidence = self._predict_single('tb', feature_vector_scaled)
            
            return {
                'tf': (tf_pred, tf_confidence),
                'tr': (tr_pred, tr_confidence),
                'tb': (tb_pred, tb_confidence)
            }
            
        except Exception as e:
            print(f"模型预测失败: {e}")
            return {}
    
    def _feature_dict_to_vector(self, features: Dict[str, Dict[str, float]]) -> np.ndarray:
        """
        将特征字典转换为模型输入所需的特征向量
        
        Args:
            features: 图像颜色特征
            
        Returns:
            一维numpy数组特征向量
        """
        try:
            # 按照预定义的特征名称顺序提取特征
            feature_vector = []
            
            for feature_name in self.feature_names:
                # 解析特征名称，如'RGB_R_mean' -> ('RGB', 'R_mean')
                parts = feature_name.split('_', 1)
                if len(parts) == 2:
                    space, name = parts
                    
                    # 获取特征值
                    value = features.get(space, {}).get(name, 0.0)
                    feature_vector.append(value)
                else:
                    # 如果格式不对，使用0.0作为默认值
                    feature_vector.append(0.0)
            
            return np.array(feature_vector, dtype=np.float64)
            
        except Exception as e:
            print(f"特征转换失败: {e}")
            return np.array([])
    
    def _predict_single(self, pigment_type: str, feature_vector: np.ndarray) -> Tuple[float, float]:
        """
        单一茶色素类型的预测（使用机器学习模型）
        
        Args:
            pigment_type: 茶色素类型 ('tf', 'tr', 'tb')
            feature_vector: 归一化后的特征向量
            
        Returns:
            预测值和置信度
        """
        try:
            model = self.models[pigment_type]
            
            # 进行预测
            pred = model.predict(feature_vector)[0]
            
            # 计算置信度
            confidence = self._calculate_confidence(pigment_type, pred)
            
            # 确保预测值在合理范围内
            pred = self._clip_value(pigment_type, pred)
            
            return round(pred, 3), round(confidence, 3)
            
        except Exception as e:
            print(f"{pigment_type}预测失败: {e}")
            return 0.0, 0.0
    
    def _calculate_confidence(self, pigment_type: str, pred: float) -> float:
        """
        计算预测结果的置信度
        
        Args:
            pigment_type: 茶色素类型
            pred: 预测值
            
        Returns:
            置信度（0-1之间）
        """
        # 根据预测值在正常范围内的位置计算置信度
        # 越接近范围中心，置信度越高
        
        if pigment_type == 'tf':
            min_val, max_val = self.tf_range
        elif pigment_type == 'tr':
            min_val, max_val = self.tr_range
        elif pigment_type == 'tb':
            min_val, max_val = self.tb_range
        else:
            return 0.0
        
        # 范围中心
        center = (min_val + max_val) / 2
        
        # 距离中心的相对距离
        rel_dist = abs(pred - center) / (max_val - min_val)
        
        # 转换为置信度，使用高斯分布模型
        confidence = np.exp(-5 * (rel_dist ** 2))
        
        return confidence
    
    def _clip_value(self, pigment_type: str, value: float) -> float:
        """
        将预测值限制在合理范围内
        
        Args:
            pigment_type: 茶色素类型
            value: 原始预测值
            
        Returns:
            限制后的值
        """
        if pigment_type == 'tf':
            return max(self.tf_range[0], min(self.tf_range[1], value))
        elif pigment_type == 'tr':
            return max(self.tr_range[0], min(self.tr_range[1], value))
        elif pigment_type == 'tb':
            return max(self.tb_range[0], min(self.tb_range[1], value))
        else:
            return value
    
    def get_model_info(self) -> Dict[str, Dict]:
        """
        获取模型信息
        
        Returns:
            包含模型信息的字典
        """
        model_info = {}
        
        if self.models_loaded:
            for pigment_type, model in self.models.items():
                if model is not None:
                    info = {
                        'name': f'{pigment_type}_model',
                        'type': type(model).__name__,
                        'is_trained': True
                    }
                    
                    # 尝试获取更多模型信息
                    if hasattr(model, 'get_params'):
                        params = model.get_params()
                        # 只记录关键参数
                        for param_name in ['kernel', 'C', 'epsilon']:
                            if param_name in params:
                                info[param_name] = params[param_name]
                    
                    model_info[pigment_type] = info
        else:
            model_info['mode'] = 'default'
            model_info['description'] = '使用默认预测模式，基于RGB颜色的简单映射'
        
        return model_info
    
    def save_predictions(self, predictions: Dict, output_path: str) -> bool:
        """
        保存预测结果到文件
        
        Args:
            predictions: 预测结果
            output_path: 输出文件路径
            
        Returns:
            保存成功返回True，失败返回False
        """
        try:
            # 转换为CSV格式
            import csv
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # 写入表头
                writer.writerow(['茶色素类型', '预测值(%)', '置信度'])
                
                # 写入数据
                for pigment_type, (pred, confidence) in predictions.items():
                    pigment_name = {
                        'tf': '茶黄素',
                        'tr': '茶红素',
                        'tb': '茶褐素'
                    }.get(pigment_type, pigment_type)
                    
                    writer.writerow([pigment_name, pred, confidence])
            
            return True
            
        except Exception as e:
            print(f"预测结果保存失败: {e}")
            return False
    
    def get_feature_importance(self, pigment_type: str) -> Optional[Dict[str, float]]:
        """
        获取特征重要性（如果模型支持）
        
        Args:
            pigment_type: 茶色素类型
            
        Returns:
            特征重要性字典，模型不支持时返回None
        """
        try:
            model = self.models[pigment_type]
            
            # 检查模型是否支持特征重要性
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                
                # 创建特征重要性字典
                importance_dict = {}
                for name, importance in zip(self.feature_names, importances):
                    importance_dict[name] = float(importance)
                
                # 按重要性排序
                importance_dict = dict(sorted(importance_dict.items(), 
                                            key=lambda item: item[1], 
                                            reverse=True))
                
                return importance_dict
                
            elif hasattr(model, 'coef_') and hasattr(self, 'feature_names'):
                # 对于线性模型，使用系数表示重要性
                coefs = model.coef_[0]
                
                importance_dict = {}
                for name, coef in zip(self.feature_names, coefs):
                    importance_dict[name] = float(abs(coef))
                
                importance_dict = dict(sorted(importance_dict.items(), 
                                            key=lambda item: item[1], 
                                            reverse=True))
                
                return importance_dict
                
            else:
                print(f"模型{type(model).__name__}不支持特征重要性获取")
                return None
                
        except Exception as e:
            print(f"获取特征重要性失败: {e}")
            return None
    
    def update_default_params(self, new_params: Dict):
        """
        更新默认预测模式的参数
        
        Args:
            new_params: 新的参数字典
        """
        try:
            # 更新参数
            for pigment_type, params in new_params.items():
                if pigment_type in self.default_params:
                    self.default_params[pigment_type].update(params)
            
            print("默认预测参数已更新")
            
        except Exception as e:
            print(f"更新参数失败: {e}")
    
    def reset_default_params(self):
        """重置默认预测参数为初始值"""
        self.default_params = {
            'tf': {
                'r_weight': 0.5,
                'g_weight': 0.3,
                'b_weight': 0.2,
                'base_value': 0.1,
                'scale_factor': 1.5
            },
            'tr': {
                'r_weight': 0.2,
                'g_weight': 0.5,
                'b_weight': 0.3,
                'base_value': 2.0,
                'scale_factor': 8.0
            },
            'tb': {
                'r_weight': 0.3,
                'g_weight': 0.2,
                'b_weight': 0.5,
                'base_value': 3.0,
                'scale_factor': 6.0
            }
        }
        print("默认预测参数已重置")