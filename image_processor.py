# -*- coding: utf-8 -*-
"""
茶色素智能检测系统 - 图像处理模块
Tea Pigment Intelligent Detection System - Image Processing Module
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


class ImageProcessor:
    """图像处理器 - 负责图像预处理和颜色特征提取"""
    
    def __init__(self):
        """初始化图像处理器"""
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        加载图像文件
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            numpy数组格式的图像，失败返回None
        """
        try:
            # 使用OpenCV读取图像
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # OpenCV读取的是BGR格式，转换为RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        except Exception as e:
            print(f"图像加载失败: {e}")
            return None
    
    def preprocess_image(self, image: np.ndarray, 
                     target_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """
        预处理图像：调整大小、归一化
        
        Args:
            image: 原始图像
            target_size: 目标尺寸
            
        Returns:
            预处理后的图像
        """
        # 调整图像大小，保持宽高比
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return resized
    
    def extract_roi(self, image: np.ndarray, 
                  region_ratio: float = 0.6) -> np.ndarray:
        """
        提取图像中心区域（ROI），避免边缘干扰
        
        Args:
            image: 输入图像
            region_ratio: 提取区域比例（中心区域占图像的比例）
            
        Returns:
            提取的中心区域图像
        """
        h, w = image.shape[:2]
        
        # 计算ROI的坐标
        roi_w = int(w * region_ratio)
        roi_h = int(h * region_ratio)
        
        x1 = (w - roi_w) // 2
        y1 = (h - roi_h) // 2
        x2 = x1 + roi_w
        y2 = y1 + roi_h
        
        # 提取ROI
        roi = image[y1:y2, x1:x2]
        
        return roi
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        提取多种颜色空间特征
        
        Args:
            image: 输入图像（RGB格式）
            
        Returns:
            包含多种颜色特征的字典
        """
        features = {}
        
        # 1. RGB颜色空间特征
        rgb_features = self._extract_rgb_features(image)
        features['RGB'] = rgb_features
        
        # 2. HSV颜色空间特征
        hsv_features = self._extract_hsv_features(image)
        features['HSV'] = hsv_features
        
        # 3. LAB颜色空间特征
        lab_features = self._extract_lab_features(image)
        features['LAB'] = lab_features
        
        # 4. YCrCb颜色空间特征
        ycrcb_features = self._extract_ycrcb_features(image)
        features['YCrCb'] = ycrcb_features
        
        return features
    
    def _extract_rgb_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取RGB颜色特征"""
        # 计算各通道的均值、标准差
        r_mean = np.mean(image[:, :, 0])
        g_mean = np.mean(image[:, :, 1])
        b_mean = np.mean(image[:, :, 2])
        
        r_std = np.std(image[:, :, 0])
        g_std = np.std(image[:, :, 1])
        b_std = np.std(image[:, :, 2])
        
        # 计算R/B和R/G比值（红茶品质相关指标）
        rb_ratio = r_mean / (b_mean + 1e-6)
        rg_ratio = r_mean / (g_mean + 1e-6)
        
        return {
            'R_mean': float(r_mean),
            'G_mean': float(g_mean),
            'B_mean': float(b_mean),
            'R_std': float(r_std),
            'G_std': float(g_std),
            'B_std': float(b_std),
            'R/B_ratio': float(rb_ratio),
            'R/G_ratio': float(rg_ratio)
        }
    
    def _extract_hsv_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取HSV颜色特征"""
        # 转换为HSV（OpenCV需要BGR输入）
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        
        # 计算各通道的均值、标准差
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        return {
            'H_mean': float(h_mean),
            'S_mean': float(s_mean),
            'V_mean': float(v_mean),
            'H_std': float(h_std),
            'S_std': float(s_std),
            'V_std': float(v_std)
        }
    
    def _extract_lab_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取LAB颜色特征"""
        # 转换为LAB
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        
        # 计算各通道的均值、标准差
        l_mean = np.mean(lab[:, :, 0])
        a_mean = np.mean(lab[:, :, 1])
        b_mean = np.mean(lab[:, :, 2])
        
        l_std = np.std(lab[:, :, 0])
        a_std = np.std(lab[:, :, 1])
        b_std = np.std(lab[:, :, 2])
        
        # a*值与红茶品质密切相关（红色程度）
        # b*值与黄色相关
        
        return {
            'L_mean': float(l_mean),
            'a_mean': float(a_mean),
            'b_mean': float(b_mean),
            'L_std': float(l_std),
            'a_std': float(a_std),
            'b_std': float(b_std)
        }
    
    def _extract_ycrcb_features(self, image: np.ndarray) -> Dict[str, float]:
        """提取YCrCb颜色特征"""
        # 转换为YCrCb
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
        
        # 计算各通道的均值、标准差
        y_mean = np.mean(ycrcb[:, :, 0])
        cr_mean = np.mean(ycrcb[:, :, 1])
        cb_mean = np.mean(ycrcb[:, :, 2])
        
        y_std = np.std(ycrcb[:, :, 0])
        cr_std = np.std(ycrcb[:, :, 1])
        cb_std = np.std(ycrcb[:, :, 2])
        
        return {
            'Y_mean': float(y_mean),
            'Cr_mean': float(cr_mean),
            'Cb_mean': float(cb_mean),
            'Y_std': float(y_std),
            'Cr_std': float(cr_std),
            'Cb_std': float(cb_std)
        }
    
    def get_color_histogram(self, image: np.ndarray, 
                       bins: int = 32) -> Dict[str, np.ndarray]:
        """
        计算颜色直方图
        
        Args:
            image: 输入图像
            bins: 直方图bin数量
            
        Returns:
            各颜色通道的直方图
        """
        histograms = {}
        
        # RGB直方图
        for i, channel in enumerate(['R', 'G', 'B']):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = hist.flatten()
            histograms[f'RGB_{channel}'] = hist
        
        return histograms
    
    def apply_white_balance(self, image: np.ndarray) -> np.ndarray:
        """
        简单的白平衡处理（灰度世界假设）
        
        Args:
            image: 输入图像
            
        Returns:
            白平衡后的图像
        """
        result = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)
