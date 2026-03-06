# -*- coding: utf-8 -*-
"""
茶色素智能检测系统 - 核心功能模块
Tea Pigment Intelligent Detection System - Core Functionality Module

这个包包含了系统的所有核心功能：
- 图像处理
- 颜色特征提取
- 色素含量预测
- 颜色校准
"""

__version__ = '1.0.0'
__author__ = 'Tea Detection Team'

# 导入核心类
try:
    from .image_processor import (
        ImageProcessor,
        CalibrationManager,
        ImageEnhancer
    )
    __all__ = [
        'ImageProcessor',
        'CalibrationManager',
        'ImageEnhancer'
    ]
except ImportError as e:
    print(f"警告：无法导入图像处理模块 - {e}")
    __all__ = []

try:
    from .pigment_predictor_fixed import PigmentPredictor
    if 'PigmentPredictor' not in __all__:
        __all__.append('PigmentPredictor')
except ImportError:
    # 如果修复版不存在，尝试导入原始版本
    try:
        from .pigment_predictor import PigmentPredictor
        if 'PigmentPredictor' not in __all__:
            __all__.append('PigmentPredictor')
    except ImportError as e:
        print(f"警告：无法导入预测器模块 - {e}")

# 模块信息
__doc__ += """
使用示例：
    >>> from core import ImageProcessor, PigmentPredictor
    >>> 
    >>> # 初始化图像处理器
    >>> processor = ImageProcessor()
    >>> img = processor.load_image('tea_soup.jpg')
    >>> features = processor.extract_color_features(img)
    >>> 
    >>> # 初始化预测器
    >>> predictor = PigmentPredictor()
    >>> predictor.load_models('models/')
    >>> predictions = predictor.predict(features)
    >>> 
    >>> # 输出结果
    >>> tf_value, tf_conf = predictions['tf']
    >>> print(f"茶黄素含量: {tf_value}% (置信度: {tf_conf})")
"""