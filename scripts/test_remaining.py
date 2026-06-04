#!/usr/bin/env python3
"""
剩余功能测试脚本
测试课程学习、视觉预处理、评估、数据生成
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image


def test_curriculum_learning():
    """测试课程学习"""
    print("\n" + "=" * 60)
    print("测试 1: 课程学习")
    print("=" * 60)

    from rl_vectorizer.curriculum.curriculum_manager import (
        CurriculumManager,
        DifficultyAnalyzer,
        DifficultyLevel,
    )

    curriculum = CurriculumManager()

    print(f"初始状态: {curriculum.get_stats()}")

    for step in [0, 50, 100, 500, 1000]:
        curriculum.current_step = step
        difficulty = curriculum.get_current_difficulty()
        level = curriculum.get_current_level()
        print(f"Step {step}: difficulty={difficulty:.3f}, level={level.name}")

    analyzer = DifficultyAnalyzer()

    test_samples = [
        {"svg": '<line x1="10" y1="10" x2="90" y2="90"/>'},
        {"svg": '<line x1="10" y1="10" x2="90" y2="90"/>\n' * 50},
    ]

    for sample in test_samples:
        features = analyzer.analyze_sample(sample)
        score = analyzer.get_difficulty_score(features)
        print(f"Sample difficulty: {score:.3f}, features: {features}")

    print("\n✓ 课程学习测试通过!")


def test_image_preprocessing():
    """测试图像预处理"""
    print("\n" + "=" * 60)
    print("测试 2: 图像预处理")
    print("=" * 60)

    from rl_vectorizer.utils.image_preprocessor import (
        ImagePreprocessor,
        TextRemovalPreprocessor,
        DrawingEnhancer,
        create_preprocessing_pipeline,
    )

    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    preprocessor = ImagePreprocessor(target_size=(128, 128))
    processed = preprocessor.preprocess(test_image)
    print(f"ImagePreprocessor: {test_image.shape} -> {np.array(processed).shape}")

    text_remover = TextRemovalPreprocessor()
    regions = text_remover.detect_text_regions(test_image)
    print(f"TextRemovalPreprocessor: detected {len(regions)} text regions")

    enhancer = DrawingEnhancer()
    enhanced = enhancer.enhance(test_image)
    print(f"DrawingEnhancer: {test_image.shape} -> {enhanced.shape}")

    pipeline = create_preprocessing_pipeline(
        ["resize", "denoise", "contrast"],
        config={"target_size": (256, 256)}
    )
    result = pipeline(test_image)
    result_np = np.array(result) if hasattr(result, 'shape') == False else result
    print(f"Pipeline: {test_image.shape} -> {result_np.shape}")

    print("\n✓ 图像预处理测试通过!")


def test_svg_evaluation():
    """测试 SVG 评估"""
    print("\n" + "=" * 60)
    print("测试 3: SVG 评估")
    print("=" * 60)

    from rl_vectorizer.evaluation.evaluator import (
        SVGEvaluator,
        CADEvaluator,
        BatchEvaluator,
    )

    svg_evaluator = SVGEvaluator()

    test_svg = '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><line x1="10" y1="10" x2="90" y2="90" stroke="black"/></svg>'
    test_image = Image.new("RGB", (100, 100), color="white")

    result = svg_evaluator.evaluate(test_svg, test_image)
    print(f"SVGEvaluator: total_score={result.total_score:.4f}, passed={result.passed}")
    print(f"  Metrics: {result.metrics}")
    print(f"  Details: {result.details}")

    cad_evaluator = CADEvaluator()
    cad_result = cad_evaluator.evaluate_cad_compatibility(test_svg)
    print(f"CADEvaluator: compatible={cad_result['compatible']}, layers={cad_result['layers']}")

    batch_evaluator = BatchEvaluator()
    batch_results = batch_evaluator.evaluate_batch([
        {
            "id": "test_1",
            "svg": test_svg,
            "target_image": test_image,
        }
    ])
    print(f"BatchEvaluator: {batch_results['summary']}")

    print("\n✓ SVG 评估测试通过!")


def test_synthetic_data():
    """测试合成数据生成"""
    print("\n" + "=" * 60)
    print("测试 4: 合成数据生成")
    print("=" * 60)

    from rl_vectorizer.data.synthetic_generator import (
        TowerSVGGenerator,
        SyntheticTowerDataset,
        generate_floor_plan,
    )

    tower_gen = TowerSVGGenerator(width=256, height=256)

    svg_code, metadata = tower_gen.generate(
        num_main_members=8,
        num_secondary_members=20,
        num_diaphragms=3,
    )
    print(f"TowerSVGGenerator: generated SVG with {len(svg_code)} chars")
    print(f"  Metadata: {metadata}")

    floor_plan = generate_floor_plan(width=256, height=256, num_rooms=4)
    print(f"generate_floor_plan: generated SVG with {len(floor_plan)} chars")

    print("\n✓ 合成数据生成测试通过!")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("剩余功能测试套件")
    print("=" * 60)

    tests = [
        test_curriculum_learning,
        test_image_preprocessing,
        test_svg_evaluation,
        test_synthetic_data,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test.__name__} 失败: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
