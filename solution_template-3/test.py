"""
test.py

Запускает выбранную(ые) атаку(и) на всех методах вотермаркинга и сравнивает
полученные студентами метрики с эталонными результатами.
Использует run_single_attack.run_one для выполнения атак.

Пример:
python test.py --attacks dip vae --reference-dir ./data/test/results/ --out ./students_summary.json 
Если --attacks не указан — тестируются все атаки из ATTACKS (в run_single_attack.py).
Если --watermarks указан — тестируются только те, иначе — все доступные WATERMARKS.
"""

import os, sys
import argparse
import json
import unittest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR += "/watermarks/arwgan/"
sys.path.append(SCRIPT_DIR)

from eval import run_one, WATERMARKS, ATTACKS

def load_reference_for(watermark_name, reference_dir):
    """
    Ожидаем файл reference_dir/{watermark_name}.json или reference_dir/{watermark_name}__{attack}.json.
    По умолчанию пытаемся reference_dir/{watermark_name}.json.
    """
    candidate = os.path.join(reference_dir, f"{watermark_name}.json")
    if os.path.exists(candidate):
        with open(candidate, "r") as f:
            return json.load(f)
    # fallback: try to collect all per-attack files into a merged dict (если такой формат)
    merged = {}
    # ищем файлы reference_dir/{watermark_name}__*.json
    for fname in os.listdir(reference_dir):
        if fname.startswith(f"{watermark_name}__") and fname.endswith(".json"):
            with open(os.path.join(reference_dir, fname), "r") as f:
                data = json.load(f)
            merged.update(data)
    if merged:
        return merged
    raise FileNotFoundError(f"No reference JSON found for watermark '{watermark_name}' in {reference_dir}")


class MetricComparisonTests(unittest.TestCase):
    """
    Создаётся динамически для каждой пары (watermark, attack).
    Проверяет, что:
      * список изображений совпадает с эталоном
      * для каждой метрики абсолютная разница <= tolerance
    """
    @classmethod
    def configure(cls, watermark_name, attack_name, student_metrics, reference_metrics, tolerances):
        cls.watermark_name = watermark_name
        cls.attack_name = attack_name
        cls.student_metrics = student_metrics
        cls.reference_metrics = reference_metrics
        cls.tol_bit = tolerances["bit"]
        cls.tol_psnr = tolerances["psnr"]
        cls.tol_vmaf = tolerances["vmaf"]
        cls.tol_lpips = tolerances["lpips"]

    def test_images_overlap(self):
        ref_imgs = set(self.reference_metrics.keys())
        stu_imgs = set(self.student_metrics.keys())
        missing = ref_imgs - stu_imgs
        extra = stu_imgs - ref_imgs
        self.assertFalse(len(missing) > 0, f"Missing images in student metrics for {self.watermark_name}/{self.attack_name}: {sorted(list(missing))[:10]}")
        # допускаем дополнительные изображения, но сообщаем если есть
        if extra:
            print(f"Note: student produced extra images for {self.watermark_name}/{self.attack_name}: {len(extra)} (showing up to 10): {sorted(list(extra))[:10]}")

    def test_metric_differences_within_tolerance(self):
        diffs = {"bit": [], "psnr": [], "vmaf": [], "lpips": []}
        for img_name, ref_entry in self.reference_metrics.items():
            stu_entry = self.student_metrics.get(img_name)
            self.assertIsNotNone(stu_entry, f"Image {img_name} missing in student output for {self.watermark_name}/{self.attack_name}")

            keys = {
                "bit": f"{self.attack_name}_bit_accur",
                "psnr": f"{self.attack_name}_psnr",
                "vmaf": f"{self.attack_name}_vmaf",
                "lpips": f"{self.attack_name}_lpips"
            }
            for k, keyname in keys.items():
                self.assertIn(keyname, ref_entry, f"Reference missing key {keyname} for image {img_name}")
                self.assertIn(keyname, stu_entry, f"Student output missing key {keyname} for image {img_name}")

                ref_val = float(ref_entry[keyname])
                stu_val = float(stu_entry[keyname])

                if k in ["bit", "lpips"]:
                    dif = stu_val - ref_val
                else:
                    dif = ref_val - stu_val

                diffs[k].append(max(dif, 0))

        max_bit = max(diffs["bit"]) if diffs["bit"] else 0.0
        max_psnr = max(diffs["psnr"]) if diffs["psnr"] else 0.0
        max_vmaf = max(diffs["vmaf"]) if diffs["vmaf"] else 0.0
        max_lpips = max(diffs["lpips"]) if diffs["lpips"] else 0.0



        self.assertLessEqual(max_bit, self.tol_bit, f"Max bit accuracy diff {max_bit} > tol {self.tol_bit} for {self.watermark_name}/{self.attack_name}")
        
        if "dip" not in self.attack_name:
            self.assertLessEqual(max_psnr, self.tol_psnr, f"Max PSNR diff {max_psnr} > tol {self.tol_psnr} for {self.watermark_name}/{self.attack_name}")
            self.assertLessEqual(max_vmaf, self.tol_vmaf, f"Max VMAF diff {max_vmaf} > tol {self.tol_vmaf} for {self.watermark_name}/{self.attack_name}")
        
        self.assertLessEqual(max_lpips, self.tol_lpips, f"Max LPIPS diff {max_lpips} > tol {self.tol_lpips} for {self.watermark_name}/{self.attack_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks", nargs="+", required=False, help="Attacks to test (default: all).")
    parser.add_argument("--watermarks", nargs="+", required=False, help="Watermarks to test (default: all).")
    parser.add_argument("--path-template", required=False, default="./data/test/{watermark}/", help="Шаблон пути к папке с изображениями. По умолчанию './data/test/{watermark}/'")
    parser.add_argument("--reference-dir", required=False, default="./data/test/results/", help="Папка с эталонными json (по watermark).")
    parser.add_argument("--device", required=False, default=None, help="device to use (cpu/cuda/xpu)")
    parser.add_argument("--out", required=False, default="./results/students_summary.json", help="Куда записать сводный JSON с результатами студентов.")
    parser.add_argument("--save-images", required=False, default=None, help="Директория для сохранения атакованных изображений (опционально).")
    parser.add_argument("--tol-bit", type=float, default=0.08)
    parser.add_argument("--tol-psnr", type=float, default=2.0)
    parser.add_argument("--tol-vmaf", type=float, default=3.0)
    parser.add_argument("--tol-lpips", type=float, default=0.08)
    args = parser.parse_args()

    selected_attacks = args.attacks if args.attacks else list(ATTACKS.keys())

    selected_watermarks = args.watermarks if args.watermarks else list(WATERMARKS.keys())

    if 'your' in selected_attacks:
        selected_attacks.remove('your')

    # проверяем имена атак/вотермарков
    for a in selected_attacks:
        if a not in ATTACKS:
            raise ValueError(f"Unknown attack: {a}")
    for w in selected_watermarks:
        if w not in WATERMARKS:
            raise ValueError(f"Unknown watermark: {w}")

    summary = {"per_watermark": {}}
    any_failed = False

    for w in selected_watermarks:
        ref = load_reference_for(w, args.reference_dir)
        summary["per_watermark"].setdefault(w, {})

        for a in selected_attacks:
            print(f"\n--- Testing attack '{a}' on watermark '{w}' ---")
            img_path = args.path_template.format(watermark=w)
            # где сохранить student images для этой парочки
            student_images_dir = None
            if args.save_images:
                student_images_dir = os.path.join(args.save_images, f"{w}__{a}")
                os.makedirs(student_images_dir, exist_ok=True)

            # запускаем атаку (переиспользуем run_one)
            student_metrics = run_one(w, a, img_path, device=args.device, save_images_dir=student_images_dir, out_json=None)

            # создаём unit тесты для этой пары
            TestCls = type(f"Compare_{w}_{a}", (MetricComparisonTests,), {})
            tol = {"bit": args.tol_bit, "psnr": args.tol_psnr, "vmaf": args.tol_vmaf, "lpips": args.tol_lpips}
            TestCls.configure(w, a, student_metrics, ref, tol)

            suite = unittest.TestSuite()
            suite.addTest(TestCls("test_images_overlap"))
            suite.addTest(TestCls("test_metric_differences_within_tolerance"))

            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)

            summary["per_watermark"][w][a] = {
                "testsRun": result.testsRun,
                "failures": [s for _, s in result.failures],
                "errors": [s for _, s in result.errors],
                "wasSuccessful": result.wasSuccessful(),
                "n_images_student": len(student_metrics),
                "n_images_reference": len(ref)
            }

            # сохраняем per-pair metrics в отдельный файл
            out_dir = os.path.dirname(args.out) or "."
            os.makedirs(out_dir, exist_ok=True)
            pair_out = os.path.join(out_dir, f"students_{w}__{a}.json")
            with open(pair_out, "w") as f:
                json.dump(student_metrics, f, indent=4)
            summary["per_watermark"][w][a]["student_metrics_file"] = pair_out

            if not result.wasSuccessful():
                any_failed = True

    # сохраняем окончательную сводку
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"\nSaved overall summary to {args.out}")

    if any_failed:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()