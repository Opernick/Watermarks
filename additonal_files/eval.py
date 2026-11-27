"""
Запуск подсчёта метрик для одного метода атаки на одном методе нанесения вотермарки.
Пример:
python eval.py --watermark ARWGAN --attack dip --save-images ./out_images/
"""

import os, sys
import argparse
import json
from tqdm import tqdm

import urllib3

os.environ["HF_HUB_NUM_RETRIES"] = "0"

urllib3.util.retry.Retry.DEFAULT = urllib3.util.retry.Retry(0)

import torch
from torchvision.transforms import ToPILImage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR += "/watermarks/arwgan/"
sys.path.append(SCRIPT_DIR)

from datasets import ImageDataset
from watermarks import Stegastamp, StableSignatureDecoder, ARWGAN
from metrics import BitAccuracy, PSNR, VMAF, LPIPS
from attacks import VAEAttack, DIPAttack, AdversarialEmbedding, DIPAttackNoise, YourAttack

to_pil = ToPILImage()


WATERMARKS = {
    "ARWGAN": (ARWGAN, {"device": "cpu"}),
    "StegaStamp": (Stegastamp, {"device": "cpu"}),
    "StableSignature": (StableSignatureDecoder, {"device": "cpu"})
}

ATTACKS = {
    "vae": (VAEAttack, {"n_avg_imgs": 50, "noise_level": 1.1}),
    "adv_resnet": (AdversarialEmbedding, {"encoder": "resnet18"}),
    "adv_clip": (AdversarialEmbedding, {"encoder": "clip"}),
    "dip": (DIPAttack, {}),
    "dip_noise": (DIPAttackNoise, {}),
    "your" : (YourAttack, {})
}


def device_or_default(device_arg):
    if device_arg:
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_one(watermark_name, attack_name, path, device, save_images_dir=None, out_json=None):
    # проверка
    if watermark_name not in WATERMARKS:
        raise ValueError(f"Unknown watermark: {watermark_name}")
    if attack_name not in ATTACKS:
        raise ValueError(f"Unknown attack: {attack_name}")

    device = device_or_default(device)

    # метрики
    bit_accur = BitAccuracy()
    psnr = PSNR()
    vmaf = VMAF()
    lpips = LPIPS().to(device)

    w_cls, w_kwargs = WATERMARKS[watermark_name]
    # override device if present in kwargs
    w_kwargs = dict(w_kwargs)
    w_kwargs["device"] = device
    watermark = w_cls(**w_kwargs)

    a_cls, a_kwargs = ATTACKS[attack_name]
    a_kwargs = dict(a_kwargs)
    a_kwargs["device"] = device
    attack = a_cls(**a_kwargs)

    imgs = ImageDataset(path)

    results = {}
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True) if out_json else None
    if save_images_dir:
        os.makedirs(save_images_dir, exist_ok=True)

    quality_score_s = []
    bit_accur_s = []

    for img in tqdm(imgs, desc=f"{watermark_name}:{attack_name}"):
        # img is pair (filename, tensor)
        fname = img[0]
        orig_tensor = img[1].to(device)

        # decode original
        true_msg = watermark.decode(orig_tensor.unsqueeze(0).to("cpu"))
        # run attack
        attacked = attack.forward(orig_tensor).to(device, dtype=torch.float)

        # decode attacked
        attacked_msg = watermark.decode(attacked.to("cpu"))

        bit_accur_v = float(bit_accur(attacked_msg, true_msg).item())
        psnr_v = float(psnr(attacked, orig_tensor.unsqueeze(0)).item())
        vmaf_v = float(vmaf(attacked.to("cpu"), orig_tensor.unsqueeze(0).to("cpu")).item())
        lpips_v = float(lpips(attacked, orig_tensor.unsqueeze(0)).item())

        quality_score = 0.5 + (-2 * psnr_v + 400 * lpips_v - 1 * vmaf_v) / 1000

        quality_score_s.append(quality_score)
        bit_accur_s.append(bit_accur_v)

        img_res = {
            "true_msg": true_msg.tolist(),
            f"{attack_name}_msg": attacked_msg.tolist(),
            f"{attack_name}_bit_accur": bit_accur_v,
            f"{attack_name}_psnr": psnr_v,
            f"{attack_name}_vmaf": vmaf_v,
            f"{attack_name}_lpips": lpips_v
        }

        results[fname] = img_res

        if save_images_dir:
            try:
                pil = to_pil(attacked.clamp(0, 1).cpu()[0])
                out_path = os.path.join(save_images_dir, f"{watermark_name}__{attack_name}__{os.path.basename(fname)}")
                pil.save(out_path)
            except Exception as e:
                print(f"Warning: failed to save image {fname}: {e}")

    wm_performance = sum(bit_accur_s) / len(bit_accur_s)
    quality_degradation = sum(quality_score_s) / len(quality_score_s)
    score = (wm_performance ** 2 + quality_degradation ** 2) ** 0.5

    results["Watermark Performance"] = wm_performance
    results["Quality Degradation"] = quality_degradation
    results["Score"] = score

    if out_json:
        with open(out_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Saved metrics to {out_json}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watermark", required=True, help="Watermark name. Choices: " + ", ".join(WATERMARKS.keys()))
    parser.add_argument("--attack", required=True, help="Attack name. Choices: " + ", ".join(ATTACKS.keys()))
    parser.add_argument("--path", required=False, help="Папка с тестовыми изображениями (ImageDataset).", default=None)
    parser.add_argument("--device", required=False, help="device (cpu or cuda or xpu)", default=None)
    parser.add_argument("--out", required=False, help="json файл для сохранения метрик", default=f"./results_{args_placeholder}.json")
    parser.add_argument("--save-images", required=False, help="директория для сохранения атакованных изображений", default=None)
    args = parser.parse_args()

    # default path if not provided
    if args.path is None:
        args.path = f"./data/test/{args.watermark}/"

    # fallback for out filename
    if args.out and "{watermark}" in args.out:
        out_json = args.out.format(watermark=args.watermark, attack=args.attack)
    else:
        if args.out and args.out != f"./results_{args_placeholder}.json":
            out_json = args.out
        else:
            out_json = f"./results/{args.watermark}__{args.attack}.json"

    results = run_one(args.watermark, args.attack, args.path, args.device, save_images_dir=args.save_images, out_json=out_json)
    print("Done. Processed", len(results), "images.")
    print(f"Watermark Performance: {results['Watermark Performance']}, Quality Degradation: {results['Quality Degradation']}, Score: {results['Score']}")


if __name__ == "__main__":
    # small hack to set default out string only after argparse exists
    args_placeholder = "placeholder"
    main()