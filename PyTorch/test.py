import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
from model import LYT
from dataloader import create_dataloaders
import os
import numpy as np
from torchvision.utils import save_image
import lpips
import pyiqa
import csv  # <-- Añadido para guardar CSV


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


def calculate_lpips(img1, img2, loss_fn, gt_mean=True):
    if gt_mean:
        img1_gray = img1.mean(dim=1, keepdim=True)
        img2_gray = img2.mean(dim=1, keepdim=True)
        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    img1 = (img1 - 0.5) * 2
    img2 = (img2 - 0.5) * 2

    score = loss_fn(img1, img2)
    return score.mean().item()


def calculate_niqe(img, niqe_metric):
    return niqe_metric(img).item()


def validate(model, dataloader, device, result_dir):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0
    total_niqe = 0

    loss_fn_lpips = lpips.LPIPS(net='alex').to(device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)

    per_image_metrics = []

    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)

            img_filename = f'result_{idx}.png'
            save_image(output, os.path.join(result_dir, img_filename))

            psnr = calculate_psnr(output, high)
            ssim = calculate_ssim(output, high)
            lpips_score = calculate_lpips(output, high, loss_fn_lpips)
            niqe_score = calculate_niqe(output, niqe_metric)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_score
            total_niqe += niqe_score

            per_image_metrics.append({
                'image': img_filename,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips_score,
                'niqe': niqe_score
            })

    num_samples = len(dataloader)
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    avg_lpips = total_lpips / num_samples
    avg_niqe = total_niqe / num_samples

    # Guardar métricas individuales en CSV
    csv_path = os.path.join(result_dir, 'metrics_per_image.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'psnr', 'ssim', 'lpips', 'niqe']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_image_metrics:
            writer.writerow(row)
        writer.writerow({
            'image': 'PROMEDIO',
            'psnr': avg_psnr,
            'ssim': avg_ssim,
            'lpips': avg_lpips,
            'niqe': avg_niqe
        })

    return {
        "PSNR": avg_psnr,
        "SSIM": avg_ssim,
        "LPIPS": avg_lpips,
        "NIQE": avg_niqe
    }


def main():
    #test_low = 'data/LOLv1/Test/input'
    #test_high = 'data/LOLv1/Test/target'
    test_low = 'data/LOLv2-syn/Test/Low'
    test_high = 'data/LOLv2-syn/Test/Normal'
    #test_low = 'data/LOLv2/Real_captured/Test/Low'
    #test_high = 'data/LOLv2/Real_captured/Test/Normal'
    weights_path = 'best_model.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = test_low.split('/')[1]
    result_dir = os.path.join('results', dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    _, test_loader = create_dataloaders(None, None, test_low, test_high, crop_size=None, batch_size=1)
    print(f'Test loader: {len(test_loader)}')

    model = LYT().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    metrics = validate(model, test_loader, device, result_dir)
    print(f"Validation PSNR: {metrics['PSNR']:.6f}, SSIM: {metrics['SSIM']:.6f}, LPIPS: {metrics['LPIPS']:.6f}, NIQE: {metrics['NIQE']:.6f}")


if __name__ == '__main__':
    main()
