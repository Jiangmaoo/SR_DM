import torch
from torchvision.utils import make_grid, save_image

from data import valid_dataloader
from utils import Adder
import os
from torchvision.transforms import functional as F
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _valid(model, args, ep):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_set = valid_dataloader(args.data_dir, batch_size=1, num_workers=0)
    model.eval()
    psnr_adder = Adder()

    with torch.no_grad():
        print('Start Denoising Evaluation')
        factor = 8
        for idx, data in enumerate(data_set):
            input_img, label_img = data
            input_img = input_img.to(device)
            label_img = label_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')

            h_gt, w_gt = label_img.shape[2], label_img.shape[3]
            H_gt, W_gt = ((h_gt + factor) // factor) * factor, ((w_gt + factor) // factor * factor)
            padh_gt = H_gt - h_gt if h_gt % factor != 0 else 0
            padw_gt = W_gt - w_gt if w_gt % factor != 0 else 0
            label_img_ = f.pad(label_img, (0, padw_gt, 0, padh_gt), 'reflect')

            #if not os.path.exists(os.path.join(args.result_dir, '%d' % (ep))):
             #   os.mkdir(os.path.join(args.result_dir, '%d' % (ep)))
              #  result_evalu=(os.path.join(args.result_dir, '%d' % (ep)))

            pred = model(input_img,label_img_)[0][2]


            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)

            p_numpy = pred_clip.squeeze(0).cpu().numpy()

            #if args.save_image and idx<9:
            #    save_name = os.path.join(result_evalu, input_img)
            #    pred_clip += 0.5 / 255
            #    pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
            #    pred.save(save_name)

            label_numpy = label_img.squeeze(0).cpu().numpy()

            psnr = peak_signal_noise_ratio(p_numpy, label_numpy, data_range=1)

            psnr_adder(psnr)
            print('\r%03d'%idx, end=' ')

    print('\n')
    model.train()
    return psnr_adder.average()
