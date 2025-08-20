import argparse
import yaml
import os
import torch
import tqdm
import torchvision

from model.unet import Unet
from noise_scheduler.noise_scheduler import NoiseScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sample(model, scheduler, model_config, diffusion_config, train_config, class_labels=None):
    xt = torch.randn((train_config['num_samples'], model_config['img_channels'], model_config['img_size'],
                      model_config['img_size'])).to(device)

    for i in tqdm.tqdm(reversed(range(diffusion_config['num_timesteps']))):
        # 如果提供了类别标签，则将其传递给模型
        if class_labels is not None:
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device), class_labels)
        else:
            noise_pred = model(xt, torch.as_tensor(i).unsqueeze(0).to(device))
        xt, x0 = scheduler.sample_prev_timestep(xt, noise_pred, torch.as_tensor(i).to(device))

    # 只保存最后生成的图片
    sample_dir = "ddpm_sample"
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    imgs = torch.clamp(xt, -1., 1.).detach().cpu()
    imgs = (imgs + 1) / 2
    grid_xt = torchvision.utils.make_grid(imgs, nrow=10)
    grid_img = torchvision.transforms.ToPILImage()(grid_xt)
    grid_img.save(os.path.join(sample_dir, "sample_final.png"))
    grid_img.close()

    print("Done sampling...")


def infer(args, class_label=None):
    with open(args.config_path) as file:
        config = yaml.safe_load(file)
    print(config)
    ####################

    model_config = config['model_config']
    diffusion_config = config['diffusion_config']
    train_config = config['train_config']

    # 如果提供了类别标签，确保它是有效的
    if class_label is not None:
        if not (0 <= class_label <= 9):
            raise ValueError("类别标签必须是0-9之间的整数")
        # 创建批次的类别标签
        class_labels = torch.tensor([class_label] * train_config['num_samples'], device=device)

    ckpt_dir = os.path.join(train_config['task_name'], train_config['ckpt_name'])
    assert os.path.exists(ckpt_dir), print("No checkpoint file found")

    model = Unet(model_config).to(device)
    model.load_state_dict(torch.load(ckpt_dir, map_location=device))
    model.eval()

    scheduler = NoiseScheduler(diffusion_config)

    with torch.no_grad():
        if class_label is not None:
            sample(model, scheduler, model_config, diffusion_config, train_config, class_labels)
        else:
            sample(model, scheduler, model_config, diffusion_config, train_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm sampling...")
    parser.add_argument("--config_path", default="../config/default.yaml")
    parser.add_argument("--class_label", type=int, help="要生成的数字类别(0-9)")
    args = parser.parse_args()

    print("parse config_path:{}".format(args.config_path))

    # 如果没有提供类别标签，则获取用户输入
    if args.class_label is None:
        try:
            class_label = int(input("请输入要生成的数字(0-9): "))
            if not (0 <= class_label <= 9):
                raise ValueError
        except ValueError:
            print("输入无效，将生成随机数字")
            class_label = None
    else:
        class_label = args.class_label

    infer(args, class_label)
