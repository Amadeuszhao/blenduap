from attention_control import *
from ptp_utils import *
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images
from distances import LpDistance
import other_attacks
import imagenet_label
from null_text_inversion import NullInversion

import torch
import torchvision.transforms as transforms

def normalize(input_tensor):
    mean = input_tensor.mean(dim=(0, 2, 3))
    std = input_tensor.std(dim=(0, 2, 3))
    normalize = transforms.Normalize(mean, std)
    return normalize(input_tensor) ,mean ,std

def unnormalize(input_tensor, mean, std):
    unnormalize = transforms.Normalize(
    mean=-mean/std,
    std=1/std
)
    return unnormalize(input_tensor)

def preprocess(image, res=512):
    image = image.resize((res, res), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)[:, :3, :, :].cuda()
    return 2.0 * image - 1.0


def encoder(image, model, res=512):
    generator = torch.Generator().manual_seed(8888)
    image = preprocess(image, res)
    gpu_generator = torch.Generator(device=image.device)
    gpu_generator.manual_seed(generator.initial_seed())
    return 0.18215 * model.vae.encode(image).latent_dist.sample(generator=gpu_generator)


@torch.enable_grad()
def latent_attack(
    model,
    label,
    custom_loss,
    target_class,
    num_inference_steps: int = 20,
    guidance_scale: float = 2.5,
    image = None,
    model_name = 'resnet',
    save_path = 'result',
    res=224,
    iterations = 30,
    eps=60/255, 
):
    original_image = image.copy()
    label = torch.from_numpy(label).long().cuda()
    target_prompt = " ".join([imagenet_label.refined_Label[label.item()] for i in range(1, 1)])
    prompts = [imagenet_label.refined_Label[label.item()] + " " + target_prompt] * 2
    # ['shark']
    prompt = [prompts[0]]
    
    blend_word = [prompts[0],imagenet_label.refined_Label[target_class]]
    print('prompt generate',prompt)
    print('blend_word generate',blend_word)
    g_cpu = torch.Generator().manual_seed(8888) 

    ori_flag = 1
    adv_flag = 1
    target_flag = 0

    criterion = custom_loss
    
    model.vae.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.unet.requires_grad_(False)
    """
    Classifier output
    """
    tokenizer = model.tokenizer
    classifier = other_attacks.model_selection(model_name).eval()
    classifier.requires_grad_(False)
    
    height = width = res
    transform = transforms.Compose([
    transforms.Resize((res, res)),  # 调整大小为224x224像素
    transforms.ToTensor(),          # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])
    test_image = transform(image).unsqueeze(0)

    pred = classifier(test_image.cuda())
    pred_accuracy_clean = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    print("\nAccuracy on benign examples: {}%".format(pred_accuracy_clean * 100))

    logit = torch.nn.Softmax()(pred)
    print("gt_label:", label[0].item(), "pred_label:", torch.argmax(pred, 1).detach().item(), "pred_clean_logit",
          logit[0, label[0]].item())

    _, pred_labels = pred.topk(1, largest=True, sorted=True)


    '''
    Null Inversion Invert
    '''
    null_inversion = NullInversion(model,num_inference_steps,guidance_scale)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image, prompt[0], offsets=(0,0,0,0), verbose=True)


    '''
    Image Cross Attention Increase
    '''
    perturb = torch.zeros(1, 4, 64, 64).cuda()
    perturb.requires_grad_(True)
    mask = torch.zeros(1, 4, 64, 64,dtype=torch.int).cuda()
    mask.requires_grad_(False)
    unmask = torch.where(mask == 0, torch.tensor(1), torch.tensor(0)).cuda()
    unmask.requires_grad_(False)
    original_latent = x_t[:1] 
    latent = x_t[1:]
    optimizer= optim.AdamW([perturb], lr=5e-2)
    for _ in tqdm(range(iterations)):
        controller =  AttentionStore()
        normalize_latent , lmean,lstd = normalize(latent)
        perturb_latent_self = unnormalize(normalize_latent + mask * perturb,lmean,lstd)
        perturb_latent_cross = unnormalize(normalize_latent + unmask * perturb,lmean,lstd)
        self_latents = torch.cat([original_latent,perturb_latent_self])
        cross_latents = torch.cat([original_latent,perturb_latent_cross])
        self_output_image, x_t = run_and_display(prompt, controller, num_diffusion_steps = num_inference_steps,guidance_scale=guidance_scale, ldm_stable=model,latent=self_latents, run_baseline=False, generator=g_cpu,uncond_embeddings=uncond_embeddings )
        cross_output_image, x_t = run_and_display(prompt, controller, num_diffusion_steps = num_inference_steps,guidance_scale=guidance_scale, ldm_stable=model,latent=cross_latents, run_baseline=False, generator=g_cpu,uncond_embeddings=uncond_embeddings )
        mask_perturb = MaskPeturb(prompt, blend_word,tokenizer,)
        mask = mask_perturb(x_t,controller).cuda()
        unmask =  torch.where(mask == 0, torch.tensor(1), torch.tensor(0)).cuda()
        mask.requires_grad_(False)
        unmask.requires_grad_(False)
        self_output_image = Image.fromarray(self_output_image[0])
        self_output_image = transform(self_output_image).unsqueeze(0)

        cross_output_image = Image.fromarray(cross_output_image[0])
        cross_output_image = transform(cross_output_image).unsqueeze(0)
        self_pred = classifier(self_output_image.cuda())
        cross_pred = classifier(cross_output_image.cuda())
        target_class = torch.tensor([target_class]).cuda()

        if optimizer is not None:
            optimizer.zero_grad()
            loss_self = - criterion(self_pred,label)
            loss_cross = criterion(cross_pred,target_class)
            loss = loss_cross + loss_self
            loss = torch.tensor(loss,requires_grad=True)
            loss.backward()
            optimizer.step()
            perturb.data = torch.clamp(perturb.data, -eps, eps)

    with torch.no_grad():
        normalize_latent , lmean,lstd = normalize(latent)
        perturb_latent = unnormalize(normalize_latent + perturb,lmean,lstd)
        latents = torch.cat([original_latent,perturb_latent])
    output_image, x_t = run_and_display(prompt, controller, num_diffusion_steps = num_inference_steps,guidance_scale=guidance_scale, ldm_stable=model,latent=latents, run_baseline=False, generator=g_cpu,uncond_embeddings=uncond_embeddings )
    output_image = Image.fromarray(output_image[0])
    adversarial_image = output_image.copy()
    output_image = transform(output_image).unsqueeze(0)

    pred = classifier(output_image.cuda())
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    target_accuracy = (torch.argmax(pred, 1).detach() == target_class).sum().item() / len(target_class)
    print("Accuracy on adversarial examples: {}%".format(pred_accuracy * 100))
    if pred_accuracy_clean == 0.0:
        ori_flag = 0.0
    if pred_accuracy == 0.0:
        adv_flag = 0.0
    if target_accuracy == 1.0:
        target_flag = 1.0
    logit = torch.nn.Softmax()(pred)
    print("after_pred:", pred_label, logit[0, pred_label[0]])
    print("after_true:", label, logit[0, label[0]])
    original_image = original_image.resize((224, 224) ,resample=Image.LANCZOS)
    adversarial_image = adversarial_image.resize((224,224), resample=Image.LANCZOS)
    original_image = np.array(original_image)
    adversarial_image = np.array(adversarial_image)
    view_images(np.concatenate([original_image, adversarial_image]), show=False,
                save_path=save_path + "_diff_{}_image_{}.png".format(model_name,
                                                                     "ATKSuccess" if pred_accuracy == 0 else "Fail"))
    view_images(original_image, show=False, save_path=save_path + "_adv_image.png")

    L1 = LpDistance(1)
    L2 = LpDistance(2)
    Linf = LpDistance(float("inf"))

    #print("L1: {}\tL2: {}\tLinf: {}".format(L1(original_image/255.0, adversarial_image / 255.0), L2(original_image/255.0, adversarial_image / 255.0), Linf(original_image/255.0, adversarial_image / 255.0)))

    diff = adversarial_image / 255.0 - original_image/255.0
    diff = (diff - diff.min()) / (diff.max() - diff.min()) * 255

    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_relative.png")

    diff = (np.abs(adversarial_image / 255.0 - original_image/255.0) * 255).astype(np.uint8)
    view_images(diff.clip(0, 255), show=False,
                save_path=save_path + "_diff_absolute.png")

    # reset_attention_control(model)

    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=7, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionBefore.jpg".format(save_path))
    # utils.show_cross_attention(prompt, model.tokenizer, controller, res=7, from_where=("up", "down"),
    #                            save_path=r"{}_crossAttentionAfter.jpg".format(save_path), select=1)
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionBefore.jpg".format(save_path))
    # utils.show_self_attention_comp(prompt, controller, res=14, from_where=("up", "down"),
    #                                save_path=r"{}_selfAttentionAfter.jpg".format(save_path), select=1)

    return adversarial_image, ori_flag, adv_flag, target_flag 