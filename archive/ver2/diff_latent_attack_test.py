from attention_control_test import *
from ptp_utils_test import *
from PIL import Image
from tqdm import tqdm
from torch import optim
from utils import view_images
from distances import LpDistance
import other_attacks
import imagenet_label
from null_text_inversion import NullInversion
import torch.nn.functional as F
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
    guidance_scale: float = 7.5,
    image = None,
    model_name = 'resnet',
    save_path = 'result',
    res=224,
    iterations = 30,
    eps=80/255, 
):
    original_image = image.copy()
    label = torch.from_numpy(label).long().cuda()
    ori_label_name = imagenet_label.refined_Label[label.item()]
    target_label_name = imagenet_label.refined_Label[target_class]
    print('ori_label_name',ori_label_name)
    print('target_label_name',target_label_name)
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
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image, ori_label_name, offsets=(0,0,0,0), verbose=True)


    '''
    Image Cross Attention Increase
    '''
    
    
    prompts = [ori_label_name,target_label_name]
    cross_replace_steps = {'default_': .1,}
    self_replace_steps = 1.0
    blend_word = (((ori_label_name,), (target_label_name,)))
    perturb = torch.zeros(1, 4, 64, 64).cuda()
    perturb.requires_grad_(True)
    optimizer= optim.AdamW([perturb], lr=1e-2)
    print('this is perturb grad 1',perturb.requires_grad)
    #print('this is the input of perturb',id(perturb))
    for _ in tqdm(range(iterations)):
        controller = make_controller(prompts, True, cross_replace_steps, self_replace_steps, blend_word,num_ddim_steps=num_inference_steps,tokenizer=tokenizer,perturb=perturb)
        images, _ = run_and_display(prompts, controller,num_diffusion_steps = num_inference_steps, run_baseline=False, latent=x_t, uncond_embeddings=uncond_embeddings,ldm_stable=model)
        print('this is the shape for images',images.shape)
        image = images[1:]
        print('imgae grad',image.requires_grad)
        print('this is the shape for image',image.shape)
        new_size = (224,224)
        image = image.permute(0,3,1,2)
        out_image = F.interpolate(image,size = new_size,mode = 'bilinear')
        print('0',out_image.shape)
        out_image = out_image.permute(0,2,3,1)
        print('1',out_image.shape)
        # save_image = Image.fromarray((out_image.detach().cpu().squeeze().numpy()*255).astype('uint8'))
        # save_image.save('test.jpg')
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=out_image.dtype, device=out_image.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=out_image.dtype, device=out_image.device)
        out_image = out_image[:, :, :].sub(mean).div(std)
        print('2',out_image.shape)
        out_image = out_image.permute(0,3,1,2)
        print('3',out_image.shape)
        pred = classifier(out_image)
        target_class = torch.tensor([target_class]).cuda()
        # if optimizer is not None:
        #     print('optimize')
        #     optimizer.zero_grad()
        #     # loss_self = - criterion(self_pred,label)
        #     loss_cross = criterion(pred,target_class)
        #     loss = loss_cross 
        #     loss.backward()
        #     optimizer.step()
            # perturb.data = torch.clamp(perturb.data, -eps, eps)
    #print('this is after perturb',id(perturb))
    #print(perturb)
    output_image = out_image.detach().cpu().squeeze().numpy()
    output_image = Image.fromarray((output_image*255).astype('uint8').transpose(1,2,0))
    adversarial_image = output_image.copy()
    pred = classifier(out_image.cuda())
    pred_label = torch.argmax(pred, 1).detach()
    pred_accuracy = (torch.argmax(pred, 1).detach() == label).sum().item() / len(label)
    target_accuracy = (torch.argmax(pred, 1).detach() == target_class).sum().item() / 1
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
    view_images(adversarial_image, show=False, save_path=save_path + "_adv_image.png")

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