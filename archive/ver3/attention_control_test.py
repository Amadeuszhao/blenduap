from typing import Optional, Union, Tuple, List, Dict
import abc
import seq_aligner
import torch
import ptp_utils_test
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import torchvision.transforms as transforms
import os
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
# MY_TOKEN = '<replace with your token>'
LOW_RESOURCE = False
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=MY_TOKEN).to(device)
tokenizer = StableDiffusionPipeline.from_pretrained("/data/plum/stable_diffusion").tokenizer
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class EmptyControl(AttentionControl):
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

# class LocalBlend:

#     def __call__(self, x_t, attention_store):
#         k = 1
#         maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
#         maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, self.max_num_words) for item in maps]
#         maps = torch.cat(maps, dim=1)
#         maps = (maps * self.alpha_layers).sum(-1).mean(1)
#         mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
#         mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
#         mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
#         mask = mask.gt(self.threshold)
#         mask = (mask[:1] + mask[1:]).float()
#         x_t = x_t[:1] + mask * (x_t - x_t[:1])
#         return x_t
       
#     def __init__(self, prompts: List[str], words: List[List[str]], max_num_words,tokenizer=None,threshold=.3,):
#         self.max_num_words = max_num_words
#         alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, max_num_words)
#         for i, (prompt, words_) in enumerate(zip(prompts, words)):
#             if type(words_) is str:
#                 words_ = [words_]
#             for word in words_:
#                 ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
#                 alpha_layers[i, :, :, :, :, ind] = 1
#         self.alpha_layers = alpha_layers.to(device)
#         self.threshold = threshold
    
class MaskPeturb:
    counter = 0
    def get_mask(self, maps, alpha, use_pool, x_t):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        # mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, controller):
            #print(self.counter)
            #print(self.start_blend)
        attention_store = controller.attention_store
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1,16, 16, self.max_num_words) for item in maps]
            #print('most important',self.alpha_layers.shape[0])
            #print('second important',self.substruct_layers)
        maps = torch.cat(maps, dim=1)
        #np.save('maps_'+str(LocalBlend.counter)+'.npy',maps.cpu().numpy())
        mask = self.get_mask(maps, self.alpha_layers, True,x_t)
        if self.substruct_layers is not None:
            maps_sub = ~self.get_mask(maps, self.substruct_layers, False,x_t)
            mask = mask * maps_sub
        mask = mask.float()

            #print('this is the shape of x_t',x_t.shape)
            #print('this is the shape of x_t[:1] ',x_t[:1].shape)
            #print('this is the shape of x_t - x_t[:1]',(x_t - x_t[:1]).shape)

            #print('this is the shape of mask',mask.shape)
            # np.save('mask_final'+'.npy',mask.cpu().numpy())
        return mask
       
    def __init__(self, prompts: List[str], words: [List[List[str]]],tokenizer=None,max_num_words=77, substruct_words=None,  th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1,max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils_test.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.max_num_words = max_num_words
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, max_num_words)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils_test.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        
        self.alpha_layers = alpha_layers.to(device)
        self.counter = 0 
        self.th=th                

class LocalBlend:
    counter = 0
    def get_mask(self, maps, alpha, use_pool, x_t):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        # print(self.counter)
        result = x_t
        if self.counter > self.start_blend:
            #print(self.counter)
            #print(self.start_blend)
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1,16, 16, self.max_num_words) for item in maps]
            #print('most important',self.alpha_layers.shape[0])
            #print('second important',self.substruct_layers)
            maps = torch.cat(maps, dim=1)
            # np.save('maps_'+str(LocalBlend.counter)+'.npy',maps.cpu().numpy())
            mask = self.get_mask(maps, self.alpha_layers, True,x_t)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False,x_t)
                mask = mask * maps_sub
            mask = mask.float()
            LocalBlend.counter += 1
            result = x_t
            #print('this is the shape of x_t',x_t.shape)
            #print('this is the shape of x_t[:1] ',x_t[:1].shape)
            #print('this is the shape of x_t - x_t[:1]',(x_t - x_t[:1]).shape)

            #print('this is the shape of mask',mask.shape)
            # np.save('mask_final'+'.npy',mask.cpu().numpy())
            if os.path.exists('mask_final.npy'):
                print('use pre mask')
                result = x_t
            else:
                result = x_t = x_t[:1] + mask * (x_t - x_t[:1])
                print('save mask now')
                np.save('mask_final'+'.npy',mask.cpu().numpy())
            #        print(mask)
            # original_latent = x_t[:1]
            # latent = (mask * (x_t - x_t[:1]))
            # latent = latent[1:]
            # zeros_tensor = torch.zeros(1,4,64,64).cuda()
            # zeros_tensor.requires_grad_(False)
            # print('this should optimize')
            # print('this is the compute perturb',id(self.perturb))   
            # mask = torch.tensor(np.load('mask_final.npy')).cuda()
            # print(mask)

            # original_latent = x_t[:1]
            # latent = x_t[1:]
            # normalize_latent , mean,  std = normalize(latent)
            # perturb_latent = unnormalize(normalize_latent+self.perturb,mean,std)
            # print('this is perturb grad 2',self.perturb.requires_grad)
            # print('this is perturb grad 3',perturb_latent.requires_grad)
            # middle = torch.cat([torch.zeros_like(perturb_latent).cuda(), perturb_latent-original_latent], dim=0)
            # x_t = mask * middle + original_latent
            # print('this is x_t grad 3', x_t.requires_grad)
            # print('this is x_t grad2 ', x_t.grad)
            # x_t = torch.cat([original_latent,latent])

        return result
       
    def __init__(self, prompts: List[str], words: [List[List[str]]],tokenizer=None,max_num_words=77,num_ddim_steps=50, substruct_words=None, start_blend=0.99 , th=(.3, .3),perturb=None):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1,max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils_test.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.max_num_words = max_num_words
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, max_num_words)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils_test.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * num_ddim_steps)
        self.counter = 0 
        self.th=th    
        # print('this is the init perturb',id(perturb))   
        self.perturb = perturb                     
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        # print('this is perturb grad1',x_t.grad)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils_test.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller

def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]],):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils_test.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

from PIL import Image

def aggregate_attention(prompts:List[str],attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def show_cross_attention(prompts:List[str],attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts,attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils_test.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils_test.view_images(np.stack(images, axis=0))
     

def show_self_attention_comp(prompts:List[str], attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils_test.view_images(np.concatenate(images, axis=1))
    
def run_and_display(prompts, controller, num_diffusion_steps = 50,guidance_scale=7.5, ldm_stable=None,latent=None, run_baseline=False, generator=None,uncond_embeddings=None,perturb=None ):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    # prompts = [prompts[0],prompts[0]]
    if perturb != None:
        print('perturb is not None')
    images, x_t = ptp_utils_test.text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, num_inference_steps=num_diffusion_steps, guidance_scale=guidance_scale, generator=generator, uncond_embeddings=uncond_embeddings,perturb=perturb)
    # ptp_utils.view_images(images)
    return images, x_t

def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None,num_ddim_steps=50,tokenizer=None,perturb=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        # print('this is the make control id',id(perturb))
        lb = LocalBlend(prompts, blend_words,tokenizer=tokenizer,num_ddim_steps=num_ddim_steps,perturb=perturb)
    if is_replace_controller:
        controller = AttentionReplace(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, num_ddim_steps, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller
