import torch.nn as nn
import torchvision.models as models
import numpy as np
import os

from art.estimators.classification import PyTorchClassifier
import timm
from torch_nets import (
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
)
import warnings
import pytorch_fid.fid_score as fid_score

warnings.filterwarnings("ignore")


def model_selection(name):
    if name == "convnext":
        model = models.convnext_base(pretrained=True)
    elif name == "resnet":
        model = models.resnet50(pretrained=True)
    elif name == "vit":
        model = models.vit_b_16(pretrained=True)
    elif name == "swin":
        model = models.swin_b(weights=models.Swin_B_Weights.IMAGENET1K_V1)
    elif name == "vgg":
        model = models.vgg19(pretrained=True)
    elif name == "mobile":
        model = models.mobilenet_v2(pretrained=True)
    elif name == "inception":
        model = models.inception_v3(pretrained=True)
    elif name == "deit-b":
        model = timm.create_model(
            'deit_base_patch16_224',
            pretrained=True
        )
    elif name == "deit-s":
        model = timm.create_model(
            'deit_small_patch16_224',
            pretrained=True
        )
    elif name == "mixer-b":
        model = timm.create_model(
            'mixer_b16_224',
            pretrained=True
        )
    elif name == "mixer-l":
        model = timm.create_model(
            'mixer_l16_224',
            pretrained=True
        )
    elif name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    elif name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
        model_path = os.path.join("pretrained_models", name + '.npy')
        model = net.KitModel(model_path)
    else:
        raise NotImplementedError("No such model!")
    return model.cuda()


def model_transfer(clean_img, adv_img, label, res, save_path, target_label):
    log = open(os.path.join(save_path, "log.txt"), mode="w", encoding="utf-8")
    models_transfer_name = ["resnet", "vgg", "mobile", "inception", "convnext", "vit", "swin",]
    all_clean_accuracy = []
    all_adv_accuracy = []
    for name in models_transfer_name:
        print("\n*********Transfer to {}********".format(name))
        print("\n*********Transfer to {}********".format(name), file=log)
        model = model_selection(name)
        model.eval()
        f_model = PyTorchClassifier(
            model=model,
            clip_values=(0, 1),
            loss=nn.CrossEntropyLoss(),
            input_shape=(3, res, res),
            nb_classes=1000,
            preprocessing=(np.array([0.5, 0.5, 0.5]), np.array([0.5, 0.5, 0.5])) if "adv" in name else (
                np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
            device_type='gpu',
        )

        clean_pred = f_model.predict(clean_img, batch_size=50)
        # print(label.shape)
        # print(target_label)
        target_labels = np.ones(len(label)) * target_label
        # print(target_labels)
        accuracy = np.sum((np.argmax(clean_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
            np.argmax(clean_pred, axis=1) == label) / len(label)
        target_accuracy = np.sum((np.argmax(clean_pred, axis=1)) == target_labels) / len(target_labels)
        print("Accuracy on benign examples: {}%".format(accuracy * 100))
        print("Target Accuracy on benign examples: {}%".format(target_accuracy * 100))
        print("Accuracy on benign examples: {}%".format(accuracy * 100), file=log)
        print("Target Accuracy on benign examples: {}%".format(target_accuracy * 100), file=log)
        all_clean_accuracy.append(accuracy * 100)

        adv_pred = f_model.predict(adv_img, batch_size=50)
        accuracy = np.sum((np.argmax(adv_pred, axis=1) - 1) == label) / len(label) if "adv" in name else np.sum(
            np.argmax(adv_pred, axis=1) == label) / len(label)
        target_accuracy = np.sum((np.argmax(adv_pred, axis=1)) == target_labels) / len(target_labels)
        print("Accuracy on adversarial examples: {}%".format(accuracy * 100))
        print("Target Accuracy on adversarial examples: {}%".format(target_accuracy * 100))
        print("Accuracy on adversarial examples: {}%".format(accuracy * 100), file=log)
        print("Target Accuracy on adversarial examples: {}%".format(target_accuracy * 100), file=log)

        all_adv_accuracy.append(accuracy * 100)

    print("clean_accuracy: ", "\t".join([str(x) for x in all_clean_accuracy]), file=log)
    print("adv_accuracy: ", "\t".join([str(x) for x in all_adv_accuracy]), file=log)


    log.close()

