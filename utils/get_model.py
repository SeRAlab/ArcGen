from classifier_models import ResNet18, PreActResNet18, DenseNet121, MobileNetV2, ResNeXt29_2x64d, SENet18, SimpleDLA, SimpleCNN_cifar10_Model, SimpleCNN_mnist_Model, SimpleCNN_GTSRB, EfficientNetB0, ShuffleNetV2, SimpleCNN_Imagenet, ViT


def get_datainfo(arg):
    if arg.dataset in ["mnist", "cifar10"]:
        arg.num_classes = 10
    elif arg.dataset == "gtsrb":
        arg.num_classes = 43
    elif arg.dataset == "imagenet":
        arg.num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    if arg.dataset == "cifar10":
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "gtsrb":
        arg.input_height = 32
        arg.input_width = 32
        arg.input_channel = 3
    elif arg.dataset == "mnist":
        arg.input_height = 28
        arg.input_width = 28
        arg.input_channel = 1
    elif arg.dataset == "imagenet":
        arg.input_height = 224
        arg.input_width = 224
        arg.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

def get_model(arg):
    Model = None
    default_model_name = None

    if arg.dataset == "cifar10":
        Model = ResNet18
        default_model_name = "resnet18"
    elif arg.dataset == "gtsrb":
        Model = ResNet18
        default_model_name = "resnet18"
    elif arg.dataset == "imagenet":
        Model = ResNet18
        default_model_name = "resnet18"
    elif arg.dataset == "mnist":
        Model = ResNet18
        default_model_name = "resnet18"

    if arg.model is not None:
        if arg.model == "resnet18":
            Model = ResNet18
        elif arg.model == "densenet121":
            Model = DenseNet121
        elif arg.model == "mobilnetv2":
            Model = MobileNetV2
        elif arg.model == "resnext29":
            Model= ResNeXt29_2x64d
        elif arg.model == "senet18":
            Model = SENet18
        elif arg.model == "simpledla":
            Model = SimpleDLA
        elif arg.model == "preactresnet18":
            Model = PreActResNet18
        elif arg.model == "SimpleCNN_cifar10":
            Model = SimpleCNN_cifar10_Model
        elif arg.model == "SimpleCNN_mnist":
            Model = SimpleCNN_mnist_Model
        elif arg.model == "SimpleCNN_gtsrb":
            Model = SimpleCNN_GTSRB
        elif arg.model == "SimpleCNN_imagenet":
            Model = SimpleCNN_Imagenet
        elif arg.model == "efficientnetb0":
            Model = EfficientNetB0
        elif arg.model == "shufflenetv2":
            Model = ShuffleNetV2
        elif arg.model == "vit":
            Model = ViT
        else:
            arg.model = default_model_name
            raise Exception("Invalid Model")
    else:
        arg.model = default_model_name

    return Model
