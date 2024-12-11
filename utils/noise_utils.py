import torch
def noise_model(img):
    dark = 1.0
    # print(img.shape)
    # print(aaaaa)
    img = img.float()
    h=img.shape[1]
    w=img.shape[2]
    # print(img.max())
    # print(aaaaaa)
    Vrst = 3.3
    Vref = 2.2
    IE = 0.4442 * 1.3
    Is_sigma = 0.1731 * 1.3
    VE = -0.0076
    Vs_sigma = 0.0497 * 1.3
    CE = 4.348
    Cs_sigma = 0.0128 * 1.3
    AlphaE = 1.0128
    Alphas_sigma = 0.0584
    #accumulator = torch.random.rand(3,h, w) * CE * (Vrst - Vref + VE) * 0.9
    noise_Idark = torch.randn( (3,h, w),device="cuda")*Is_sigma+IE
    Cs = torch.randn((3,h, w))*Cs_sigma+CE
    Vs = torch.randn((3,h, w))*Vs_sigma+VE
    It = torch.zeros((3,h, w))
    It = (noise_Idark + 20 * CE * (Vrst - Vref + VE) * img * dark)/(20 * CE * (Vrst - Vref + VE))
    return It