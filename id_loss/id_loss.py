import torch
from torch import nn
from id_loss.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print("Loading ResNet ArcFace")
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode="ir_se"
        )
        ir_se50_path = "./pretrained_styleswin/model_ir_se50.pth"
        self.facenet.load_state_dict(torch.load(ir_se50_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet = self.facenet.cuda().eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x.cuda())
        return x_feats


    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)

        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count