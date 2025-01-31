from torch import nn

# ----------------------------------------------------------------------------------------------------------
class CustomLoss(nn.Module):        # (1-l)*L_MSE + l*L_phys
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true, y_phys, l_p):
        mse_loss_value = self.mse_loss(y_pred, y_true)                      # loss w.r.t. data
        phys_loss_value = self.mse_loss(y_pred, y_phys)                     # loss w.r.t. physical model
        total_loss = (1 - l_p) * mse_loss_value + l_p * phys_loss_value     # total loss, weighted by l_p-factor

        loss_components = (mse_loss_value, phys_loss_value)
        return total_loss, loss_components

# ----------------------------------------------------------------------------------------------------------
class CustomLoss_2(nn.Module):      # L_MSE + l*L_phys
    def __init__(self):
        super(CustomLoss_2, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true, y_phys, l_p):
        mse_loss_value = self.mse_loss(y_pred, y_true)                      # loss w.r.t. data
        phys_loss_value = self.mse_loss(y_pred, y_phys)                     # loss w.r.t. physical model
        total_loss = mse_loss_value + l_p * phys_loss_value     # total loss, weighted by l_p-factor
        
        loss_components = (mse_loss_value, phys_loss_value)
        return total_loss, loss_components

# ----------------------------------------------------------------------------------------------------------
class CustomLoss_3(nn.Module):      # L_MSE + l*L_phys  + l*L_constraint
    def __init__(self):
        super(CustomLoss_3, self).__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true, y_phys, l_p):
        mse_loss_value = self.mse_loss(y_pred, y_true)                              # loss w.r.t. data
        phys_loss_value = self.mse_loss(y_pred, y_phys)                             # loss w.r.t. physical model
        constraint = torch.mean((torch.clamp(y_pred - y_phys, min=0)) ** 2)         # penalty for y_pred > y_phys
        total_loss = mse_loss_value + l_p * phys_loss_value + l_p * constraint      # total loss, weighted by l_p-factor
        loss_components = (mse_loss_value, phys_loss_value, constraint)
        return total_loss, loss_components