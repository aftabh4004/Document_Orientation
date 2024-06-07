from model import DocumentOrientationModel
model_path = '../API/best_model.pth'
import torch

model = DocumentOrientationModel(8)
model_state_dict = torch.load(model_path, map_location='cpu')
model.load_state_dict(model_state_dict)
model.eval()

traced_model = torch.jit.trace(model, torch.rand((1, 3, 448, 448)))
torch.jit.save(traced_model, 'model_repository/doc-orientation/1/model.pt')
