from facenet_pytorch import InceptionResnetV1
import torch

print("Loading FaceNet...")
model = InceptionResnetV1(pretrained='vggface2').eval()

x = torch.rand(1, 3, 160, 160)
with torch.no_grad():
    emb = model(x)

print("Embedding shape:", emb.shape)
print("Loaded successfully!")
