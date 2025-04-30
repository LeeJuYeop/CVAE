import torch
from model import CDCGAN
from dataset import train_dataset
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

import gradio as gr

model = CDCGAN.load_from_checkpoint("./cdcgan.ckpt")
model.eval()
model.to(torch.device("cpu"))

def generate_image(cls, z0, z1, z2, z3, z4):
    z = torch.tensor([[z0, z1, z2, z3, z4]], dtype=torch.float, device=model.device)
    # c = torch.tensor([cls], device=model.device)

    # 클래스 벡터를 one-hot 인코딩으로 변환
    c = torch.zeros(1, model.num_classes, device=model.device)  # 크기 (1, num_classes)
    c[0, cls] = 1  # 지정된 클래스 인덱스에 1을 설정

    x_pred = model(z,c)
    
    gen_img = to_pil_image(x_pred[0])

    return gen_img


with gr.Blocks() as demo:
    gr.Markdown("# GAN")
    with gr.Row(equal_height=True):
        cls = gr.Number(value=0, minimum=0, maximum=9)
        with gr.Column():
            z0 = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.01, label="z0")
            z1 = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.01, label="z1")
            z2 = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.01, label="z2")
            z3 = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.01, label="z3")
            z4 = gr.Slider(minimum=-1.0, maximum=1.0, value=0.0, step=0.01, label="z4")
        button = gr.Button("Generate", variant="primary")
    with gr.Row(equal_height=True):
        gen_image = gr.Image(height=250, width=250)
    
    button.click(
        generate_image, 
        inputs  = [cls, z0, z1, z2, z3, z4],
        outputs = [gen_image],
    )
    

# 실행
demo.launch()
