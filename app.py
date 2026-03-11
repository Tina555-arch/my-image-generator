# 轻量化AI图片生成应用（适配Hugging Face免费版）
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# 加载轻量级模型（避免显存不足）
model_id = "runwayml/stable-diffusion-v1-5"
# 关键优化：用float32+CPU模式，适配免费版无GPU环境
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    safety_checker=None  # 关闭安全检查，减少显存占用
)

# 生成图片函数（适配免费版）
def generate_image(prompt, width=384, height=384):
    # 限制分辨率，避免免费版内存不足
    width = min(width, 512)
    height = min(height, 512)
    # 生成图片
    image = pipe(
        prompt,
        width=width,
        height=height,
        num_inference_steps=20  # 减少生成步数，加快速度
    ).images[0]
    return image

# 构建界面（你的专属风格）
with gr.Blocks(title="我的AI画图工具") as demo:
    gr.Markdown("""
    # 🎨 我的专属AI画图工具
    输入文字描述，AI帮你生成图片～
    """)
    
    # 输入区
    with gr.Row():
        prompt = gr.Textbox(
            label="请输入图片描述",
            placeholder="例如：一只可爱的小狐狸在森林里看书，水彩风格，温暖色调",
            lines=3
        )
    
    # 参数调节区（轻量化）
    with gr.Row():
        width = gr.Slider(minimum=256, maximum=512, value=384, label="图片宽度")
        height = gr.Slider(minimum=256, maximum=512, value=384, label="图片高度")
    
    # 按钮+输出区
    generate_btn = gr.Button("✨ 生成图片", variant="primary")
    output_img = gr.Image(label="生成结果", type="pil")
    
    # 绑定生成事件
    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, width, height],
        outputs=output_img
    )

# 启动应用（适配Spaces环境）
demo.launch()
