from diffusers import StableDiffusionPipeline
import torch

def gerar_imagem(prompt: str, saída: str = "output.png"):
    # 1) Seleciona dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # 2) Carrega o pipeline
    if device == "cuda":
        # GPU: carrega em float16 para economizar VRAM
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16
        )
        pipe.enable_attention_slicing()  # reduz o pico de memória
    else:
        # CPU: carrega em float32 (padrão), sem torch_dtype
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5"
        )
    pipe = pipe.to(device)

    # 3) Geração da imagem
    if device == "cuda":
        # FP16 autocast só funciona em CUDA
        with torch.autocast("cuda"):
            out = pipe(prompt, guidance_scale=7.5, num_inference_steps=50)
    else:
        # CPU: chamado padrão
        out = pipe(prompt, guidance_scale=7.5, num_inference_steps=50)

    imagem = out.images[0]
    imagem.save(saída)
    print(f"✅ Imagem salva em: {saída}")

if __name__ == "__main__":
    gerar_imagem(
        prompt="Uma floresta encantada ao pôr do sol, em estilo pintura a óleo",
        saída="floresta.png"
    )
