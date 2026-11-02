"""
Generates modern DigitVision AI branding logo in assets/logo.png.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def generate_logo():
    size = (256, 256)
    img = Image.new("RGBA", size, (10, 15, 29, 255))
    draw = ImageDraw.Draw(img)
    
    # Outer glowing rounded rectangle
    for offset in range(12, 0, -2):
        alpha = int(35 - offset * 2.5)
        draw.rounded_rectangle([20 - offset, 20 - offset, 236 + offset, 236 + offset], radius=48, outline=(99, 102, 241, max(alpha, 5)), width=2)
        
    draw.rounded_rectangle([20, 20, 236, 236], radius=44, fill=(17, 24, 39, 255), outline=(99, 102, 241, 180), width=3)
    
    # Neural network connections / tech motif
    nodes = [(80, 80), (176, 80), (70, 160), (186, 160), (128, 200)]
    for n1 in nodes:
        for n2 in nodes:
            if n1 != n2:
                draw.line([n1, n2], fill=(99, 102, 241, 60), width=1)
                
    for n in nodes:
        draw.ellipse([n[0] - 8, n[1] - 8, n[0] + 8, n[1] + 8], fill=(6, 182, 212, 220), outline=(255, 255, 255, 255), width=2)
        
    # Large central digit '8' styled like infinity / AI intelligence
    # Center text
    try:
        font = ImageFont.truetype("arial.ttf", 92)
    except Exception:
        font = ImageFont.load_default()
        
    draw.text((128, 128), "DV", fill=(255, 255, 255, 255), anchor="mm", font=font)
    
    img.save("assets/logo.png")
    print("Logo saved to assets/logo.png")

if __name__ == "__main__":
    generate_logo()

