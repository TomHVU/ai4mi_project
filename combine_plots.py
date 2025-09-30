from PIL import Image

def combine_images_to_a4(img_paths, output_path="combined_training_metrics.png"):
    # A4 size in pixels at 300 DPI (standard print resolution)
    a4_width, a4_height = 2480, 3508

    # Open images
    images = [Image.open(p).convert("RGBA") for p in img_paths]

    # Create blank A4 canvas (white background)
    canvas = Image.new("RGBA", (a4_width, a4_height), "WHITE")

    # Calculate target size for each image (2x2 grid)
    target_width = a4_width // 2
    target_height = a4_height // 2

    # Resize images to fit
    resized = [img.resize((target_width, target_height), Image.LANCZOS) for img in images]

    # Paste images into 2x2 layout
    canvas.paste(resized[0], (0, 0))
    canvas.paste(resized[1], (target_width, 0))
    canvas.paste(resized[2], (0, target_height))
    canvas.paste(resized[3], (target_width, target_height))

    # Save as PNG
    canvas.save(output_path, "PNG")
    print(f"Saved combined image to {output_path}")

if __name__ == "__main__":
    # Example usage: replace with your PNG file names
    imgs = ["results/SEGTHOR/ce/dice_tra.png", 
            "results/SEGTHOR/ce/dice_val.png", 
            "results/SEGTHOR/ce/loss_tra.png", 
            "results/SEGTHOR/ce/loss_val.png"]
    combine_images_to_a4(imgs, "results/SEGTHOR/ce/combined_a4.png")
