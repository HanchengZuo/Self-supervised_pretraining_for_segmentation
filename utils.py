import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_segmentation_visual(image, mask, color_background, color_edge, transparency):
    image[mask == 2] = color_background
    edge_mask = mask == 3
    blend_color = np.array(color_edge, dtype=np.uint8)
    image[edge_mask] = (transparency * image[edge_mask] +
                        (1 - transparency) * blend_color).astype(np.uint8)
    return Image.fromarray(image)


def visualize_segmentation_comparison(images, true_masks, pred_masks, num_images,
                                      color_background=(255, 255, 255),
                                      color_edge=(255, 0, 0),
                                      transparency=0.75,
                                      subtitle="Segmentation Comparison",
                                      save_path=None, dpi=300):
    try:
        # Smaller font size, adjust the path as needed.
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=14)
        subtitlefont = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()

    for i in range(num_images):
        image_np = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        true_mask_np = true_masks[i].cpu().detach().numpy()
        pred_mask_np = pred_masks[i].cpu().detach().numpy()

        pil_orig = Image.fromarray(image_np)
        pil_true = create_segmentation_visual(
            image_np.copy(), true_mask_np, color_background, color_edge, transparency)
        pil_pred = create_segmentation_visual(
            image_np.copy(), pred_mask_np, color_background, color_edge, transparency)

        # Create a new image to place the three images side by side
        total_width = pil_orig.width * 3
        max_height = pil_orig.height + 60  # More height for the title and subtitle
        combined_image = Image.new(
            'RGB', (total_width, max_height), (255, 255, 255))

        # Paste each image
        combined_image.paste(pil_orig, (0, 60))
        combined_image.paste(pil_true, (pil_orig.width, 60))
        combined_image.paste(pil_pred, (pil_orig.width * 2, 60))

        # Draw the titles
        draw = ImageDraw.Draw(combined_image)
        draw.text((60, 40), "Original Image", font=font, fill="black")
        draw.text((pil_orig.width + 30, 40),
                  "True Segmentated Image", font=font, fill="black")
        draw.text((pil_orig.width * 2 + 10, 40),
                  "Predicted Segmentated Image", font=font, fill="black")
        draw.text((total_width / 2 - 160, 10), subtitle,
                  font=subtitlefont, fill="black")  # Centralized subtitle

        combined_image.show()

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            filename = subtitle.replace(" ", "_") + f"_{i}.png"
            save_path_file = os.path.join(save_path, filename)
            combined_image.save(save_path_file, 'PNG')


def test_visualization(model, loader, mask, device, subtitle, save_path, num_images=3):
    model.eval()
    with torch.no_grad():
        for images, true_masks in loader:
            images, true_masks = images.to(device), true_masks.to(device)
            predictions = model(images, mask)
            predictions = predictions.reshape(-1, 3, 224, 224)
            pred_masks = predictions.argmax(dim=1)
            visualize_segmentation_comparison(
                images, true_masks, pred_masks, num_images, subtitle=subtitle, save_path=save_path)
            break  # Remove or modify this line to process more batches
