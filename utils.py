import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def seg_visualize(data: tuple, color_background=[255, 255, 255], color_edge=[255, 0, 0], transparency=0.75):
    '''Visualize the image with the egde

    Args:
      data: (image,trimap) where images is (channels, width, height) and trimap is (width, height)
      color_background: RGB vales to cover the backgroung, i.e. trimap == 2
      color_edge: RGB vales to show the edge with transparency, i.e. trimap == 3
      transparency: the transparency factor for the edge
    '''

    image = data[0].numpy()
    trimap = data[1].numpy()

    # convert to image with (width,height,channels) as shape and unit8 0~255 as values
    image = (image * 255).astype(np.uint8)
    image = image.transpose((1, 2, 0))

    image_original = image.copy()

    # show background
    image[trimap == 2] = color_background

    # show edge
    blend = np.array(color_edge, dtype=np.uint8)
    mask = trimap == 3
    image[mask] = (transparency * image[mask] +
                   (1 - transparency) * blend).astype(np.uint8)

    # show the original image and the segmentated one
    plt.subplot(1, 2, 1)
    plt.imshow(image_original)
    plt.axis('off')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Segmentated Image')

    plt.show()


def create_segmentation_visual(image, mask, color_background, color_edge, transparency):
    segmentated_image = image.copy()
    segmentated_image[mask == 2] = color_background  # Apply background color
    edge_mask = mask == 3
    blend_color = np.array(color_edge, dtype=np.uint8)
    segmentated_image[edge_mask] = (
        transparency * segmentated_image[edge_mask] + (1 - transparency) * blend_color).astype(np.uint8)
    return segmentated_image


def visualize_segmentation_comparison(images, true_masks, pred_masks, num_images,
                                      color_background=[255, 255, 255],
                                      color_edge=[255, 0, 0],
                                      transparency=0.75,
                                      subtitle="Segmentation Comparison",
                                      save_path=None, dpi=300):
    # Adjust subplot grid for multiple images
    fig, axs = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))
    if num_images == 1:
        axs = [axs]  # Ensure axs is iterable for a single image case

    for i in range(num_images):
        image_np = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        image_np = (image_np * 255).astype(np.uint8)
        true_mask_np = true_masks[i].cpu().detach().numpy()
        pred_mask_np = pred_masks[i].cpu().detach().numpy()

        true_segmentated_image = create_segmentation_visual(
            image_np, true_mask_np, color_background, color_edge, transparency)
        pred_segmentated_image = create_segmentation_visual(
            image_np, pred_mask_np, color_background, color_edge, transparency)

        axs[i][0].imshow(image_np)
        axs[i][0].set_title("Original Image", fontsize=16)
        axs[i][1].imshow(true_segmentated_image)
        axs[i][1].set_title("True Segmentated Image", fontsize=16)
        axs[i][2].imshow(pred_segmentated_image)
        axs[i][2].set_title("Predicted Segmentated Image", fontsize=16)

        for ax in axs[i]:
            ax.axis('off')

    plt.suptitle(subtitle, fontsize=22, fontweight='bold',
                 y=1.01)  # Adjust y for better spacing
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)  # Create the directory if it does not exist
        filename = subtitle.replace(" ", "_") + ".png"
        save_path = os.path.join(save_path, filename)
        # Save with high resolution
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
    plt.show()


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
