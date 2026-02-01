import argparse
import numpy as np
import matplotlib.pyplot as plt

import random

def visualize_data(text, images):
    """
    Visualizes data with text and a list of images.
    
    Args:
        text (str): Description text to display.
        images (list): List of 6 PIL images [Left_t, Right_t, Hand_t, Left_next, Right_next, Hand_next].
    """
    if len(images) != 6:
        print(f"Warning: Expected 6 images, got {len(images)}")
        return

    # Unpack images
    img_left_t = images[0]
    img_right_t = images[1]
    img_hand_t = images[2]
    img_left_next = images[3]
    img_right_next = images[4]
    img_hand_next = images[5]

    # Visualization
    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Action Visualization", fontsize=16)

    # Define Grid: 3 Rows (Left, Right, Eye-in-Hand), 3 Columns (t, t+H, Info)
    
    # Row 1: Left Camera
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(img_left_t)
    ax1.set_title(f"AgentView Left (t)")
    ax1.axis('off')

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(img_left_next)
    ax2.set_title(f"AgentView Left (t+H)")
    ax2.axis('off')

    # Row 2: Right Camera
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(img_right_t)
    ax4.set_title(f"AgentView Right (t)")
    ax4.axis('off')

    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(img_right_next)
    ax5.set_title(f"AgentView Right (t+H)")
    ax5.axis('off')

    # Row 3: Eye-in-Hand Camera
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(img_hand_t)
    ax7.set_title(f"Eye-in-Hand (t)")
    ax7.axis('off')

    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(img_hand_next)
    ax8.set_title(f"Eye-in-Hand (t+H)")
    ax8.axis('off')

    # Text Info on the Right Column
    ax_info = fig.add_subplot(1, 3, 3)
    ax_info.axis('off')
    
    ax_info.text(0, 0.9, text, fontsize=12, verticalalignment='top', fontfamily='monospace', wrap=True)

    plt.tight_layout()
    plt.show()