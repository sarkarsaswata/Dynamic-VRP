import cv2
import numpy as np

def create_video(images, output_file, fps=30):
    """
    Create a video from a list of images.

    Parameters:
    - images: List of NumPy arrays representing images.
    - output_file: Output video file path.
    - fps: Frames per second for the video.

    Returns:
    - None
    """
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    for image in images:
        video_writer.write(image)

    video_writer.release()

# Example usage:
# Assuming you have a list of images called image_list and want to create a video called output_video.mp4
# with a frame rate of 24 fps.
# create_video(image_list, 'output_video.mp4', fps=24)
