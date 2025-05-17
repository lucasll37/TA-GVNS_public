"""
create_video.py

This script reads PNG frames from a specified directory, sorts them by frame number,
and creates a video (MP4 format) using imageio.
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import shutil
import imageio.v2 as imageio

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FRAME_DIR = "./tmp/berlin52"  # Directory containing frame images
SECONDS_PER_FRAME = 0.05      # Duration of each frame in the video  
# SECONDS_PER_FRAME = 0.1      # Duration of each frame in the video  

def get_frame_number(filename: str) -> int:
    """
    Extracts the frame number from a filename.
    
    Assumes filenames follow the pattern: <prefix>_<number>.png
    
    Args:
        filename (str): The filename to extract the number from.
        
    Returns:
        int: The extracted frame number.
    """
    return int(filename.split('_')[1].split('.')[0])

# -----------------------------------------------------------------------------
# Main Video Creation Logic
# -----------------------------------------------------------------------------
def main() -> None:
    """
    Main function that reads PNG frames from a directory, sorts them, and writes
    them to a video file in MP4 format.
    """
    # List all PNG filenames in the frames directory.
    filenames = [fn for fn in os.listdir(FRAME_DIR) if fn.endswith(".png")]
    filenames = sorted(filenames, key=get_frame_number)
    
    # Create a video writer.
    os.makedirs("./video", exist_ok=True)
    video_filename = f"./video/{FRAME_DIR.split('/')[-1]}.mp4"
    writer = imageio.get_writer(video_filename, fps=int(1/SECONDS_PER_FRAME), codec="libx264")
    
    # Process each frame and add to the video.
    for fn in filenames:
        path = os.path.join(FRAME_DIR, fn)
        img = imageio.imread(path)
        try:
            writer.append_data(img)
        except Exception as e:
            print(f"Error processing {fn}: {e}")
            continue

    writer.close()
    print(f"Video saved as: {video_filename}")
    
    # — Remover a pasta de frames e todo o seu conteúdo —
    # shutil.rmtree(FRAME_DIR)

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()