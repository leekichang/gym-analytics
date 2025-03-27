import cv2

def convert_avi_to_mp4(input_path, output_path):
    """Convert an AVI video file to MP4 format using OpenCV."""
    cap = cv2.VideoCapture(input_path)
    
    # Get the video's properties (frame width, height, and FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create a VideoWriter object for MP4 output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Read and write each frame to the new video file
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Video saved as {output_path}")
    
def get_format(files, video_name):
    """Get the format of the video file."""
    for f in files:
        if video_name+'.' in f:
            return f.split('.')[-1]
    return None

def get_data(results):
    confs = {k:[] for k in range(17)}
    datas = {k:[] for k in range(17)}
    for frame_idx, r in enumerate(results):
        for k, c in enumerate(r.keypoints.conf[0]):
            confs[k].append(c.cpu())
            datas[k].append(r.keypoints.data[0][k].cpu())
    return confs, datas 