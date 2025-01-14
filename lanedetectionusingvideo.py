import cv2
import numpy as np

# Open input video
video = cv2.VideoCapture('road_car_view.mp4')

# Get video properties
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(video.get(cv2.CAP_PROP_FPS))

# Resize to a smaller resolution
output_width = frame_width // 2  # Reduce width by half
output_height = frame_height // 2  # Reduce height by half

# Define codec and create VideoWriter object (H.264 for high compression)
fourcc = cv2.VideoWriter_fourcc(*'X264')  # H.264 codec for .mp4 files
output_video = cv2.VideoWriter('compressed_output.mp4', fourcc, fps // 2, (output_width, output_height))

frame_count = 0
while True:
    ret, orig_frame = video.read()
    if not ret:
        break

    # Resize frame
    orig_frame = cv2.resize(orig_frame, (output_width, output_height))
    
    # Process only every 2nd frame to reduce FPS
    if frame_count % 2 == 0:
        frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        low_yellow = np.array([18, 94, 140])
        up_yellow = np.array([48, 255, 255])
        mask = cv2.inRange(hsv, low_yellow, up_yellow)
        edges = cv2.Canny(mask, 75, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, maxLineGap=250)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Write the processed frame to the output video
        output_video.write(frame)

        # Display the frames (Optional)
        cv2.imshow("Processed Frame", frame)

    frame_count += 1

    key = cv2.waitKey(25)
    if key == 27:  # ESC key to break the loop
        break

# Release resources
video.release()
output_video.release()
cv2.destroyAllWindows()
