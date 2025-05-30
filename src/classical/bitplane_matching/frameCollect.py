import cv2
import sys

def read_video(filename):
    """returns video object with its properties"""
    video_name = filename.split('/')[-1].split('.')[0]
    video = cv2.VideoCapture(filename)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    videofps = int(video.get(cv2.CAP_PROP_FPS))
    videoframecount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    return (video_name, video, frame_width, frame_height, videofps, videoframecount)

def read_frame(video, skipFrame=0):
    """returns the next frame from the video object, or skip frames"""
    for skip in range(skipFrame+1):
        ok, frame = video.read()

        if not ok:
            print('Cannot read video file')
            break

    return (ok, frame)

def getFrames(filepath, skip=10, count=2, outfolder="testImages", name="frame", frametype="png"):
    """
    Retrives and saves frames from a video
    :param filepath: file path of video
    :param skip: number of frames to skip by
    :param count: how many frames total to save
    :param outfolder: where to save frames
    :param name: frame name
    :return: None
    """
    video_name, video, frame_width, frame_height, videofps, videoframecount = read_video(filepath)

    for x in range(count):
        ok, frame = read_frame(video, skipFrame=skip)
        if ok:
            cv2.imwrite(f"{outfolder}/{name}{x+1}.{frametype}", frame)
        else:
            print("EOF")
            break

def downSampleVideo(filepath, factor=0.5, outfolder="VIDEOS"):
    """Downsamples a video by the factor to reduce the resolution of the video."""
    video_name, video, frame_width, frame_height, videofps, videoframecount = read_video(filepath)
    print(f"Original Resolution: {frame_height}x{frame_width}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
    output_path = f"{outfolder}/{video_name}_resized{int(frame_height * factor)}.mp4"
    out = cv2.VideoWriter(output_path, fourcc, videofps, 
                          (int(frame_width * factor), int(frame_height * factor)))

    while True:
        ok, frame = video.read()
        if not ok:
            print("End of video reached.")
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (int(frame_width * factor), int(frame_height * factor)))

        # Write the resized frame to the output video
        out.write(resized_frame)

    # Release resources
    video.release()
    out.release()
    print(f"Resized video saved to {output_path}")


# if __name__ == "__main__":

#     timer = time.time()

#     # COMMAND STARTS HERE

#     #videopath = "VIDEOS/CityWalk.mp4"
#     #downSampleVideo(videopath)
#     #videopath = "VIDEOS/HouseTour.mp4"
#     #downSampleVideo(videopath)

#     #COMMAND ENDS HERE

#     totTime = time.time() - timer
#     print(f"Done, executed in {totTime:.2f} s")

