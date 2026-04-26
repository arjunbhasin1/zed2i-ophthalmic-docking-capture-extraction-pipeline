import cv2

for i in range(0, 10):
    cap = cv2.VideoCapture(i)
    ok, frame = cap.read()
    cap.release()
    if ok:
        print(f"Camera index {i} works, frame shape: {frame.shape}")


#activate environemtn
# source .venv/bin/activate

#run code
#python capture/find_camera.py

# everytime:
# cd zed_data_pipeline
# source .venv/bin/activate

#copying over files

# scp -r data arjunbhasin@192.168.64.2:~/zed_data_pipeline/
# cd ~/zed_data_pipeline
# ls data
# ls data/frames
# ls data/videos

