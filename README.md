# Nvidia AI Specialist Certification

---

## **Title: Drone detection and identification using YOLOv5**

---

## **OverView of the Project:**

- **Opening Background Information**:

Recently, drone technology has rapidly advanced for commercial, military, and personal purposes. While these advancements unlock new possibilities, they also pose challenges such as unauthorized intrusion, illegal filming, and public safety threats. Consequently, detecting and identifying drones has become a critical task in the security domain. AI-based object detection models can serve as powerful tools to address these challenges.

- **General Description of the Current Project:**

This project aims to develop a drone detection and identification system using the YOLOv5 object detection model. By collecting data, including drone images, and training the model, the system will be designed to accurately detect drones and distinguish them from other objects in various environments. Furthermore, optimization and efficiency will be considered to enable real-time detection.

- **Proposed Idea for Enhancements to the Project:**
1. **Utilizing Data Augmentation for Varied Environmental Conditions:**Data augmentation techniques will be actively employed to enhance drone detection under various lighting conditions (day, night, cloudy), weather (rain, snow, fog), and angles (downward, upward). This will enable the model to handle diverse variables in real-world scenarios.
2. **Improving Initial Performance Through Transfer Learning:**A pre-trained YOLOv5 model will be fine-tuned with a drone-specific dataset. This approach will reduce training time and maximize initial performance by leveraging existing knowledge.

- **Value and Significance of this Project:**

Drone detection and identification technology can be applied in various fields, including airport security, military base protection, and privacy protection in civilian areas. This project provides an efficient and practical system to preemptively detect and respond to drone-related security threats, contributing to enhanced public safety and security.

- **Current Limitations:**

The current project faces the following limitations: First, there is a lack of high-quality, large-scale datasets for drone detection. Second, there is a potential performance drop in diverse environments. Lastly, additional optimization of the model is required to meet real-time processing demands.

- **Literature Review:**

Object detection technology has significantly advanced with the development of deep learning. Notably, the YOLO (You Only Look Once) model has gained attention for real-time detection and high accuracy. Recent studies have applied YOLOv3 and v4 models for drone detection, and this project aims to use YOLOv5 for more efficient drone detection. Additionally, data augmentation and transfer learning techniques have been proven effective in enhancing drone detection performance.

## **Image Acquisition Method:**

I downloaded the drone video from YouTube and secured the video needed for the project.

<aside>
💡 Youtube:  [https://www.youtube.com/](https://www.youtube.com/)

</aside>

### Youtube_Drone_Video

[https://drive.google.com/file/d/1FA2Cxm5uyVqalt1ZuCbBklemhMrUNlVK/view?usp=drive_link](https://drive.google.com/file/d/1FA2Cxm5uyVqalt1ZuCbBklemhMrUNlVK/view?usp=drive_link)

## **Learning Data Extraction and Learning Annotation**:

To learn from YOLOv5 with 640 resolution images, we first made the images into 640 x 640 resolution images.

### Video resolution adjustment:

[비디오 리사이저 - 온라인에서 무료로 비디오 해상도 변경](https://online-video-cutter.com/ko/resize-video)

![KakaoTalk_20241124_001100077.png](KakaoTalk_20241124_001100077.png)

DarkLabel, also known as Video/Image Labeling and Annotation Tool, was used to image or annotate images made at 640 x 640 resolution in frame units.

### DarkLabel.zip

[https://drive.google.com/file/d/1fERIfKXkshCXhUAZ7lkzrAiPM8M5fvwa/view?usp=drive_link](https://drive.google.com/file/d/1fERIfKXkshCXhUAZ7lkzrAiPM8M5fvwa/view?usp=drive_link)

![캡처.PNG](%25EC%25BA%25A1%25EC%25B2%2598.png)

In the DarkLabel program, an image can be converted into an image in units of frames. First, a 640 x 640 resolution image is selected through Open Video. After that, labeled frames only will have the check mark enabled but deactivate the check mark. After that, it is converted into an image in a folder called images through as images.

![KakaoTalk_20241118_181232351.png](KakaoTalk_20241118_181232351.png)

You can check that the image came into the images folder.

![캡처.PNG](%25EC%25BA%25A1%25EC%25B2%2598%201.png)

Now the converted image is annnotated through DarkLabel.

First, add classes through darklabel.yml before annnotation.

![캡처.PNG](%25EC%25BA%25A1%25EC%25B2%2598%202.png)

Create my_classes2 in the yml file and add a drone for the class name.

```
my_classes2: ["drone"]
```

Now, in order to see the classes set in the Dark Label GUI during annnotation, classes_set puts the pre-set my_classes2 and sets the ball name in the GUI to darknet yolo.

```
format1:    # darknet yolo (predefined format]
  fixed_filetype: 1                 # if specified as true, save setting isn't changeable in GUI
  data_fmt: [classid, ncx, ncy, nw, nh]
  gt_file_ext: "txt"                 # if not specified, default setting is used
  gt_merged: 0                    # if not specified, default setting is used
  delimiter: " "                     # if not spedified, default delimiter(',') is used
  classes_set: "my_classes2"     # if not specified, default setting is used
  name: "darknet yolo"           # if not specified, "[fmt%d] $data_fmt" is used as default format name
```

It can be seen that a class called darknet yolo has been added to the DarkLabel program and one class has been added at the bottom.

![캡처.PNG](%25EC%25BA%25A1%25EC%25B2%2598%203.png)

From Dark Label, the converted image was retrieved by selecting the images folder through the Open Images Folder. After selecting Box + Label, annnotation was made in the drone that matches the class as shown in the picture below. After annnotation was completed, a folder called labels was created through GT Save As and saved in the folder.

![KakaoTalk_20241118_181306518.png](KakaoTalk_20241118_181306518.png)

You can see that there is an annotated txt file in labels.

![캡처.PNG](%25EC%25BA%25A1%25EC%25B2%2598%204.png)

### dataset_lane.zip:

[https://drive.google.com/file/d/1_1SYJZbpsoWRAx6U3DLyROW3PtsmRSBZ/view?usp=drive_link](https://drive.google.com/file/d/1_1SYJZbpsoWRAx6U3DLyROW3PtsmRSBZ/view?usp=drive_link)

## Nvidia Jetson Nano **Learning Course**:

Install Google Colaboratory on Google Drive.

![캡처.PNG](%25EC%25BA%25A1%25EC%25B2%2598%205.png)

Enter the code to connect to Google Drive in the command prompt:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### YOLOv:

Access yolov5 on Google Drive, download it, and install all the necessary libraries in bulk using the Requirements.txt file.

```python
# If you have already completed the installation, you only need to move to the corresponding path.
%cd /content/drive/MyDrive/yolov5

#clone YOLOv5 and
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies

!pip install Pillow==10.3
```

A folder named yolov5 is created, and inside it there is a Val file. Place the photos and txt files into the images and labels folders created in DarkLabel, respectively. Then, modify the data.yaml file according to the classes.

```
nc: 1
names:  # Classes name
- Drone

train: /content/drive/MyDrive/yolov5/Train/images  # train images 경로
val: /content/drive/MyDrive/yolov5/Val/images # val images 경로
```

### **Image File Management:**

We create folders to manage image files.

```python
!mkdir -p Train/labels
!mkdir -p Train/images
!mkdir -p Val/labels
!mkdir -p Val/images
```

After creating the label folder and image folder, select the yolov5 folder and place the images in each folder. Then, put the **yolov5n.pt** file provided with the guide and the **data.yaml** file modified to match the names of the labeled classes into the yolov5 folder.

```python
##데이터를 학습용 : 검증용 7:3 검증 데이터 만들기
import os
import shutil
from sklearn.model_selection import train_test_split

def create_validation_set(train_path, val_path, split_ratio=0.3):
  """
  Train 데이터의 일부를 Val로 이동
  """
  #필요한 디렉토리 생성
  os.makedirs(os.path.join(val_path, 'images'), exist_ok=True)
  os.makedirs(os.path.join(val_path, 'labels'), exist_ok=True)

  #Train 이미지 리스트 가져오기
  train_images = os.listdir(os.path.join(train_path, 'images'))
  train_images = [f for f in train_images if f.endswith(('.jpg','.jpeg','.png'))]

  #Train 이미지를 Train, Val로 분할
  _, val_images = train_test_split(train_images, test_size=split_ratio, random_state=42)

  #Val로 파일 복사
  for img in val_images:
    #이미지 복사
    src_image = os.path.join(train_path, 'images', img)
    dst_image = os.path.join(val_path, 'images', img)
    shutil.copy(src_image, dst_image)
    # 라벨 파일 복사
    label_file = os.path.splitext(img)[0] + '.txt'
    src_label = os.path.join(train_path, 'labels', label_file)
    dst_label = os.path.join(val_path, 'labels', label_file)
    if os.path.exists(src_label):
      shutil.copy(src_label, dst_label)

  print(f"Validation set created with {len(val_images)} images.")

#실행
train_path = '/content/drive/MyDrive/yolov5/Train'
val_path = '/content/drive/MyDrive/yolov5/Val'

create_validation_set(train_path, val_path)

def check_dataset():
  train_path = '/content/drive/MyDrive/yolov5/Train'
  val_path = '/content/drive/MyDrive/yolov5/Val'

  #Train 데이터셋 확인
  train_images = len(os.listdir(os.path.join(train_path, 'images')))
  train_labels = len(os.listdir(os.path.join(train_path, 'labels')))

  #Val 데이터셋 확인
  val_images = len(os.listdir(os.path.join(val_path, 'images')))
  val_labels = len(os.listdir(os.path.join(val_path, 'labels')))

  print("Dataset status:")
  print(f"Train - Images: {train_images}, {train_labels}")
  print(f"Val - Images: {val_images}, Labels: {val_labels}")

check_dataset()
```

### **Start of YOLOv5 Model Training:**

Start learning the yolov5 model.

```python
!python train.py --img 640 --batch 16 --epochs 300 --data /content/drive/MyDrive/yolov5/data.yaml --weights yolov5n.pt --cache
```

Run `train.py`, a Python script to train the YOLOv5 model. Various options are included as follows.

- **`!python train.py`:**Runs the `train.py` script to start training the YOLOv5 model.
- **`-img 640`:**Sets the input image size to 640x640 pixels for training.
- **`-batch 16`:**Specifies the batch size. Each training iteration processes 16 images.
- **`-epochs 300`:**Defines the number of training epochs. The model will go through the dataset 300 times.
- **`-data /content/drive/MyDrive/yolov5/data.yaml`:**Points to the data configuration file, which includes dataset paths, class names, and train/validation splits.
- **`-weights yolov5n.pt`:**Uses pre-trained weights from the YOLOv5n (nano) model, optimized for speed and efficiency.
- **`-cache`:**Caches the dataset in memory to speed up training.

You can check multiple data as shown below.There are losses used during training, there are losses used during training.YOv5 is mainly used three losses functions.

1. **Objectness Loss (Obj):**

- Objectness loss shall determine whether each grid cells must include objects.This loss measures measurement is measured regardless of whether the object is exist regardless of whether the object exists.
- Objectness loss is calculated only for grid cells with real objects.That is, if there is no object in the grid cell, losses will be zero.

2. **Classification Loss (Cls):**

- Classification losses calculates the classification loss for the object.This loss is calculated only for grid cells, and is used to accurately classify the class with the object.

3. **Localization Loss (Box):**

- Localization loss is measured differences between coordinate and the coordinates of the predicted bar of the predicted bar.This loss is induced to be closer to the actual object.

These losses are optimized for YOv5, and final loss is defined as the sum of these three losses.In the training process, these losses are used to update the weight of the model through reverse wave.Based on these results, YOv5 models are learning to perform more better performance to perform object detection tasks.

![KakaoTalk_20241120_234920992.png](KakaoTalk_20241120_234920992.png)

After the training is complete, you can see that exp has been trained in the train folder inside the yolov5 folder.

![캡처.PNG](%25EC%25BA%25A1%25EC%25B2%2598%206.png)

The learning outcomes are as follows.

### **exp:**

[15l9xqikZVF1pWbJ1Nb4uZIyYsLS7JkB2?usp=drive_link](https://drive.google.com/drive/folders/15l9xqikZVF1pWbJ1Nb4uZIyYsLS7JkB2?usp=drive_link)

**labels_correlogram**

![labels_correlogram.jpg](labels_correlogram.jpg)

**labels**

![labels.jpg](labels.jpg)

**F1_curve**

![F1_curve.png](F1_curve.png)

**PR_curve**

![PR_curve.png](PR_curve.png)

**P_curve**

![P_curve.png](P_curve.png)

**R_curve**

![R_curve.png](R_curve.png)

**results**

![results.png](results.png)

**train_batch**

![train_batch0.jpg](train_batch0.jpg)

**val_batch**

![val_batch0_labels.jpg](val_batch0_labels.jpg)

### **Validation of YOLOv5 Model Training Results:**

After learning, work is needed to check the learning results. There is a weights file in the lane file. Among them, use `best.pt` to go to `detect.py` to check the learning results.

```python
!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source /content/drive/MyDrive/yolov5/Train/images
```

- **`!python detect.py`:**Executes the `detect.py` script to perform object detection using YOLOv5.
- **`-weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt`:**Specifies the weights file to be used for detection. In this case, it uses the `best.pt` file saved from training.
- **`-img 640`:**Sets the input image size to 640x640 pixels. All input images are resized to this dimension.
- **`-conf 0.1`:**Sets the confidence threshold for object detection. Only objects with confidence scores of 10% or higher will be displayed in the results.
- **`-source /content/drive/MyDrive/yolov5/Train/images`:**Specifies the source path for the images or videos where detection will be performed. Here, the detection will run on images located in the `Train/images` folder.

**detect.py execution image:**

![KakaoTalk_20241120_235200837.png](KakaoTalk_20241120_235200837.png)

**Learning outcomes with detect.py:**

![00000000.png](00000000.png)

![00000161.png](00000161.png)

![00000060.png](00000060.png)

![00000297.png](00000297.png)

**detect.py execution video:**

```python
!python detect.py --weights /content/drive/MyDrive/yolov5/runs/train/exp/weights/best.pt --img 640 --conf 0.1 --source /content/drive/MyDrive/test/Drone.mp4
```

[Drone_final.mp4](Drone_final.mp4)

**lane_detect(video)**

[https://drive.google.com/file/d/1n3G9zoo-Tjt1hSlWuAnp31GYahRKYRr8/view?usp=drive_link](https://drive.google.com/file/d/1n3G9zoo-Tjt1hSlWuAnp31GYahRKYRr8/view?usp=drive_link)

---
