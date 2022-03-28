# Lab 2. Object tracking

> **Warning:** Make sure you have performed the steps described in the Prerequisites section before beginning this lab.

## Overview

This Lab will walk you through step-by-step instructions on how to build AWS Panorama application, starting with importing existing People detection application, and extending it to People tracking. By completing this Lab, you will learn 1) how to import existing application to start quickly, 2) how to extend the object detection application to object tracking application by customizing application code, 3) how to run the application on notebooks or your local PC environment with Test Utility, and 4) how to deploy applications to real Panorama appliance devices programatically.

The previous Lab-1 covers how to create an Object detection application from scratch, in more step-by-step manner. While completing Lab-1 is not a strong requirement before this Lab, it is recommended to finish Lab-1 first to understand the basics.

**Test Utility** is a simulation environment for application developers. While it doesn't provide full compatibility with real hardware, it helps you develop Panorama application without real hardware, with quick development iteration time. For more details about the Test Utility, please refer to [this page](https://github.com/aws-samples/aws-panorama-samples/blob/main/docs/AboutTestUtility.md).

**panorama-cli** is a command-line utility to help Panorama application developers construct necessary components such as code node, model node, camera node, node packages, container image for code, and node graph structure. It also helps uploading packages to AWS Cloud before deploying application to the device. For more details about panorama-cli, please refer to [this page](https://github.com/aws/aws-panorama-cli).


## How to open and run notebook

This Lab uses SageMaker Notebook environment. 
1. Visit [SageMaker Notebooks instances page](https://console.aws.amazon.com/sagemaker/home#/notebook-instances) and find "PanoramaWorkshop". Click "Open JupyterLab". 
1. In the file browser pane in left hand side, locate "aws-panorama-immersion-day" >  "labs" > "2. Object tracking.ipynb", and double click it. Notebook opens.
1. Select the first cell, and hit Shift-Enter key to execute a single selected cell and move to next cell.


## Preparation

1. Hit **Shift-Enter**, and execute the first code cell **"Import libraries"**. This cell imports necessarily Python modules for this Lab.
    ``` python
    # Import libraries

    import sys
    import os
    import time
    import json
    import glob
    import tarfile

    import boto3
    import sagemaker
    import IPython
    import gluoncv

    sys.path.insert( 0, os.path.abspath( "../common/test_utility" ) )
    import panorama_test_utility
    ```

1. Execute next cell **Initialize variables and configurations**. This cell initializes some basic variables such as AWS account ID, region name, S3 bucket name, and SageMaker execution role ARN.

    ``` python
    # Initialize variables and configurations

    boto3_session = boto3.session.Session()
    sm_session = sagemaker.Session()

    account_id = boto3.client("sts").get_caller_identity()["Account"]
    region = boto3_session.region_name
    s3_bucket = sm_session.default_bucket()
    sm_role = sagemaker.get_execution_role()

    print( "account_id :", account_id )
    print( "region :", region )
    print( "s3_bucket :", s3_bucket )
    print( "sm_role :", sm_role )
    ```

## Start with "People detection" application

In this Lab, we start with importing existing People detection application. You can find existing application project files under `lab2/` directory.

1. Run "panorama-cli import-application". This command essentially replaces placeholder account-IDs in the directory names and JSON file contents, with your AWS account-ID.

    ``` python
    app_name = "lab2"

    !cd {app_name} && panorama-cli import-application
    ```

1. Let's preview the source code of application. Because we imported existing application, we already have the source code.

    ``` python
    code_package_name = f"{app_name}_code"
    code_package_version = "1.0"
    source_filename = f"./lab2/packages/{account_id}-{code_package_name}-{code_package_version}/src/app.py"

    panorama_test_utility.preview_text_file(source_filename)
    ```

1. Export 'yolo3_mobilenet1.0_coco' from GluonCV's model zoo.

    This Lab uses pre-trained model exported from GluonCV's model zoo. Please run following cell to export 'yolo3_mobilenet1.0_coco'.

    ``` python
    def export_model_and_create_targz( prefix, name, model ):
        os.makedirs( prefix, exist_ok=True )
        gluoncv.utils.export_block( os.path.join( prefix, name ), model, preprocess=False, layout="CHW" )

        tar_gz_filename = f"{prefix}/{name}.tar.gz"
        with tarfile.open( tar_gz_filename, "w:gz" ) as tgz:
            tgz.add( f"{prefix}/{name}-symbol.json", f"{name}-symbol.json" )
            tgz.add( f"{prefix}/{name}-0000.params", f"{name}-0000.params" )
            
        print( f"Exported : {tar_gz_filename}" )
        
    # Export object detection model. Reset the classes for human detection only.
    people_detection_model = gluoncv.model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    people_detection_model.reset_class(["person"], reuse_weights=['person'])
    export_model_and_create_targz( "models", "yolo3_mobilenet1.0_coco_person", people_detection_model )
    ```

    > Note: `reset_class()` used here in order to detect only people faster.

    Exported model data is saved under "models/" directory.
    About the pre-trained model more in detail, please see [this GluonCV page](https://cv.gluon.ai/model_zoo/detection.html).

1. Add the exported model data file and model descriptor file with "panorama-cli add-raw-model" command.

    We can use existing model descriptor file as-is. You can find it here `./lab2/packages/{account-id}-lab2_model-1.0/descriptor.json`. In this file, ML framework is specified as "MXNET", input data name as "data" and input data shape as [1, 3, 480, 600].

    ``` python
    model_package_name = f"{app_name}_model"
    model_package_version = "1.0"
    people_detection_model_name = "people_detection_model"

    !cd {app_name} && panorama-cli add-raw-model \
        --model-asset-name {people_detection_model_name} \
        --model-local-path ../models/yolo3_mobilenet1.0_coco_person.tar.gz \
        --descriptor-path packages/{account_id}-{model_package_name}-{model_package_version}/descriptor.json \
        --packages-path packages/{account_id}-{model_package_name}-{model_package_version}
    ```

1. Compile the model  

    'panorama_test_utility_compile.py' is the Test Utility "Compile Model" script. You can use this python script either on notebook environment or on regular command-line terminal.

    > Note: This Model compilation is needed just for Test Utility. For real hardware, the model compilation is done automatically as a part of application deployment process. 

    ``` python
    people_detection_model_data_shape = '{"data":[1,3,480,600]}'

    %run ../common/test_utility/panorama_test_utility_compile.py \
    \
    --s3-model-location s3://{s3_bucket}/panorama-workshop/{app_name} \
    \
    --model-node-name {people_detection_model_name} \
    --model-file-basename ./models/yolo3_mobilenet1.0_coco_person \
    --model-data-shape '{people_detection_model_data_shape}' \
    --model-framework MXNET
    ```

1. Run the People detection application with "Test Utility".

    ``` python
    video_filepath = "../../videos/TownCentreXVID.avi"

    %run ../common/test_utility/panorama_test_utility_run.py \
    \
    --app-name {app_name} \
    --code-package-name {code_package_name} \
    --py-file {source_filename} \
    \
    --model-package-name {model_package_name} \
    --model-node-name {people_detection_model_name} \
    --model-file-basename ./models/yolo3_mobilenet1.0_coco_person \
    \
    --camera-node-name lab2_camera \
    \
    --video-file {video_filepath} \
    --video-start 0 \
    --video-stop 10 \
    --video-step 1 \
    \
    --output-screenshots ./screenshots/%Y%m%d_%H%M%S
    ```

    Please confirm that you see following log in the output cell. The simulation should quickly finish because you specified `--video-stop 10`. This means simulation ends after 10 frames.

    ```
    Loading graph: ./lab1/graphs/lab2/graph.json
        :
        :
    Frame : 0
    media[0] : media.image.dtype=uint8, media.image.shape=(1080, 1920, 3)
    2022-03-14 22:35:22,234 INFO Found libdlr.so in model artifact. Using dlr from ./models/people_detection_model/yolo3_mobilenet1.0_coco_person-LINUX_X86_64/libdlr.so
    Frame : 1
    media[0] : media.image.dtype=uint8, media.image.shape=(1080, 1920, 3)
    Frame : 2
    media[0] : media.image.dtype=uint8, media.image.shape=(1080, 1920, 3)
        :
        :
    Frame : 10
    media[0] : media.image.dtype=uint8, media.image.shape=(1080, 1920, 3)
    Frame : 11
    Reached end of video. Stopped simulation.
    ```

1. View the generated screenshot.

    As you specified `--output-screenshots` in the previous command, video frames passed to `node.outputs.video_out.put()` are written as sequencially numbered PNG files under the specified directory. Running following cell, you can render the latest screenshot file in the output cell.

    ``` python
    # View latest screenshot image

    latest_screenshot_dirname = sorted( glob.glob( "./screenshots/*" ) )[-1]
    screenshot_filename = sorted( glob.glob( f"{latest_screenshot_dirname}/*.png" ) )[-1]

    print(screenshot_filename)
    IPython.display.Image( filename = screenshot_filename )    
    ```

    > FIXME : add screenshot here
    ![](images/people-detection-output.png)


## Extend to "People tracking" application

In this section, we will extend the People detection application to People tracking application. To track people in camera streams, we use bounding boxes of detected people. As people move as time goes, bounding boxes also move. To track those moving bounding boxes, we use `SimpleObjectTracker` class which is a simple pure-python implementation of object tracking based on bounding boxes.

1. Create a new python source file `simple_object_tracker.py` and define `SimpleObjectTracker` class in it.

    1. Using the file browser pane, browse to "lab1/lab2/packages/{account_id}-lab2_code-1.0/src/". You should see `app.py` already.

    1. From the menu bar, select "File" > "New" > "Text File". A new empty text file "untitled.txt" is created.

    1. From the menu bar, select "File" > "Rename File". Rename the text file to "simple_object_tracker.py".

    1. Edit the simple_object_tracker.py with code, and save.

        ``` python
        import math
        import uuid

        # A class which represents a tracked object
        class TrackedObject:
            
            def __init__( self, box, forget_timer ):
                self.box = box
                self.forget_timer = forget_timer
                self.mapped = False
                self.uuid = uuid.uuid4()

        # A class to track multiple objects using moving bounding boxes
        class SimpleObjectTracker:
            
            def __init__( self, tick_count_before_forget = 30 ):
                self.objs = []
                self.tick_count_before_forget = tick_count_before_forget
            
            # track objects for bounding boxes and return a list of TrackedObject corresponding to passed boxes
            def track( self, boxes ):
                
                mapped_objects = []
                
                # mark all objects unmapped
                for o in self.objs:
                    o.mapped = False
                
                # for each bounding box, check all tracked objects and choose best fit one
                for b in boxes:
                    
                    found = None
                    closest_distance2 = math.inf
                
                    for o in self.objs:
                        
                        # skip if this object is already mapped to other box
                        if o.mapped:
                            continue
                        
                        # skip if there is no intersection intersection
                        if (b[2] < o.box[0] or # left
                            b[0] > o.box[2] or # right
                            b[3] < o.box[1] or # above
                            b[1] > o.box[3]):  # below
                            continue

                        # choose closest object being tracked, using distances of top-left, bottom-right corners
                        distance2 = ((b-o.box)**2).sum()
                        if distance2 < closest_distance2:
                            found = o
                            closest_distance2 = distance2
                    
                    # if it seems the object is not being tracked, create new one
                    if found is None:                
                        found = TrackedObject( b, self.tick_count_before_forget )
                        self.objs.append(found)
                    
                    # update the tracked object
                    found.mapped = True
                    found.box = b
                    found.forget_timer = self.tick_count_before_forget

                    # add the found one in the result
                    mapped_objects.append(found)

                # count down forget timer of objects
                for o in self.objs:
                    o.forget_timer -= 1

                # forget objects
                self.objs = list( filter( lambda o: o.forget_timer > 0, self.objs) )
                
                assert len(mapped_objects) == len(boxes)
                
                return mapped_objects
        ```

        This `SimpleObjectTracker` class internally creates and maintains a list of `TrackedObject` to track objects. When `track()` is called with updated bounding boxes, it does following processes.
        
        1. For each new bounding box, iterate over the list of existing `TrackedObject` objects to find the closest TrackedObject which intersects the new bounding box, and has the shortest distance to the new bounding box with respect to the top-left and bottom-right corners.

        1. When there is no existing `TrackedObject` which intersects new bounding box, create new one.

        1. When TrackedObject is not mapped to new bounding boxes for a certain period (30 frames by default), consider it disappeared and remove it from the list.


1. Manually edit the app.py to use the SimpleObjectTracker class

    ``` python
    import numpy as np
    import cv2

    import panoramasdk

    import simple_object_tracker

    model_input_resolution = (600,480)        
    #box_color = (0,0,255)
    box_thickness = 1

    # application class
    class Application(panoramasdk.node):
        
        # initialize application
        def __init__(self):
            
            super().__init__()
            
            self.frame_count = 0
            
            # create a object tracker
            self.tracker = simple_object_tracker.SimpleObjectTracker()

        # run top-level loop of application  
        def run(self):
            
            while True:
                
                print( f"Frame : {self.frame_count}", flush=True )
                
                # get video frames from camera inputs 
                media_list = self.inputs.video_in.get()
                
                for i_media, media in enumerate(media_list):
                    print( f"media[{i_media}] : media.image.dtype={media.image.dtype}, media.image.shape={media.image.shape}", flush=True )

                    # pass the video frame, and get formatted data for model input
                    image_formatted = self.format_model_input(media.image)
                    #print( f"image_formatted : image_formatted.dtype={image_formatted.dtype}, image_formatted.shape={image_formatted.shape}", flush=True )
                    
                    # pass the formatted model input data, run people detection, and get detected bounding boxes
                    detected_boxes = self.detect_people( image_formatted )
                    #print( f"detected_boxes : len(detected_boxes)={len(detected_boxes)}, detected_boxes={detected_boxes}", flush=True )
                    
                    # pass the bounding boxes and track objects
                    mapped_objects = self.track_people(detected_boxes)
                    #print( f"mapped_objects : {mapped_objects}", flush=True )
                    
                    # render the detected bounding boxes on the video frame
                    self.render_boxes( media.image, detected_boxes, mapped_objects )
                    
                # put video output to HDMI
                self.outputs.video_out.put(media_list)
                
                self.frame_count += 1

        # convert video frame from camera to model input data
        def format_model_input( self, image ):
            
            # scale to resolution expected by the model
            image = cv2.resize( image, model_input_resolution )

            # uint8 -> float32
            image = image.astype(np.float32) / 255.0

            # [480,600,3] -> [1,3,480,600]
            B = image[:, :, 0]
            G = image[:, :, 1]
            R = image[:, :, 2]
            image = [[[], [], []]]
            image[0][0] = R
            image[0][1] = G
            image[0][2] = B
            
            return np.asarray(image)

        # run people detection, and return detected bounding boxes
        def detect_people( self, data ):
            
            detected_boxes = []
            
            model_node_name = "people_detection_model"
            score_threshold = 0.5
            klass_person = 0
            
            # call people detection model
            people_detection_results = self.call( {"data":data}, model_node_name )
            
            # None result means empty
            if people_detection_results is None:
                return detected_boxes
            
            classes, scores, boxes = people_detection_results

            assert classes.shape == (1,100,1)
            assert scores.shape == (1,100,1)
            assert boxes.shape == (1,100,4)
            
            # scale bounding box to 0.0 ~ 1.0 space
            def to_01_space( box ):
                return box / np.array([
                    model_input_resolution[0], 
                    model_input_resolution[1], 
                    model_input_resolution[0], 
                    model_input_resolution[1] 
                ])
            
            # gather bounding boxes to return
            for klass, score, box in zip( classes[0], scores[0], boxes[0] ):
                if klass[0] == klass_person:
                    if score[0] >= score_threshold:
                        box = to_01_space( box )
                        detected_boxes.append( box )

            return detected_boxes
        
        # track people
        def track_people( self, boxes ):
            
            mapped_objects = self.tracker.track(boxes)
            return mapped_objects
        
        # render bounding boxes
        def render_boxes( self, image, boxes, mapped_objects ):
            
            for box, obj in zip( boxes, mapped_objects ):
                
                colors = [
                    (255,   0,   0),
                    (153,  76,   0),
                    (153, 204,   0),
                    (  0, 204,   0),
                    (  0, 102, 204),
                    (  0,   0, 204),
                    (  0,   0, 102),
                    (102,   0, 204),
                    (204,   0, 204),
                    (204,   0, 102),
                    (255,   0,   0),
                    (153,  76,   0),
                    (153, 204,   0),
                    (  0, 204,   0),
                    (  0, 102, 204),
                    (  0,   0, 204),
                    (  0,   0, 102),
                    (102,   0, 204),
                    (204,   0, 204),
                    (204,   0, 102),
                ]
                
                box_color = colors[ int(obj.uuid) % len(colors) ]
                
                # scale 0.0-1.0 space to camera image resolution
                h = image.shape[0]
                w = image.shape[1]
                box = (box * np.array([ w, h, w, h ])).astype(int)
                
                # render red rectancle
                cv2.rectangle( 
                    image, 
                    tuple(box[0:2]),
                    tuple(box[2:4]),
                    color = box_color,
                    thickness = box_thickness, 
                    lineType = cv2.LINE_8,
                )

    app = Application()
    app.run()
    ```

    Let's check what are changed from previous version of app.py. Firstly we import simple_object_tracker which defines SimpleObjectTracker.

    ``` python
    import simple_object_tracker
    ```

    In the constructor of Application class, we create an instance of SimpleObjectTracker class.

    ``` python
    # create a object tracker
    self.tracker = simple_object_tracker.SimpleObjectTracker()
    ```

    In the main loop, after calling `Application.detect_people()`, we call `Application.track_people()`, with detected bounding boxes. It is essentially a thin wrapper of `SimpleObjectTracker.track()`.

    ``` python
    # pass the bounding boxes and track objects
    mapped_objects = self.track_people(detected_boxes)
    ```

    ``` python
    # track people
    def track_people( self, boxes ):
        mapped_objects = self.tracker.track(boxes)
        return mapped_objects
    ```

    In the `Application.render_boxes()`, we added an argument `mapped_objects` which is a list of `TrackedObject`. Using `TrackedObject.uuid` property, we render bounding boxes in different colors, so that we can make sure object tracking is working as expected.

    ``` python
    # render bounding boxes
    def render_boxes( self, image, boxes, mapped_objects ):
        :
    ```

1. Run the People tracking application with "Test Utility".

    We use the same panorama_test_utility_run.py script again, to run this application. Command line arguments are exactly same as previous time.

    ``` python
    video_filepath = "../../videos/TownCentreXVID.avi"

    %run ../common/test_utility/panorama_test_utility_run.py \
    \
    --app-name {app_name} \
    --code-package-name {code_package_name} \
    --py-file {source_filename} \
    \
    --model-package-name {model_package_name} \
    --model-node-name {people_detection_model_name} \
    --model-file-basename ./models/yolo3_mobilenet1.0_coco_person \
    \
    --camera-node-name lab2_camera \
    \
    --video-file {video_filepath} \
    --video-start 0 \
    --video-stop 10 \
    --video-step 1 \
    \
    --output-screenshots ./screenshots/%Y%m%d_%H%M%S
    ```

1. View the generated screenshot.

    As same as last time, let's see generated screenshot. Please confirm that you see bounding boxes in multiple colors. You can open different screenshot files in the same directory to confirm that the application used same color for same object across multple frames.

    ``` python
    # View latest screenshot image

    latest_screenshot_dirname = sorted( glob.glob( "./screenshots/*" ) )[-1]
    screenshot_filename = sorted( glob.glob( f"{latest_screenshot_dirname}/*.png" ) )[-1]

    print(screenshot_filename)
    IPython.display.Image( filename = screenshot_filename )
    ```

    > FIXME : add screenshot here
    ![](images/people-tracking-output.png)


## Run the people tracking application on real device

> Note: This section is only for people who provisioned a Panorama appliance device with the AWS account. If you don't have, please skip to "Conclusion".

(drafting)