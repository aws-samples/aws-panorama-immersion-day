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

    (drafting)

