# Lab 1. Object detection

> **Warning:** Make sure you have performed the steps described in the Prerequisites section before beginning this lab.


## Overview

This Lab will walk you through step-by-step instructions on how to build AWS Panorama application, starting with "Hello World" application, then Video pass-through application, and finally People counter application. By completing this Lab, you will learn 1) how to create necessary components of Panorama application with panorama-cli command and text editor, 2) how to use panoramasdk module APIs, 3) how to run the application on notebooks or your local PC environment with Test Utility, and 4) how to deploy applications to real Panorama appliance devices.

**Test Utility** is a simulation environment for application developers. While it doesn't provide full compatibility with real hardware, it helps you develop Panorama application without real hardware, with quick development iteration time. For more details about the Test Utility, please refer to [this page](https://github.com/aws-samples/aws-panorama-samples/blob/main/docs/AboutTestUtility.md).

**panorama-cli** is a command-line utility to help Panorama application developers construct necessary components such as code node, model node, camera node, node packages, container image for code, and node graph structure. It also helps uploading packages to AWS Cloud before deploying application to the device. For more details about panorama-cli, please refer to [this page](https://github.com/aws/aws-panorama-cli).

## How to open and run notebook

This Lab uses SageMaker Notebook environment. 
1. Visit [SageMaker Notebooks instances page](https://console.aws.amazon.com/sagemaker/home#/notebook-instances) and find "PanoramaWorkshop". Click "Open JupyterLab". 
1. In the file browser pane in left hand side, locate "aws-panorama-immersion-day" >  "labs" > "1. Object detection.ipynb", and double click it. Notebook opens.
1. Select the first cell, and hit Shift-Enter key to execute a single selected cell and move to next cell.


## Preparation

1. Hit **Shift-Enter**, and execute the first code cell **"Import libraries"**. This cell imports necessarily Python modules for this Lab.
    ``` python
    # Import libraries

    import sys
    import os
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

## "Hello World!" application

In this Lab, we create a simplest one-line application "Hello World!". Even for such simple application, Panorama application requires code package, Dockerfile, and manifest file. "panorama-cli" helps us create these files.

1. Run "panorama-cli init-project". This command creates a top directory ("lab1") for the application, 3 sub-directories ( "assets", "graphs", "packages" ), and empty graph.json file under "lab1/graphs/lab1/".

    ``` python
    app_name = "lab1"

    !panorama-cli init-project --name {app_name}
    ```

1. Run "panorama-cli create-package --type Container". This command creates a code package "lab1_code" under "lab1/packages/". 

    ``` python
    code_package_name = f"{app_name}_code"
    code_package_version = "1.0"

    !cd {app_name} && panorama-cli create-package --type Container --name {code_package_name} --version {code_package_version}
    ```

1. Manually create a one-line python source code.

    1. Using the file browser pane, browse to "lab1/lab1/packages/{account_id}-lab1_code-1.0/src/". This directory is created by the previous step, but it is still empty.

    1. From the menu bar, select "File" > "New" > "Text File". A new empty text file "untitled.txt" is created.

    1. From the menu bar, select "File" > "Rename File". Rename the text file to "app.py".

    1. Edit the app.py with following single line of code, and save.

        ``` python
        print( "Hello World!", flush=True )
        ```

       > **Note:** On Panorama appliance devices, log outputs to sys.stdout / sys.stderr are automatically uploaded to CloudWatch Logs. But to make sure logs are uploaded immediately, flush() has to be called explicitly. Otherwise, logs are not uploaded until buffer becomes full. The argument flush=True indicates print() function calls flush() internally.

1. Manually edit the code node descriptor file, and specify the entry point of the application code.

    1. Using the file browser pane, browse to "lab1/lab1/packages/{account_id}-lab1_code-1.0/". Find "descriptor.json".

    1. Right-click the file, and select "Open With" > "Editor", to open this file with text editor.

    1. Replace '<entry_file_name_under_src>' with 'app.py'. The modified version of file should look like this:

        ``` json
        {
            "runtimeDescriptor": {
                "envelopeVersion": "2021-01-01",
                "entry": {
                    "path": "python3",
                    "name": "/panorama/app.py"
                }
            }
        }
        ```
1. Build container image with "panorama-cli build-container" command.

    Building docker image and adding it to the code package as the asset. Test Utility doesn't use the container image itself, but it uses graph & package information to run.
    
    ```
    !cd {app_name} && panorama-cli build-container --container-asset-name code --package-path packages/{account_id}-{code_package_name}-{code_package_version}
    ```

    >This process takes some time (~ 5 min)

1. Run the Hello World application with "Test Utility".

    'panorama_test_utility_run.py' is the Test Utility "Run" script. You can use this python script either on notebook environment or on regular command-line terminal.

    ```
    %run ../common/test_utility/panorama_test_utility_run.py \
    \
    --app-name {app_name} \
    --code-package-name {code_package_name} \
    --py-file {source_filename}
    ```

    You should see console output in the output cell.

    ```
    Hello World!
    ```

## Video pass-through application

(drafting)

## People detection application

(drafting)

## Deployment to real device

(drafting)

## Clean up

(drafting)

## Conclusion

(drafting)


