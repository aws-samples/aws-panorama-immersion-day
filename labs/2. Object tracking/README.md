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





(drafting)

