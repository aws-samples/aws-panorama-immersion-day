# AWS Panorama Immersion Day contents

## How to set up

1. Create a CloudFormation stack with [this template file](./cloudformation/cf-panorama-workshop.yaml).
    ```
    $ aws cloudformation create-stack \
    --template-body file://./cloudformation/cf-panorama-workshop.yaml \
    --capabilities CAPABILITY_IAM \
    --stack-name panorama-workshop-1
    ```
1. Visit [SageMaker Notebooks instances page](https://console.aws.amazon.com/sagemaker/home#/notebook-instances) and find "PanoramaWorkshop". Click "Open JupyterLab".
1. From the file browser pane in left hand side, find "aws-panorama-immersion-day" directory, and double click it to see the contents of this repository.
1. Open "setup-sm.ipynb". This notebook contains required environment setting steps.
1. From menu bar, select "Run" > "Run All Cells"


## How to clean up

1. Stop the "PanoramaWorkshop" instance from [SageMaker Notebooks instances page](https://console.aws.amazon.com/sagemaker/home#/notebook-instances).

1. Empty the contents of following S3 path.
    ```
    $ aws s3 rm s3://sagemaker-{region}-{account-id}/panorama-workshop --recursive
    ```

1. Delete the CloudFormation stack.
    ```
    $ aws cloudformation delete-stack --stack-name panorama-workshop-1
    ```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.
