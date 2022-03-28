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
