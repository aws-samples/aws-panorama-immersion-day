import numpy as np
import cv2

import panoramasdk

model_input_resolution = (600,480)        
box_color = (0,0,255)
box_thickness = 1

# application class
class Application(panoramasdk.node):
    
    # initialize application
    def __init__(self):
        
        super().__init__()
        
        self.frame_count = 0

    # run top-level loop of application  
    def run(self):
        
        while True:
            
            print(f"Frame : {self.frame_count}")
            
            # get video frames from camera inputs 
            media_list = self.inputs.video_in.get()
            
            for i_media, media in enumerate(media_list):
                print( f"media[{i_media}] : media.image.dtype={media.image.dtype}, media.image.shape={media.image.shape}", flush=True )

                image_formatted = self.format_model_input(media.image)
                #print(image_formatted.shape)
                
                detected_boxes = self.detect_people( image_formatted )
                #print(detected_boxes)
                
                self.render_boxes( media.image, detected_boxes )
                
            # put video output to HDMI
            self.outputs.video_out.put(media_list)
            
            self.frame_count += 1

    def format_model_input( self, image ):
        
        image = cv2.resize( image, model_input_resolution )

        image = image.astype(np.float32) / 255.0
        B = image[:, :, 0]
        G = image[:, :, 1]
        R = image[:, :, 2]

        image = [[[], [], []]]
        image[0][0] = R
        image[0][1] = G
        image[0][2] = B
        
        return np.asarray(image)

    def detect_people( self, data ):
        
        detected_boxes = []
        
        model_node_name = "people_detection_model"
        score_threshold = 0.5
        klass_person = 0
        
        people_detection_results = self.call( {"data":data}, model_node_name )
        
        classes, scores, boxes = people_detection_results

        assert classes.shape == (1,100,1)
        assert scores.shape == (1,100,1)
        assert boxes.shape == (1,100,4)
        
        def to_01_space( box ):
            return box / np.array([
                model_input_resolution[0], 
                model_input_resolution[1], 
                model_input_resolution[0], 
                model_input_resolution[1] 
            ])
        
        for klass, score, box in zip( classes[0], scores[0], boxes[0] ):
            if klass[0] == klass_person:
                if score[0] >= score_threshold:
                    box = to_01_space( box )
                    detected_boxes.append( box )

        return detected_boxes
    
    def render_boxes( self, image, boxes ):
        
        for box in boxes:
            
            h = image.shape[0]
            w = image.shape[1]
            box = (box * np.array([ w, h, w, h ])).astype(int)
            
            print(box)
            
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
