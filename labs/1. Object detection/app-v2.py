import panoramasdk

# application class
class Application(panoramasdk.node):
    
    # initialize application
    def __init__(self):
        
        super().__init__()
        
        self.frame_count = 0

    # run top-level loop of application  
    def run(self):
        
        while True:
            
            # get video frames from camera inputs 
            media_list = self.inputs.video_in.get()
            
            if self.frame_count==0:
                for i_media, media in enumerate(media_list):
                    print( f"media[{i_media}] : media.image.dtype={media.image.dtype}, media.image.shape={media.image.shape}", flush=True )

            # put video output to HDMI
            self.outputs.video_out.put(media_list)
            
            self.frame_count += 1

app = Application()
app.run()
