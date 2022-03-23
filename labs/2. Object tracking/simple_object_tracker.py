import math
import uuid

class TrackedObject:
    
    def __init__( self, box, forget_timer ):
        self.box = box
        self.forget_timer = forget_timer
        self.mapped = False
        self.uuid = uuid.uuid4()
        print( f"New object : {self.box}", flush=True )

class SimpleObjectTracker:
    
    def __init__( self, tick_count_before_forget = 30 ):
        self.objs = []
        self.tick_count_before_forget = tick_count_before_forget
    
    def track( self, boxes ):
        
        self._begin_track()
        
        mapped_objects = []
        
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

                distance2 = ((b-o.box)**2).sum()
                if distance2 < closest_distance2:
                    found = o
                    closest_distance2 = distance2
                
            if found is None:
                
                found = TrackedObject( b, self.tick_count_before_forget )
                self.objs.append(found)
            
            # reset the forget-timer
            found.mapped = True
            found.forget_timer = self.tick_count_before_forget

            mapped_objects.append(found)

        self._end_track()
        
        return mapped_objects
            
    def _begin_track(self):
        
        # mark unmapped in this frame
        for o in self.objs:
            o.mapped = False
        
    def _end_track(self):
        
        # count down forget timer of objects
        for o in self.objs:
            o.forget_timer -= 1

        # forget objects
        self.objs = list( filter( lambda o: o.forget_timer > 0, self.objs) )
        
        print( f"Num tracked objects : {len(self.objs)}", flush=True )
        