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
        
