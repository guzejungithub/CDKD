import torch.nn as nn
import torch.nn.functional as F

class attention_manager(object):
    def __init__(self, model):        
        self.feature = []
        self.handler = []

        self.model = model.module
        

        self.register_hook(self.model)
            
        
    def register_hook(self, model):
        def get_features(_, inputs, outputs):

            self.feature.append(outputs)

        for name, layer in model.named_modules():
           if True:
                if name == "final_layer":                                    
                    handle = layer.register_forward_hook(get_features)
                    self.handler.append(handle)
                if name == "mlp_head_x":
                    handle = layer.register_forward_hook(get_features)
                    self.handler.append(handle)
                if name == "mlp_head_y":
                    handle = layer.register_forward_hook(get_features)
                    self.handler.append(handle)
                
    
    def remove_hook(self):
        for handler in self.handler:
            handler.remove()




class attention_manager_teacher(object):
    def __init__(self, model):        
        self.feature = []
        self.handler = []

        self.model = model.module
        

        self.register_hook(self.model)
            
        
    def register_hook(self, model):
        def get_features(_, inputs, outputs):

            self.feature.append(outputs)

        def get_features_print(_, inputs, outputs):
            
            self.feature.append(outputs)

 
        for name, layer in model.named_modules():

            if True:

                if name == "final_layer":                     
                    handle = layer.register_forward_hook(get_features)
                    self.handler.append(handle)
                if name == "mlp_head_x":                     
                    handle = layer.register_forward_hook(get_features_print)
                    self.handler.append(handle)
                if name == "mlp_head_y":                    
                    handle = layer.register_forward_hook(get_features)
                    self.handler.append(handle)
                
    
    def remove_hook(self):
        for handler in self.handler:
            handler.remove()