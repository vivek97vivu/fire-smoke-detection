from ultralytics import YOLO

model = YOLO("/home/algosium/Downloads/fire_smoke/models/yolo/best.pt")

model.export(
    format="engine",  
    device=0,          
    half=True,        
    dynamic=True       
)