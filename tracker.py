from ultralytics import YOLO, checks, hub

hub.login('86c4ecf832e20d274aa7488d1946c45d836ccd3707')
model = YOLO('https://hub.ultralytics.com/models/6LDB83hmaCxMPV7xqea6')

results = model.track(source=0, show=True, tracker = "bytetrack.yaml")