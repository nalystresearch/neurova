#!/usr/bin/env python3
# neurova library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)

"""
Step 1: Annotate Images

This script provides tools to annotate images with bounding boxes.
You can use it to mark objects in your images for training a detector.

How to use:
    python 01_annotate_images.py                    (Opens interactive mode)
    python 01_annotate_images.py --folder ./images  (Annotate all images in folder)
    python 01_annotate_images.py --image test.jpg   (Annotate single image)

Controls when using graphical mode:
    Left click and drag to draw a bounding box
    Press n for next image
    Press p for previous image
    Press u to undo last box
    Press s to save annotations
    Press q to quit
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import TRAIN_DIR, CLASS_NAMES


class BoundingBoxAnnotator:
    """
    A simple tool for drawing bounding boxes on images.
    Helps you mark where objects are located in your pictures.
    """
    
    def __init__(self, class_names=None):
        self.class_names = class_names or ['object']
        self.current_class = 0
        self.boxes = []
        self.drawing = False
        self.start_point = None
        self.current_box = None
    
    def annotate_image(self, image_path, output_dir=None):
        """
        Open an image and let you draw boxes around objects.
        
        Parameters:
            image_path: Where the image file is located
            output_dir: Where to save the annotation file
        
        Returns:
            A dictionary containing all the annotation information
        """
        output_dir = output_dir or image_path.parent.parent / "annotations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
# try to use cv2 for the graphical interface
        try:
            import cv2
            return self._annotate_nv(image_path, output_dir)
        except ImportError:
            return self._annotate_text(image_path, output_dir)
    
    def _annotate_nv(self, image_path, output_dir):
        """Annotate using the graphical interface."""
        import cv2
        
        image = cv2.imread(str(image_path))
        if image is None:
            print("Error: Could not load image: " + str(image_path))
            return None
        
        height, width = image.shape[:2]
        self.boxes = []
        self.drawing = False
        display = image.copy()
        
        window_name = "Annotate: " + image_path.name + " | Class: " + self.class_names[self.current_class]
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal display
            
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.start_point = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                display = image.copy()
# draw existing boxes
                for box in self.boxes:
                    cv2.rectangle(display, 
                                 (box['x'], box['y']),
                                 (box['x'] + box['w'], box['y'] + box['h']),
                                 (0, 255, 0), 2)
                    cv2.putText(display, box['class'], 
                               (box['x'], box['y'] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
# draw current box being drawn
                cv2.rectangle(display, self.start_point, (x, y), (0, 0, 255), 2)
            
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                x1, y1 = self.start_point
                x2, y2 = x, y
                
# calculate box coordinates
                bx = min(x1, x2)
                by = min(y1, y2)
                bw = abs(x2 - x1)
                bh = abs(y2 - y1)
                
# only add box if it is large enough
                if bw > 10 and bh > 10:
                    self.boxes.append({
                        'class': self.class_names[self.current_class],
                        'x': bx, 'y': by, 'w': bw, 'h': bh
                    })
                    class_name = self.class_names[self.current_class]
                    print("  Added box: " + class_name + " at position (" + str(bx) + ", " + str(by) + ", " + str(bw) + ", " + str(bh) + ")")
                
                display = image.copy()
                for box in self.boxes:
                    cv2.rectangle(display,
                                 (box['x'], box['y']),
                                 (box['x'] + box['w'], box['y'] + box['h']),
                                 (0, 255, 0), 2)
                    cv2.putText(display, box['class'],
                               (box['x'], box['y'] - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        print("")
        print("Annotating: " + str(image_path.name))
        print("   Controls: draw box, press c to change class, u to undo, s to save, q to quit")
        
        while True:
# show instructions on the image
            info_display = display.copy()
            cv2.putText(info_display, "Class: " + self.class_names[self.current_class], 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(info_display, "Boxes: " + str(len(self.boxes)),
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(info_display, "c=class u=undo s=save q=quit",
                       (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow(window_name, info_display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
# switch to next class
                self.current_class = (self.current_class + 1) % len(self.class_names)
                print("   Class changed to: " + self.class_names[self.current_class])
            elif key == ord('u'):
# remove last box
                if self.boxes:
                    removed = self.boxes.pop()
                    print("   Removed: " + removed['class'])
                    display = image.copy()
                    for box in self.boxes:
                        cv2.rectangle(display,
                                     (box['x'], box['y']),
                                     (box['x'] + box['w'], box['y'] + box['h']),
                                     (0, 255, 0), 2)
            elif key == ord('s'):
# save and exit
                annotation = self._create_annotation(image_path, width, height)
                self._save_annotation(annotation, image_path, output_dir)
                break
        
        cv2.destroyAllWindows()
        
        return self._create_annotation(image_path, width, height)
    
    def _annotate_text(self, image_path, output_dir):
        """Text-based annotation when cv2 is not available."""
        from PIL import Image
        
        with Image.open(image_path) as img:
            width, height = img.size
        
        print("")
        print("Annotating: " + str(image_path.name) + " (size: " + str(width) + "x" + str(height) + ")")
        print("   Enter bounding boxes manually since no graphical interface is available")
        print("   Format: class_name x y width height")
        print("   Type done when finished")
        print("")
        
        self.boxes = []
        
        while True:
            user_input = input("   Box (or done): ").strip()
            
            if user_input.lower() == 'done':
                break
            
            try:
                parts = user_input.split()
                if len(parts) == 5:
                    cls_name = parts[0]
                    x, y, w, h = map(int, parts[1:5])
                    self.boxes.append({
                        'class': cls_name,
                        'x': x, 'y': y, 'w': w, 'h': h
                    })
                    print("   Added: " + cls_name + " at (" + str(x) + ", " + str(y) + ", " + str(w) + ", " + str(h) + ")")
                else:
                    print("   Please use format: class_name x y width height")
            except ValueError:
                print("   Invalid input, please try again")
        
        annotation = self._create_annotation(image_path, width, height)
        self._save_annotation(annotation, image_path, output_dir)
        
        return annotation
    
    def _create_annotation(self, image_path, width, height):
        """Create a dictionary with all the annotation data."""
        return {
            'image': image_path.name,
            'width': width,
            'height': height,
            'annotated_at': datetime.now().isoformat(),
            'objects': [
                {
                    'class': box['class'],
                    'bbox': [box['x'], box['y'], box['w'], box['h']]
                }
                for box in self.boxes
            ]
        }
    
    def _save_annotation(self, annotation, image_path, output_dir):
        """Save annotation data to a JSON file."""
        output_path = output_dir / (image_path.stem + ".json")
        with open(output_path, 'w') as f:
            json.dump(annotation, f, indent=2)
        print("   Saved: " + str(output_path))


def annotate_folder(folder, class_names=None):
    """Annotate all images found in a folder."""
    annotator = BoundingBoxAnnotator(class_names=class_names)
    
# find all image files
    images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        images.extend(folder.glob(ext))
        images.extend(folder.glob(ext.upper()))
    
    images = sorted(images)
    
    if not images:
        print("Error: No images found in: " + str(folder))
        return
    
    print("")
    print("Found " + str(len(images)) + " images in " + str(folder))
    print("")
    
    for i, image_path in enumerate(images):
        print("")
        print("[" + str(i + 1) + "/" + str(len(images)) + "]")
        annotator.annotate_image(image_path)


def main():
    parser = argparse.ArgumentParser(description="Annotate images with bounding boxes")
    parser.add_argument("--folder", type=str, default=None,
                       help="Folder containing images to annotate")
    parser.add_argument("--image", type=str, default=None,
                       help="Single image to annotate")
    parser.add_argument("--classes", type=str, nargs='+', default=['object'],
                       help="Class names for annotation")
    
    args = parser.parse_args()
    
    print("")
    print("STEP 1: ANNOTATE IMAGES")
    print("")
    
# use provided classes or default ones
    class_names = args.classes if args.classes else CLASS_NAMES[1:]
    
    print("")
    print("Classes: " + str(class_names))
    
    if args.image:
# annotate single image
        annotator = BoundingBoxAnnotator(class_names=class_names)
        annotator.annotate_image(Path(args.image))
    elif args.folder:
# annotate all images in folder
        annotate_folder(Path(args.folder), class_names)
    else:
        # Default: use train images folder
        train_images = TRAIN_DIR / "images"
        if train_images.exists() and list(train_images.glob("*")):
            annotate_folder(train_images, class_names)
        else:
            print("")
            print("No images found in: " + str(train_images))
            print("")
            print("How to use:")
            print("   1. Add images to: " + str(train_images))
            print("   2. Run: python 01_annotate_images.py")
            print("")
            print("   Or specify folder:")
            print("   python 01_annotate_images.py --folder ./my_images")
    
    print("")
    print("Annotation complete!")
    print("")
    print("Next step: python 02_prepare_dataset.py")


if __name__ == "__main__":
    main()

# neurova library
# Copyright (c) 2026 Neurova Team
# licensed under the apache license 2.0
# @squid consultancy group (scg)
