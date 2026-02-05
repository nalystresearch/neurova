# copyright (c) 2025 squid consultancy group (scg)
# all rights reserved.
# licensed under the apache license 2.0.

"""
Haar cascade loader - parses XML cascade files and loads into C++ native detector.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any, Optional


def parse_cascade_xml(xml_path: str) -> Optional[Dict[str, Any]]:
    """Parse Haar cascade XML file.
    
    Args:
        xml_path: Path to the cascade XML file
        
    Returns:
        Dictionary with cascade data or None if parsing fails
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Failed to parse XML: {e}")
        return None
    
    # find cascade element
    cascade = root.find('.//cascade')
    if cascade is None:
        cascade = root.find('./cascade')
    if cascade is None:
        print("No cascade element found")
        return None
    
    # get window size
    width_elem = cascade.find('width')
    height_elem = cascade.find('height')
    window_width = int(width_elem.text) if width_elem is not None else 24
    window_height = int(height_elem.text) if height_elem is not None else 24
    
    # parse stages
    stages_elem = cascade.find('stages')
    if stages_elem is None:
        print("No stages element found")
        return None
    
    # parse features separately (cascade format stores them separately)
    features_elem = cascade.find('features')
    all_features = []
    
    if features_elem is not None:
        for feat in features_elem:
            rects_elem = feat.find('rects')
            if rects_elem is None:
                continue
            
            feature_rects = []
            for rect in rects_elem:
                # format: "x y w h weight"
                parts = rect.text.strip().split()
                if len(parts) >= 5:
                    feature_rects.append({
                        'x': int(parts[0]),
                        'y': int(parts[1]),
                        'w': int(parts[2]),
                        'h': int(parts[3]),
                        'weight': float(parts[4].rstrip('.'))
                    })
            
            all_features.append(feature_rects)
    
    # parse stages
    stages = []
    for stage in stages_elem:
        stage_threshold_elem = stage.find('stageThreshold')
        if stage_threshold_elem is None:
            continue
        
        stage_threshold = float(stage_threshold_elem.text)
        
        # parse weak classifiers
        weak_classifiers = stage.find('weakClassifiers')
        if weak_classifiers is None:
            continue
        
        features = []
        for classifier in weak_classifiers:
            internal_nodes = classifier.find('internalNodes')
            leaf_values = classifier.find('leafValues')
            
            if internal_nodes is None or leaf_values is None:
                continue
            
            # parse internal nodes: "0 -1 feature_idx threshold"
            int_parts = internal_nodes.text.strip().split()
            if len(int_parts) < 4:
                continue
            
            feature_idx = int(int_parts[2])
            threshold = float(int_parts[3])
            
            # parse leaf values: "left_val right_val"
            leaf_parts = leaf_values.text.strip().split()
            if len(leaf_parts) < 2:
                continue
            
            left_val = float(leaf_parts[0])
            right_val = float(leaf_parts[1])
            
            # get feature rectangles
            if feature_idx < len(all_features):
                rects = all_features[feature_idx]
            else:
                rects = []
            
            features.append({
                'threshold': threshold,
                'left_val': left_val,
                'right_val': right_val,
                'rects': rects
            })
        
        if features:
            stages.append({
                'threshold': stage_threshold,
                'features': features
            })
    
    return {
        'window_width': window_width,
        'window_height': window_height,
        'stages': stages
    }


def load_cascade(xml_path: str) -> bool:
    """Load cascade XML and initialize native detector.
    
    Args:
        xml_path: Path to the cascade XML file
        
    Returns:
        True if loaded successfully
    """
    cascade_data = parse_cascade_xml(xml_path)
    if cascade_data is None:
        return False
    
    try:
        from neurova.face import haar_native
        
        success = haar_native.load_cascade(
            cascade_data['window_width'],
            cascade_data['window_height'],
            cascade_data['stages']
        )
        
        if success:
            info = haar_native.get_info()
            print(f"Loaded cascade: {info['num_stages']} stages, "
                  f"{info['window_width']}x{info['window_height']} window")
        
        return success
        
    except ImportError:
        print("haar_native module not available")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "example/data/haarcascades/haarcascade_frontalface_default.xml"
    
    data = parse_cascade_xml(path)
    if data:
        print(f"Window: {data['window_width']}x{data['window_height']}")
        print(f"Stages: {len(data['stages'])}")
        for i, stage in enumerate(data['stages'][:3]):
            print(f"  Stage {i}: {len(stage['features'])} features, threshold={stage['threshold']:.2f}")
    else:
        print("Failed to parse cascade")
