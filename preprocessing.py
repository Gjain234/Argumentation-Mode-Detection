import xml.etree.ElementTree as ET
import numpy as np
import os

sentences = []
pathos_labels = []
ethos_labels = []
logos_labels = []
for dirname in os.listdir('v2.0/'):
    for filename in os.listdir('v2.0/' + dirname + '/'):
        tree = ET.parse('v2.0/' + dirname + '/' + filename)
        root = tree.getroot()
        for child in root:
            for val in child:
                sentences.append(val.text)
                if 'pathos' in val.attrib['type']:
                    pathos_labels.append(1)
                else:
                    pathos_labels.append(0)
                if 'ethos' in val.attrib['type']:
                    ethos_labels.append(1)
                else:
                    ethos_labels.append(0)
                if 'logos' in val.attrib['type']:
                    logos_labels.append(1)
                else:
                    logos_labels.append(0)
