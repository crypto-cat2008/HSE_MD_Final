# HSE_MD_Final
<p>The purpose of this project is to develop an ML-based application that discovers stains on medical equipment as part of cleaning validation process. Today, the validation is performed by trained technicians using visual inspection. </p> 
<p>Three models (YOLOv7, Faster R-CNN and Mask R-CNN) were trained to detect stains on custom dataset.</p>
<p>Two best performing models (YOLOv7 and Faster R-CNN) were selected to be used in the application. The application uses Flask web server to upload and process images with stains. It provides bounding boxes for each identified stain and a tag (dirty/clean). Docker file provided for building a Docker image for application deployment.</p>
