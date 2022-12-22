# DTU-U-net-course
**Special course in deep learning for medical image segmentation**
- 5-ECTS
- 3-Weeks

## Course description
The aim of this course is to implements, test and evaluate a complete software framework for segmenting anatomical structures in 3D medical scans.
The data used for this project is a public data set of 3D computed tomography cardiac scans with ground truth anatomical annotations (MM-WHS). Alternatively, a data set containing abdominal structures can be used.

## Learning objectives

After the course, the student can:
- Describe the nature of 3D computed tomography scans including spatial resolution, inter-slice distance and Hounsfield units.
- Describe the concept of anatomical annotations
- Use 3D slicer to visualize 3D medical data including annotations and segmentation results
- Describe the U-net deep learning architecture including convolution and pooling
- Describe loss functions including mean squared error
- Implement a basic 2D U-net architecture in Pytorch
- Transfer code and data to the DTU Compute GPU cluster or to a dedicated GPU server
- Train and test a deep learning algorithm on the DTU Compute GPU cluster or on a dedicated GPU server
- Test a 2D U-net on an independent test set
- Compare the output of a 2D U-net with ground truth annotations using the DICE similarity measure
- Evaluate the quality of a segmentation using visual inspection and determine if the segmentation is anatomically plausible.

## Course material
- Selected material from **02456schedule**: https://docs.google.com/document/d/e/2PACX-1vRL1M_zEyzH8d4jKHltDauAYIPgtudXV0uRwTspy7i1mt2WcMaUms1H2RBcANxfFKfiMm4BbJ5cYL9C/pub
- Selected material from https://www.deeplearningbook.org/
- **U-net paper**: [U-net: Convolutional networks for biomedical image segmentation](https://arxiv.org/abs/1505.04597)
- **GPU Cluster Wiki**: https://itswiki.compute.dtu.dk/index.php/GPU_Cluster

## Teaching and supervision
The course is to some degree a group based self-study course where the supervisor will have daily meetings with the students. There will also be a teaching assistant associated to the course.

## Evaluation:
7-grade scale based on written report of approximately 15 pages written by the student group.

## Schedule 2023:
- Week 1: January 9. – 13.
- Week 2: January 16. – 20.
- Holidays: January 23. – 27.
- Week 3: January 30. – February 3.

## Preparations:
- Have a Python / Anaconda environment up and running. 
- Install and learn to use a code editor. Recommended Visual Studio Code.
- Do the following exercise:
   - https://github.com/RasmusRPaulsen/DTUImageAnalysis/tree/main/exercises/ex1-IntroductionToImageAnalysis



### Week 1:
- Monday : Week 1-2 from 02456schedule
- Tuesday : Week 1-2 from 02456schedule
- Wednesday: Visit to Rigshospitalet. 3D Slicer on abdominal data / heart data. Try TotalSegmentator.
- Thursday: Week 2-3 from 02456schedule
- Friday: Week 2-3 from 02456schedule

### Week 2:
- Monday : GPU Cluster intro. Week 2-3 from 02456schedule
- Tuesday : Week 2-3 from 02456schedule
- Wednesday: U-Net intro. Week 2-3 from 02456schedule
- Thursday: Week 3-4 from 02456schedule
- Friday: Week 3-4 from 02456schedule

### Week 3:
- Monday :  U-net startup. Data preparation
- Tuesday :  U-net implementation, training, validation and testing
- Wednesday: U-net implementation, training, validation and testing
- Thursday: U-net implementation, training, validation and testing
- Friday: U-net implementation, training, validation and testing

