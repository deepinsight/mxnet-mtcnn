# MTCNN MXNET C++ Implementation

This is a C++ project to implement MXNET MTCNN, a perfect face detect algorithm, on different DL frameworks.

## install dependencies
```sh
# For linux(ubuntu)
sudo apt-get install -y libopenblas-dev liblapack-dev
sudo apt-get install -y libopencv-dev

# For Mac
brew install openblas
brew install opencv
```

## Build
```sh
mkdir -p build
cd build
cmake -DUSE_CUDA=0 ..
```

* make -j ${nproc}

## Run

### run picture
```sh
bin/test_picture -f <photo> -m models
```

### run camera
If the basic work is ready (build caffe/Mxnet/Tensorflow sucessfully) followed by above steps. You can run the test now.
### 1. Test on single picture:

	./test -f photo_fname [ -t DL_type] [-s] 
	  -f photo_fname  picture to be  detected
	  -t DL_type      DL frame: "caffe" , "mxnet"(default) or "tensorflow"
	  -s              Save face chop into jpg files

The new picture, which boxed face and 5 landmark points will be created and saved as "new.jpg"

### 2. Test on camera (DL Framework is caffe)

 	./run.sh


# Release History

### Version 0.1.0 - 2018-2-11
   
  * Modified readme file.  
  * Modified makefile.mk.  
  * Add run.sh script  

# Credit

### MTCNN algorithm

https://github.com/kpzhang93/MTCNN_face_detection_alignment

### MTCNN C++ on Caffe

https://github.com/wowo200/MTCNN

### MTCNN python on Mxnet

https://github.com/pangyupo/mxnet_mtcnn_face_detection

### MTCNN python on Tensorflow

FaceNet uses MTCNN to align face

https://github.com/davidsandberg/facenet

From this directory:

    facenet/src/align
