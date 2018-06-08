#project name
PROJECT(test)
#head file path
INCLUDE_DIRECTORIES(
/wzx_ssd/caffe/include
)
#source directory
AUX_SOURCE_DIRECTORY(/wzx_ssd/caffe/detectCpp DIR_SRCS)
#set environment variable
SET(TEST
${DIR_SRCS}
)
#set extern libraries
SET(LIBRARIES

)
#add executable file
ADD_EXECUTABLE(/wzx_ssd/caffe/detectCpp/bin/bin /wzx_ssd/caffe/detectCpp)
#add link library
TARGET_LINK_LIBRARIES(/wzx_ssd/caffe/detectCpp/bin/bin ${LIBRARIES})
 
