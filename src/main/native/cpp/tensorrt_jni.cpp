#include "tensorrt_jni.h"

#include <cstdio>

#include "wpi_jni_common.h"
#include "YOLOv11.h"

static JClass detectionClass;

extern "C" {

/**
 * @brief Setting up Tensorrt logger
*/
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // Only output logs with severity greater than warning
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv(reinterpret_cast<void **>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  detectionClass =
      JClass(env, "org/photonvision/tensorrt/TensorRTJNI$TensorRTResult");

  if (!detectionClass) {
    std::printf("Couldn't find class!");
    return JNI_ERR;
  }

  return JNI_VERSION_1_6;
}

static jobject MakeJObject(JNIEnv *env, const Detection &result) {
  jmethodID constructor =
      env->GetMethodID(detectionClass, "<init>", "(IIIIFI)V");

  // Actually call the constructor
  return env->NewObject(detectionClass, constructor, result.bbox.x,
                        result.bbox.y, result.bbox.x + result.bbox.width, result.bbox.y + result.bbox.height,
                        result.conf, result.class_id);
}

/*
 * Class:     org_photonvision_tensorrt_TensorRTJNI
 * Method:    create
 * Signature: (Ljava/lang/String;III)J
 */
JNIEXPORT jlong JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_create
  (JNIEnv *env, jclass, jstring javaString, jint numClasses, jint modelVer, jint coreNum)
{
    const char *nativeString = env->GetStringUTFChars(javaString, 0);
  std::printf("Creating for %s\n", nativeString);

  YOLOv11 *ret;
//   if (static_cast<ModelVersion>(modelVer) == ModelVersion::YOLO_V5) {
//     std::printf("Starting with version 5\n");
//     ret = new YoloV5Model(nativeString, numClasses, coreNum);
//   } else if (static_cast<ModelVersion>(modelVer) == ModelVersion::YOLO_V8) {
//     std::printf("Starting with version 8\n");
//     ret = new YoloV8Model(nativeString, numClasses, coreNum);
//   } else if (static_cast<ModelVersion>(modelVer) == ModelVersion::YOLO_V11) {
//     std::printf("Starting with version 11\n");
//     ret = new YoloV11Model(nativeString, numClasses, coreNum);
//   } else {
//     std::printf("Unknown version\n");
//     return 0;
//   }
  ret = new YOLOv11(nativeString, logger);
  env->ReleaseStringUTFChars(javaString, nativeString);
  return reinterpret_cast<jlong>(ret);
}

/*
 * Class:     org_photonvision_tensorrt_TensorRTJNI
 * Method:    setCoreMask
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_setCoreMask
  (JNIEnv *env, jclass, jlong ptr, jint coreMask) {
    return ptr;
  }

/*
 * Class:     org_photonvision_tensorrt_TensorRTJNI
 * Method:    destroy
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_destroy
  (JNIEnv *env, jclass, jlong ptr) {
    delete reinterpret_cast<YOLOv11 *>(ptr);
  }

/*
 * Class:     org_photonvision_tensorrt_TensorRTJNI
 * Method:    detect
 * Signature: (JJDD)[Lorg/photonvision/tensorrt/TensorRTJNI/TensorRTResult;
 */
JNIEXPORT jobjectArray JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_detect
  (JNIEnv *env, jclass, jlong detector_, jlong input_cvmat_ptr, jdouble nms_thresh, jdouble box_thresh) {
    YOLOv11 *yolo = reinterpret_cast<YOLOv11 *>(detector_);
    cv::Mat *input_img = reinterpret_cast<cv::Mat *>(input_cvmat_ptr);

//     DetectionFilterParams params{
//       .nms_thresh = nms_thresh,
//       .box_thresh = box_thresh,
//   };

  vector<Detection> objects;
  yolo->preprocess(*input_img);
  yolo->infer();

  yolo->postprocess(objects);
  yolo->draw(*input_img, objects);

  if (objects.size() < 1) {
    std::cout << "No objects detected" << std::endl;
    return nullptr;
  }

  jobjectArray jarr =
      env->NewObjectArray(objects.size(), detectionClass, nullptr);

  for (size_t i = 0; i < objects.size(); i++) {
    objects[i].bbox.x *= input_img->cols;
    objects[i].bbox.y *= input_img->rows;
    objects[i].bbox.width *= input_img->cols;
    objects[i].bbox.height *= input_img->rows;
    std::cout << "Detection: " << i << " "
              << "x: " << objects[i].bbox.x << " y: " << objects[i].bbox.y
              << " w: " << objects[i].bbox.width
              << " h: " << objects[i].bbox.height
              << " conf: " << objects[i].conf
              << " class_id: " << objects[i].class_id
              << std::endl;
    jobject obj = MakeJObject(env, objects[i]);
    env->SetObjectArrayElement(jarr, i, obj);
  }

  return jarr;
}

/*
 * Class:     org_photonvision_tensorrt_TensorRTJNI
 * Method:    isQuantized
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_isQuantized
  (JNIEnv *, jclass, jlong) {
    return JNI_TRUE;
  }

}