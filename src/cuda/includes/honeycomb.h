#ifndef HONEYCOMB
#define HONEYCOMB

#include <stdint.h>

typedef uint32_t DartIdType;
typedef uint32_t VertexIdType;
typedef uint32_t EdgeIdType;
typedef uint32_t FaceIdType;
typedef uint32_t VolumeIdType;

struct CuVertex2 {
    float data[2];
};

struct CuVertex3 {
    float data[3];
};

#endif // HONEYCOMB

