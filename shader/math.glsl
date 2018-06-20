#pragma once

#include "mathConst.glsl"

// zero checks
bool iszero(float f)
{
    return (abs(f)<F_EPSILON);
}

bool iszero(vec2 v)
{
    return iszero(length(v));
}

bool iszero(vec3 v)
{
    return iszero(length(v));
}

// interpolation
float bilinear(float a, float b, float c, float d, float s, float t)
{
      float x = mix(a, b, t);
      float y = mix(c, d, t);
      return mix(x, y, s);
}

vec2 bilinear(vec2 a, vec2 b, vec2 c, vec2 d, float s, float t)
{
      vec2 x = mix(a, b, t);
      vec2 y = mix(c, d, t);
      return mix(x, y, s);
}

vec3 bilinear(vec3 a, vec3 b, vec3 c, vec3 d, float s, float t)
{
      vec3 x = mix(a, b, t);
      vec3 y = mix(c, d, t);
      return mix(x, y, s);
}

vec4 bilinear(vec4 a, vec4 b, vec4 c, vec4 d, float s, float t)
{
      vec4 x = mix(a, b, t);
      vec4 y = mix(c, d, t);
      return mix(x, y, s);
}



void bezier(out float result, out float tangent, float p1, float p2, float p3, float p4, float s)
{
    // using De Casteljau
    float p12 = mix(p1,p2,s);
    float p23 = mix(p2,p3,s);
    float p34 = mix(p3,p4,s);

    float p13 = mix(p12,p23,s);
    float p24 = mix(p23,p34,s);

    result = mix(p13,p24,s);
    tangent = normalize(p24-p13);
}

float bezier( float p1, float p2, float p3, float p4, float s)
{
    // using De Casteljau
    float p12 = mix(p1,p2,s);
    float p23 = mix(p2,p3,s);
    float p34 = mix(p3,p4,s);

    float p13 = mix(p12,p23,s);
    float p24 = mix(p23,p34,s);

    return mix(p13,p24,s);
}

void bezier(out vec2 result, out vec2 tangent, vec2 p1, vec2 p2, vec2 p3, vec2 p4, float s)
{
    // using De Casteljau
    vec2 p12 = mix(p1,p2,s);
    vec2 p23 = mix(p2,p3,s);
    vec2 p34 = mix(p3,p4,s);

    vec2 p13 = mix(p12,p23,s);
    vec2 p24 = mix(p23,p34,s);

    result = mix(p13,p24,s);
    tangent = normalize(p24-p13);
}

vec2 bezier( vec2 p1, vec2 p2, vec2 p3, vec2 p4, float s)
{
    // using De Casteljau
    vec2 p12 = mix(p1,p2,s);
    vec2 p23 = mix(p2,p3,s);
    vec2 p34 = mix(p3,p4,s);

    vec2 p13 = mix(p12,p23,s);
    vec2 p24 = mix(p23,p34,s);

    return mix(p13,p24,s);
}

void bezier(out vec3 result, out vec3 tangent, vec3 p1, vec3 p2, vec3 p3, vec3 p4, float s)
{
    // using De Casteljau
    vec3 p12 = mix(p1,p2,s);
    vec3 p23 = mix(p2,p3,s);
    vec3 p34 = mix(p3,p4,s);

    vec3 p13 = mix(p12,p23,s);
    vec3 p24 = mix(p23,p34,s);

    result = mix(p13,p24,s);
    tangent = normalize(p24-p13);
}

vec3 bezier( vec3 p1, vec3 p2, vec3 p3, vec3 p4, float s)
{
    // using De Casteljau
    vec3 p12 = mix(p1,p2,s);
    vec3 p23 = mix(p2,p3,s);
    vec3 p34 = mix(p3,p4,s);

    vec3 p13 = mix(p12,p23,s);
    vec3 p24 = mix(p23,p34,s);

    return mix(p13,p24,s);
}

void bezier(out vec4 result, out vec4 tangent, vec4 p1, vec4 p2, vec4 p3, vec4 p4, float s)
{
    // using De Casteljau
    vec4 p12 = mix(p1,p2,s);
    vec4 p23 = mix(p2,p3,s);
    vec4 p34 = mix(p3,p4,s);

    vec4 p13 = mix(p12,p23,s);
    vec4 p24 = mix(p23,p34,s);

    result = mix(p13,p24,s);
    tangent = normalize(p24-p13);
}

vec4 bezier( vec4 p1, vec4 p2, vec4 p3, vec4 p4, float s)
{
    // using De Casteljau
    vec4 p12 = mix(p1,p2,s);
    vec4 p23 = mix(p2,p3,s);
    vec4 p34 = mix(p3,p4,s);

    vec4 p13 = mix(p12,p23,s);
    vec4 p24 = mix(p23,p34,s);

    return mix(p13,p24,s);
}

void hermite(out float result, out float tangent, float p1, float t1, float p2, float t2, float s)
{
    bezier(result, tangent, p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}

float hermite( float p1, float t1, float p2, float t2, float s)
{
    return bezier( p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}

void hermite(out vec2 result, out vec2 tangent, vec2 p1, vec2 t1, vec2 p2, vec2 t2, float s)
{
    bezier(result, tangent, p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}

vec2 hermite( vec2 p1, vec2 t1, vec2 p2, vec2 t2, float s)
{
    return bezier( p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}

void hermite(out vec3 result, out vec3 tangent, vec3 p1, vec3 t1, vec3 p2, vec3 t2, float s)
{
    bezier(result, tangent, p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}

vec3 hermite( vec3 p1, vec3 t1, vec3 p2, vec3 t2, float s)
{
    return bezier( p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}

void hermite(out vec4 result, out vec4 tangent, vec4 p1, vec4 t1, vec4 p2, vec4 t2, float s)
{
    bezier(result, tangent, p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}

vec4 hermite( vec4 p1, vec4 t1, vec4 p2, vec4 t2, float s)
{
    return bezier( p1, p1 + 1.0/3.0 * t1, p2 + 1.0/3.0 * t2, p2, s);
}


// sprecial functions
float gaussian(float mu,float sigma2,float x)
{
    return 1/sqrt(2*PI*sigma2) * exp(- (x-mu)*(x-mu) / (2*sigma2));
}