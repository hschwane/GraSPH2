#version 450 core

#include "math.glsl"

layout(location=0) in vec4 input_position; // positions where spheres are rendered
layout(location=1) in vec4 input_velocity; // vector field for color
layout(location=2) in float input_density; // scalar field for color

uniform vec3 defaultColor; // particle color in color mode 5
uniform uint colorMode; // 1: color by vector field direction, 2: color by vector field magnitude, 3: color by scalar field, 0: constant color
uniform float upperBound; // highest value of scalar field / vector field magnitude
uniform float lowerBound; // lowest value of scalar field / vector field magnitude

out vec3 sphereColor;

// see https://github.com/tdd11235813/spheres_shader/tree/master/src/shader
// and: https://paroj.github.io/gltut/Illumination/Tutorial%2013.html
// as well as chapter 14 and 15
// for sphere imposter rendering
void main()
{
	gl_Position = input_position;

    switch(colorMode)
    {
    case 1: // vector field direction
        if(iszero(vec4(input_velocity).xyz))
            sphereColor = defaultColor;
        else
        {
            sphereColor = 0.5f*normalize(vec4(input_velocity).xyz)+vec3(0.5f);
        }
        break;
    case 2: // vector magnitude
        float leng = smoothstep(lowerBound,upperBound,length(input_velocity));
        sphereColor = vec3((leng*2.0f) +0.3f, (leng) +0.1f, (0.5f*leng) +0.1f);
        break;
    case 3: // scalar
        float rho = smoothstep(lowerBound , upperBound, input_density);
        sphereColor = vec3((rho*2.0f) +0.3f, (rho) +0.1f, (0.5f*rho) +0.1f);
        break;
    case 0: // constant
    default:
        sphereColor = defaultColor;
        break;
    }
}