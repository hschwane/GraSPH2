#version 450 core

layout(points) in;
layout(triangle_strip, max_vertices=4) out;

uniform mat4 modelView; // model-view matrix (view * model)
uniform mat4 projection; // projection matrix
uniform float sphereRadius; // radius of the spheres
uniform float spriteScale =1.1f; // increase for high fild of view, if spheres seem to be cut of on the edges

in vec3 sphereColor[];

flat out vec3 color;
flat out float radius;
flat out vec4 viewSphereCenter;
smooth out vec2 texcoord;
smooth out vec4 viewPosOnPlane;

// see https://github.com/tdd11235813/spheres_shader/tree/master/src/shader
// and: https://paroj.github.io/gltut/Illumination/Tutorial%2013.html
// as well as chapter 14 and 15
// for sphere imposter rendering
void main()
{
    // Output vertex position
    color = sphereColor[0];
    radius = sphereRadius;
    viewSphereCenter = modelView * gl_in[0].gl_Position;

    // build transform matrix to get the quad from camera origin to it's proper position in camera (view) coordinates
    // matrix is filled collum major
    // move(viewSphereCenter) * rot * move(radius)
    mat4 transform;
//    if(projection[3][3] > 0 )
//    {
//        // we are dealing with orthografic projection,
//        // only move it to the proper position
//        transform = mat4(   0.0f, 0.0f, 0.0f, 0.0f, //LEFT
//                            0.0f, 0.0f, 0.0f, 0.0f, //UP
//                            0.0f, 0.0f, 0.0f, 0.0f, //FORWARD
//                            viewSphereCenter.x + radius,
//                            viewSphereCenter.y + radius,
//                            viewSphereCenter.z + radius,
//                            1.0f);//POSITION
//    }
//    else
//    {
        // we deal with perspective projection
        // rotate to camera
        const vec3 direction = normalize(-viewSphereCenter.xyz);
        const vec3 upGuess = vec3(0,1,0);
        const vec3 left = cross(direction,upGuess);
        const vec3 up = cross(left,direction);

        transform = mat4(   left.x, left.y, left.z, 0.0f, //LEFT
                            up.x, up.y, up.z, 0.0f, //UP
                            direction.x, direction.y, direction.z, 0.0f, //FORWARD
                            viewSphereCenter.x + direction.x*radius,
                            viewSphereCenter.y+ direction.y*radius,
                            viewSphereCenter.z+direction.z*radius,
                            1.0f);//POSITION
//    }

    // Vertex 1
    texcoord = vec2(-1.0f,-1.0f) * spriteScale;
    viewPosOnPlane = viewSphereCentertransform * vec4( texcoord * radius, 0.0f, 1.0f);
    gl_Position = projection * viewPosOnPlane;
    EmitVertex();

    // Vertex 2
    texcoord = vec2(-1.0f,1.0f) * spriteScale;
    viewPosOnPlane = transform * vec4( texcoord * radius, 0.0f, 1.0f);
    gl_Position = projection * viewPosOnPlane;
    EmitVertex();

    // Vertex 3
    texcoord = vec2(1.0f,-1.0f) * spriteScale;
    viewPosOnPlane = transform * vec4( texcoord * radius, 0.0f, 1.0f);
    gl_Position = projection * viewPosOnPlane;
    EmitVertex();

    // Vertex 4
    texcoord = vec2(1.0f,1.0f) * spriteScale;
    viewPosOnPlane = transform * vec4( texcoord * radius, 0.0f, 1.0f);
    gl_Position = projection * viewPosOnPlane;
    EmitVertex();

    EndPrimitive();
}