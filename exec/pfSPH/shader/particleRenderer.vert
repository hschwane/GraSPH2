#version 450

#include "math.glsl"

layout(location=0) in vec4 input_position;
layout(location=1) in vec4 input_velocity;

uniform mat4 model_view_projection;
uniform mat4 projection;
uniform vec2 viewport_size;
uniform float render_size;
uniform vec4 defaultColor;

out vec2 center;
out float radius;
out vec4 vel_color;

void main()
{
	gl_Position = model_view_projection * input_position;

    float size = render_size;

#ifdef PARTICLES_PERSPECTIVE
	gl_PointSize = viewport_size.y * projection[1][1] * size / gl_Position.w;
#else
    gl_PointSize = size;
#endif

#ifdef COLORCODE_VELOCITY
    if(iszero(input_velocity.xyz))
        vel_color = defaultColor;
    else
        vel_color = vec4(normalize(input_velocity.xyz), 1);
#else
    vel_color = defaultColor;
#endif

	center = (0.5 * gl_Position.xy/gl_Position.w +0.5) * viewport_size;
	radius = gl_PointSize / 2;
}