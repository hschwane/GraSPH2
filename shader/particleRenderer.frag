#version 450

in vec2 center;
in float radius;
in vec4 vel_color;

uniform float brightness;

out vec4 fragment_color;

void main()
{
    vec2 coord = (gl_FragCoord.xy - center) / radius;
    float distFromCenter = length(coord);

#ifdef PARTICLES_ROUND
    // make it round
    if(distFromCenter > 1.0)
        discard;
#endif

    vec4 color = vel_color;
    PARTICLE_FALLOFF(); // this is defined via preprocessor macros when compiling
	fragment_color = color*brightness;
}